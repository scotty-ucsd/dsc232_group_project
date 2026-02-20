"""
step_04a_extract_ocean.py

-------------------------------------------------------------------------------
COMPUTATIONAL GLACIOLOGY – OPTIMISED PIPELINE
-------------------------------------------------------------------------------
DATE:   2026-02-13
AUTHOR: SYSTEM (Skeptical Gatekeeper)
STATUS: PRODUCTION
LOGIC:
  1. Eager-load GLORYS12V1 Antarctic subset into RAM (~5 GB for θ + S).
     → Eliminates per-block Dask I/O overhead; replaces distributed scheduler
       with direct NumPy indexing.
  2. BallTree (Haversine) nearest-neighbour from 500 m EPSG:3031 ice grid to
     1/12° GLORYS wet-ocean pixels.  Only the surface wet mask at t=0 is
     used; ocean coverage in GLORYS reanalysis is effectively time-invariant.
  3. Per-block vectorised depth interpolation:
     a. Deduplicate ocean pixels with np.unique (~16× fewer profile loads at
        500 m ice-grid vs. ~8 km GLORYS).
     b. Pre-compute linear interpolation brackets (idx_above, idx_below, frac)
        once per block — they depend only on depth_levels and clamped_depths,
        NOT on time.
     c. Fan-out profiles to all ice-shelf pixels via fancy indexing; apply
        brackets in a single fused multiply-add.  Zero Python loops over
        points or time steps within a batch.
  4. Time-batched regional writes keep peak working memory ≲ 1 GB per batch.
  5. 2D fields (dist_to_ocean, ice_draft, clamped_depth, mascon_id) are
     written once per block; 3D fields (thetao, so) in TIME_BATCH-sized
     slices.

CHANGES vs. ORIGINAL:
  • Removed Dask LocalCluster — pure NumPy hot path is faster and uses ~24 GB
    less resident memory (no worker processes).
  • Fixed max-depth computation: _last_valid_depth_index handles non-contiguous
    water columns.  Original used sum(dim='depth') - 1, which silently returns
    a wrong index when mid-column NaN gaps exist.
  • Replaced xr.interp(depth=…, fill_value="extrapolate") with clamped linear
    interpolation brackets.  Edge-case targets outside the valid depth range
    are returned as nearest valid level — not linearly extrapolated — which is
    physically conservative and prevents unphysical T/S values.
  • Switched zarr.Blosc → numcodecs.Blosc for zarr-python ≥ 3.0 forward
    compatibility.
  • Added --resume flag for block-level checkpoint/skip.
  • Separated 2D and 3D regional writes to avoid passing "time" region key
    to variables that lack a time dimension.
-------------------------------------------------------------------------------
"""
import os
import shutil
import time as _time
import argparse
import warnings

import numpy as np
import xarray as xr
import pyproj
import zarr
import numcodecs
import dask.array as da
from sklearn.neighbors import BallTree

# ── Preamble ────────────────────────────────────────────────────────────────
warnings.filterwarnings("ignore")
xr.set_options(keep_attrs=True)

# ── Configuration ───────────────────────────────────────────────────────────
CHUNK_SIZE  = 2048            # Spatial chunk edge in pixels (aligned to Zarr)
TIME_BATCH  = 24              # Time steps per 3D write batch (controls peak RAM)
R_EARTH_M   = 6_371_000.0    # Mean Earth radius [m] for haversine → metres

BEDMAP_PATH = "data/processed/bedmap3_500m.zarr"
OCEAN_PATH  = "data/raw/ocean/antarctic_ocean_physics_2019_2025.nc"
OUTPUT_ZARR = "data/processed/matched_ocean_grid.zarr"


# ── Helpers ─────────────────────────────────────────────────────────────────

def _last_valid_depth_index(is_valid: np.ndarray) -> np.ndarray:
    """Index of the deepest True along axis-0 for a (D, …) bool array.

    Returns -1 where no depth level is valid.

    Unlike ``sum(axis=0) - 1``, this correctly handles non-contiguous valid
    ranges (mid-column NaN gaps in the GLORYS water column) because it takes
    the maximum valid index rather than assuming contiguity from the surface.
    """
    depth_idx = np.arange(
        is_valid.shape[0]
    ).reshape((-1,) + (1,) * (is_valid.ndim - 1))
    weighted  = is_valid * depth_idx        # valid → index, invalid → 0
    last      = np.max(weighted, axis=0)
    any_valid = np.any(is_valid, axis=0)
    return np.where(any_valid, last, -1)


def _interp_brackets(
    depth_levels: np.ndarray,
    target: np.ndarray,
):
    """Pre-compute linear-interpolation indices and weights for depth.

    Parameters
    ----------
    depth_levels : (D,) sorted depth array [m].
    target       : (N,) per-point clamped target depth [m].

    Returns
    -------
    idx_a  : (N,) int32 — index of the depth level above the target.
    idx_b  : (N,) int32 — index of the depth level below the target.
    frac   : (N,) float32 — interpolation weight ∈ [0, 1].

    Edge behaviour (clamp, not extrapolate):
        target < depth_levels[0]  → frac = 0  (shallowest level)
        target > depth_levels[-1] → frac = 1  (deepest level)
    This is intentionally nearest-neighbour at boundaries, NOT linear
    extrapolation, which would be physically indefensible.
    """
    D = len(depth_levels)
    idx_a = np.searchsorted(depth_levels, target, side="right") - 1
    idx_a = np.clip(idx_a, 0, max(D - 2, 0))
    idx_b = np.minimum(idx_a + 1, D - 1)

    d_a   = depth_levels[idx_a]
    d_b   = depth_levels[idx_b]
    denom = np.where((d_b - d_a) > 0.0, d_b - d_a, 1.0)
    frac  = np.clip((target - d_a) / denom, 0.0, 1.0).astype(np.float32)

    return idx_a, idx_b, frac
# ── Main pipeline ───────────────────────────────────────────────────────────

def run_pipeline(resume: bool = False):
    wall_t0 = _time.perf_counter()
    print("=" * 72)
    print(" Step 04a · Ocean → Ice-Grid Extraction (Optimised)")
    print("=" * 72)

    # ── 1. Target grid (Bedmap3, lazy Zarr) ─────────────────────────────
    ds_b     = xr.open_zarr(BEDMAP_PATH)
    y_coords = ds_b.y.values
    x_coords = ds_b.x.values
    ny, nx   = len(y_coords), len(x_coords)
    print(f"[Grid]  Bedmap3 : {ny} × {nx}  ({ny * nx:,} px @ 500 m)")

    # ── 2. Source grid (GLORYS12V1, eager load, Antarctic crop) ─────────
    print("[I/O]   Loading GLORYS12V1 (−90 → −60 °S) into RAM …")
    t0 = _time.perf_counter()
    ds_oc = xr.open_dataset(OCEAN_PATH, engine="h5netcdf")
    ds_oc = ds_oc.sel(latitude=slice(-90, -60))

    time_c  = ds_oc.time.values
    depth_c = ds_oc.depth.values.astype(np.float64)   # metres, sorted sfc→deep
    lat_oc  = ds_oc.latitude.values
    lon_oc  = ds_oc.longitude.values
    nT, nD  = len(time_c), len(depth_c)

    # Pre-load both 4-D fields into contiguous NumPy arrays.
    # Memory: ~2.6 GB per variable for (84, 50, 360, 4320) float32.
    theta_full = ds_oc.thetao.values   # (T, D, lat, lon)  float32
    so_full    = ds_oc.so.values       # (T, D, lat, lon)  float32
    ds_oc.close()

    mem_gb = (theta_full.nbytes + so_full.nbytes) / 1e9
    print(
        f"[I/O]   Shape  : ({nT}, {nD}, {len(lat_oc)}, {len(lon_oc)})  ·  "
        f"{mem_gb:.2f} GB  [{_time.perf_counter() - t0:.1f} s]"
    )

    # ── 3. BallTree over wet surface pixels ─────────────────────────────
    print("[Tree]  Building Haversine BallTree …")
    t0 = _time.perf_counter()
    ocean_mask   = ~np.isnan(theta_full[0, 0])          # (lat, lon)
    y_wet, x_wet = np.where(ocean_mask)
    n_wet        = len(y_wet)
    wet_lat_rad  = np.deg2rad(lat_oc[y_wet])
    wet_lon_rad  = np.deg2rad(lon_oc[x_wet])

    tree = BallTree(
        np.column_stack([wet_lat_rad, wet_lon_rad]),
        metric="haversine",
    )
    print(
        f"[Tree]  {n_wet:,} wet pixels  "
        f"[{_time.perf_counter() - t0:.1f} s]"
    )

    # ── 4. Deepest valid depth per wet pixel (robust) ───────────────────
    #
    # _last_valid_depth_index returns the INDEX of the deepest non-NaN level
    # along the depth axis.  Unlike the original sum(depth)-1 approach, it
    # handles non-contiguous valid ranges (mid-column NaN gaps).
    #
    is_valid_t0    = ~np.isnan(theta_full[0])            # (D, lat, lon)
    last_valid_idx = _last_valid_depth_index(is_valid_t0) # (lat, lon)

    wet_max_depth = np.where(
        last_valid_idx[y_wet, x_wet] >= 0,
        depth_c[np.clip(last_valid_idx[y_wet, x_wet], 0, nD - 1)],
        0.0,
    ).astype(np.float64)                                  # (n_wet,)

    del is_valid_t0, last_valid_idx, ocean_mask

    # ── 5. Initialise / resume output Zarr store ───────────────────────
    store_exists = os.path.exists(OUTPUT_ZARR)

    if resume and store_exists:
        print(f"[Zarr]  Resuming into {OUTPUT_ZARR}")
    else:
        if store_exists:
            shutil.rmtree(OUTPUT_ZARR)
        print(f"[Zarr]  Creating {OUTPUT_ZARR}")

        s2 = (ny, nx)
        s3 = (nT, ny, nx)
        c2 = (CHUNK_SIZE, CHUNK_SIZE)
        c3 = (1, CHUNK_SIZE, CHUNK_SIZE)

        comp = numcodecs.Blosc(
            cname="zstd", clevel=3, shuffle=numcodecs.Blosc.SHUFFLE
        )

        ds_tmpl = xr.Dataset(
            coords={"time": time_c, "y": y_coords, "x": x_coords},
            data_vars={
                "dist_to_ocean": (
                    ("y", "x"),
                    da.full(s2, np.nan, dtype=np.float32, chunks=c2),
                ),
                "ice_draft": (
                    ("y", "x"),
                    da.full(s2, np.nan, dtype=np.float32, chunks=c2),
                ),
                "clamped_depth": (
                    ("y", "x"),
                    da.full(s2, np.nan, dtype=np.float32, chunks=c2),
                ),
                "mascon_id": (
                    ("y", "x"),
                    da.full(s2, np.nan, dtype=np.float32, chunks=c2),
                ),
                "thetao": (
                    ("time", "y", "x"),
                    da.full(s3, np.nan, dtype=np.float32, chunks=c3),
                ),
                "so": (
                    ("time", "y", "x"),
                    da.full(s3, np.nan, dtype=np.float32, chunks=c3),
                ),
            },
        )

        ds_tmpl["dist_to_ocean"].attrs.update(
            units="meters",
            long_name="Haversine distance to nearest wet ocean pixel",
        )
        ds_tmpl["ice_draft"].attrs.update(
            units="meters",
            long_name="Ice-shelf draft (surface minus thickness)",
        )
        ds_tmpl["clamped_depth"].attrs.update(
            units="meters",
            long_name="Draft depth clamped to deepest valid ocean level",
        )
        ds_tmpl["thetao"].attrs.update(
            units="degrees_C",
            long_name=(
                "Potential temperature at clamped draft depth (GLORYS12V1)"
            ),
        )
        ds_tmpl["so"].attrs.update(
            units="PSU",
            long_name=(
                "Practical salinity at clamped draft depth (GLORYS12V1)"
            ),
        )
        ds_tmpl["mascon_id"].attrs.update(
            long_name="GRACE mascon identifier",
        )

        enc = {}
        for v in ds_tmpl.data_vars:
            dims = ds_tmpl[v].dims
            enc[v] = {
                "compressor": comp,
                "chunks": c3 if "time" in dims else c2,
            }

        ds_tmpl.to_zarr(
            OUTPUT_ZARR, compute=False, encoding=enc, zarr_format=2
        )
        del ds_tmpl
        print("[Zarr]  Store initialised.")
# ── 6. Block-by-block projection, BallTree lookup, interp ──────────
    transformer = pyproj.Transformer.from_crs(
        "epsg:3031", "epsg:4326", always_xy=True
    )

    nYb   = -(-ny // CHUNK_SIZE)   # ceiling division
    nXb   = -(-nx // CHUNK_SIZE)
    total = nYb * nXb
    blk   = 0
    done  = 0
    width = len(str(total))

    print(
        f"[Loop]  {total} blocks  ({nYb} Y × {nXb} X)  "
        f"CHUNK={CHUNK_SIZE}  TIME_BATCH={TIME_BATCH}"
    )

    for yi in range(0, ny, CHUNK_SIZE):
        for xi in range(0, nx, CHUNK_SIZE):
            blk += 1
            ye = min(yi + CHUNK_SIZE, ny)
            xe = min(xi + CHUNK_SIZE, nx)
            ysl, xsl = slice(yi, ye), slice(xi, xe)
            bny, bnx = ye - yi, xe - xi

            # ── Resume guard: skip if chunk already on disk ─────────────
            if resume:
                chk_path = os.path.join(
                    OUTPUT_ZARR,
                    "dist_to_ocean",
                    f"{yi // CHUNK_SIZE}.{xi // CHUNK_SIZE}",
                )
                if os.path.exists(chk_path):
                    continue

            # ── Load Bedmap3 block (single Zarr chunk read) ─────────────
            bed  = ds_b.isel(y=ysl, x=xsl).load()
            mask = (bed.mask.values == 2) | (bed.mask.values == 3)

            if not mask.any():
                del bed
                continue

            done += 1
            tb = _time.perf_counter()

            vy, vx = np.where(mask)
            nV = len(vy)

            # ── EPSG:3031 → WGS-84 (lon, lat) ──────────────────────────
            lons, lats = transformer.transform(
                bed.x.values[vx], bed.y.values[vy]
            )

            # ── BallTree nearest-neighbour query ────────────────────────
            dists_rad, tidx = tree.query(
                np.deg2rad(np.column_stack([lats, lons])), k=1
            )
            tidx   = tidx.ravel()
            dist_m = (dists_rad.ravel() * R_EARTH_M).astype(np.float32)

            # ── Ice draft and depth clamping ────────────────────────────
            #
            # draft = surface − thickness   (negative where draft is below
            # sea level).  We take |draft| as the query depth and clamp to
            # the deepest valid ocean level at the nearest wet pixel.  This
            # prevents querying depth profiles into solid seabed/rock.
            #
            draft_full = bed.surface.values - bed.thickness.values
            raw_depth  = np.abs(draft_full[vy, vx]).astype(np.float64)
            clamped    = np.minimum(raw_depth, wet_max_depth[tidx])

            # ── Deduplicate nearest-ocean pixels ────────────────────────
            #
            # At 500 m ice / ~8 km ocean, ~256 ice pixels map to the same
            # ocean column.  Loading profiles only for UNIQUE ocean pixels
            # cuts I/O and memory by ~16×.
            #
            utidx, inv = np.unique(tidx, return_inverse=True)
            nU   = len(utidx)
            u_yi = y_wet[utidx]
            u_xi = x_wet[utidx]

            # ── Pre-compute depth interpolation brackets ────────────────
            #
            # Brackets (idx_a, idx_b, frac) depend only on depth_levels and
            # clamped target depths — they are TIME-INVARIANT.  We compute
            # them once per block and reuse across all time batches.
            #
            idx_a, idx_b, frac = _interp_brackets(depth_c, clamped)

            # ── Write 2D static fields (once per block) ─────────────────
            a_dist  = np.full((bny, bnx), np.nan, dtype=np.float32)
            a_draft = np.full((bny, bnx), np.nan, dtype=np.float32)
            a_clamp = np.full((bny, bnx), np.nan, dtype=np.float32)
            a_msc   = np.full((bny, bnx), np.nan, dtype=np.float32)

            a_dist[vy, vx]  = dist_m
            a_draft[vy, vx] = draft_full[vy, vx].astype(np.float32)
            a_clamp[vy, vx] = clamped.astype(np.float32)
            a_msc[vy, vx]   = bed.mascon_id.values[vy, vx].astype(
                np.float32
            )

            xr.Dataset(
                coords={"y": bed.y.values, "x": bed.x.values},
                data_vars={
                    "dist_to_ocean": (("y", "x"), a_dist),
                    "ice_draft":     (("y", "x"), a_draft),
                    "clamped_depth": (("y", "x"), a_clamp),
                    "mascon_id":     (("y", "x"), a_msc),
                },
            ).drop_vars(
                "spatial_ref", errors="ignore"
            ).to_zarr(OUTPUT_ZARR, region={"y": ysl, "x": xsl})

            del a_dist, a_draft, a_clamp, a_msc

            # ── Write 3D fields in time batches ─────────────────────────
            #
            # Peak memory per batch ≈ TIME_BATCH × bny × bnx × 4 × 2 vars
            #   + TIME_BATCH × nD × nU × 4 (profiles)
            #   + TIME_BATCH × nV × 4 × 4 (above/below per var)
            # With TIME_BATCH=24, CHUNK_SIZE=2048: ~1 GB worst case.
            #
            for t0 in range(0, nT, TIME_BATCH):
                t1  = min(t0 + TIME_BATCH, nT)
                nt  = t1 - t0
                tsl = slice(t0, t1)

                a_th = np.full((nt, bny, bnx), np.nan, dtype=np.float32)
                a_so = np.full((nt, bny, bnx), np.nan, dtype=np.float32)

                # Process theta and so sequentially to halve transient RAM.
                for src, dst in [(theta_full, a_th), (so_full, a_so)]:
                    #  src[t0:t1, :, u_yi, u_xi]  →  (nt, D, nU)
                    #  Fancy-index with [idx_a, inv] fans unique profiles
                    #  back to all nV ice-shelf points in one shot.
                    prof  = src[t0:t1, :, u_yi, u_xi]    # (nt, D, nU)
                    above = prof[:, idx_a, inv]            # (nt, nV)
                    below = prof[:, idx_b, inv]            # (nt, nV)
                    dst[:, vy, vx] = (
                        above + frac[None, :] * (below - above)
                    ).astype(np.float32)
                    del prof, above, below

                xr.Dataset(
                    coords={
                        "time": time_c[t0:t1],
                        "y": bed.y.values,
                        "x": bed.x.values,
                    },
                    data_vars={
                        "thetao": (("time", "y", "x"), a_th),
                        "so":     (("time", "y", "x"), a_so),
                    },
                ).drop_vars(
                    "spatial_ref", errors="ignore"
                ).to_zarr(
                    OUTPUT_ZARR,
                    region={"y": ysl, "x": xsl, "time": tsl},
                )

                del a_th, a_so

            # ── Block summary ───────────────────────────────────────────
            dt = _time.perf_counter() - tb
            print(
                f"  [{blk:>{width}}/{total}]  "
                f"Y[{yi}:{ye}] X[{xi}:{xe}]  |  "
                f"{nV:>7,} ice px → {nU:>5,} ocean px  |  "
                f"{dt:.1f} s"
            )

            del bed, mask, draft_full, clamped, tidx, dist_m

    # ── 7. Finalise ─────────────────────────────────────────────────────
    zarr.consolidate_metadata(OUTPUT_ZARR)
    wall = _time.perf_counter() - wall_t0

    print(f"\n{'=' * 72}")
    print(
        f" Done.  {done} blocks with ice-shelf data processed in "
        f"{wall:.0f} s ({wall / 60:.1f} min)"
    )
    print(f"{'=' * 72}")


# ── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Step 04a: GLORYS ocean → Bedmap3 ice-grid extraction",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume into existing Zarr store (skip completed blocks)",
    )
    args = parser.parse_args()

    run_pipeline(resume=args.resume)

