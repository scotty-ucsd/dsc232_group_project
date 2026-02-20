# Step 05a — Ocean Data Extraction (GLORYS → Ice Grid)

> **Script:** [`step_05a_extract_ocean.py`](file:///home/scotty/dsc232_group_project/pre_pre_processing_pipeline/src/step_05a_extract_ocean.py)
> **Output:** `data/processed/matched_ocean_grid.zarr`

---

## What This Script Does

This is the **most computationally intensive** script in the pipeline. It extracts ocean temperature (`thetao`) and salinity (`so`) from the GLORYS12V1 ocean reanalysis product and maps them onto the 500m Antarctic ice grid at the depth of the ice-ocean interface (ice draft). The result is a 3D `(time, y, x)` ocean property store spatially aligned with all other 500m products.

### Detailed Breakdown

#### 1. Source Data Loading (Eager, Not Lazy)
- Opens GLORYS12V1 NetCDF (`antarctic_ocean_physics_2019_2025.nc`) and crops to `latitude ∈ [-90, -60]`.
- **Critically, loads both 4D arrays entirely into RAM** as contiguous NumPy arrays:
  - `theta_full`: `(T, D, lat, lon)` — ~2.6 GB
  - `so_full`: `(T, D, lat, lon)` — ~2.6 GB
  - Total resident: ~5.2 GB
- This is a deliberate design choice: eager loading eliminates per-block Dask I/O overhead and enables direct NumPy indexing, which is ~10× faster than lazy Dask reads for the random-access pattern needed by BallTree lookups.

#### 2. BallTree Spatial Index (Haversine)
- Constructs a `sklearn.neighbors.BallTree` on the wet (non-NaN) surface ocean pixels using Haversine metric.
- This tree answers the query: "For each 500m ice-grid pixel, which is the nearest ocean pixel with valid data?"
- Typical result: ~100,000 wet ocean pixels indexed.

#### 3. Deepest Valid Depth Computation
- Custom function `_last_valid_depth_index()` finds the deepest non-NaN depth level at each wet ocean pixel.
- **This fixes a bug** in an earlier implementation that used `sum(dim='depth') - 1`, which fails when the water column has mid-depth NaN gaps (non-contiguous valid ranges).
- The corrected approach takes `max(weighted_index)` — always finds the true deepest valid level.

#### 4. Block-by-Block Processing
The 12,288 × 12,288 grid is processed in `CHUNK_SIZE × CHUNK_SIZE` (2048 × 2048) blocks:

For each block:
1. **Load Bedmap3 chunk** — read the ice mask and topography for this spatial block.
2. **Filter to ice-shelf pixels** — only `mask == 2` (ice shelf) or `mask == 3` (floating ice) have an ice-ocean interface.
3. **CRS transformation** — convert `(x, y)` in EPSG:3031 → `(lat, lon)` in EPSG:4326 using `pyproj`.
4. **BallTree query** — find the nearest ocean pixel for each ice-shelf pixel. Record the Haversine distance in metres.
5. **Ice draft computation** — `draft = surface - thickness`. Take `|draft|` as the depth below sea level.
6. **Depth clamping** — clamp the draft depth to the deepest valid ocean level at the matched ocean pixel. This prevents querying into the seabed where no ocean data exists.
7. **Deduplication** — at 500m ice vs ~8 km ocean, ~256 ice pixels map to the same ocean column. Using `np.unique()` reduces profile loads by ~16×.
8. **Interpolation bracket pre-computation** — linear interpolation indices and weights are computed once per block (they depend only on depth levels and clamped depths, which are time-invariant).
9. **Time-batched 3D writes** — for each `TIME_BATCH` (24 steps):
   - Load the relevant depth profiles for this batch.
   - Fan out unique profiles to all ice-shelf pixels via fancy indexing.
   - Apply pre-computed interpolation brackets in a single vectorised multiply-add.
   - Write `thetao` and `so` to the Zarr store using regional writes.

#### 5. Output Variables

| Variable | Dims | Description |
|---|---|---|
| `dist_to_ocean` | `(y, x)` | Haversine distance to nearest GLORYS wet pixel [m] |
| `ice_draft` | `(y, x)` | Ice base depth below sea level [m] |
| `clamped_depth` | `(y, x)` | Draft depth clamped to deepest valid ocean level [m] |
| `mascon_id` | `(y, x)` | GRACE mascon identifier (copied from Bedmap3 block) |
| `thetao` | `(time, y, x)` | Potential temperature at clamped draft depth [°C] |
| `so` | `(time, y, x)` | Practical salinity at clamped draft depth [PSU] |

#### 6. Resume Capability
- `--resume` flag skips blocks whose Zarr chunks already exist on disk — enables crash recovery without reprocessing the entire grid.

---

## Critique

> [!WARNING]
> The eager loading of ~5 GB into RAM is a calculated tradeoff. On a machine with <16 GB RAM, this would fail. The decision is documented but should be gated by a runtime memory check.

> [!NOTE]
> The BallTree approach uses the surface wet mask at `t=0` only — implicitly assuming ocean coverage is time-invariant. For GLORYS12V1 reanalysis this is correct, but for observational products with missing coverage, this could miss pixels that become valid at later time steps.

---

## Why Pre-Process Here?

> [!IMPORTANT]
> **This step performs BallTree spatial matching, depth-profile interpolation, and CRS transformation — operations that are fundamentally incompatible with PySpark's row-based execution model.**

1. **BallTree nearest-neighbour lookup.** Matching 100M+ ice-grid pixels to their nearest ocean pixel requires a spatial index (BallTree, KD-Tree). PySpark has no built-in spatial indexing. Implementing this as a UDF would serialise the entire BallTree per task, destroying parallelism.

2. **Depth-profile interpolation.** Extracting ocean properties at the ice draft depth requires indexing into 4D arrays `(time, depth, lat, lon)` at non-grid-aligned depth values. This is a vertical interpolation through the water column — an operation that has no SQL equivalent.

3. **CRS transformation (EPSG:3031 → EPSG:4326).** The BallTree operates in geographic coordinates (Haversine), but the ice grid is in projected coordinates (EPSG:3031). This per-pixel CRS conversion would require a PySpark UDF wrapping `pyproj`, which serialises Python objects and runs single-threaded.

4. **The 16× deduplication optimisation.** The insight that ~256 ice pixels share the same ocean column at 500m vs 8 km resolution requires understanding the spatial relationship between grids — knowledge that is lost once data is flattened to rows.

5. **Block-level checkpoint/resume.** The `--resume` flag enables crash recovery at the Zarr chunk level. PySpark's equivalent (stage-level retry) operates at a much coarser granularity and would re-process entire partitions.
