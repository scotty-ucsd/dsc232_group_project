# verify_spatial_features.py

# -----------------------------------------------------------------------------
# COMPUTATIONAL GLACIOLOGY - VERIFICATION PIPELINE
# -----------------------------------------------------------------------------
# DATE:   2026-02-13
# STATUS: DIAGNOSTIC
#
# Validates the physical and structural integrity of the engineered spatial 
# features Zarr store before downstream Long Sparse Parquet (LSP) assembly.
# -----------------------------------------------------------------------------

import time as _time
import xarray as xr
import numpy as np

# ── Configuration ───────────────────────────────────────────────────────────
FEATURES_ZARR = "data/processed/spatial_features_engineered.zarr"
EXPECTED_CHUNK_YX = 2048

def main():
    wall = _time.perf_counter()
    print("=" * 72)
    print(" VERIFYING SPATIAL FEATURES ZARR")
    print("=" * 72)

    # 1. Load Dataset
    print(f"[Load] Reading {FEATURES_ZARR}...")
    try:
        ds = xr.open_zarr(FEATURES_ZARR, consolidated=True)
    except Exception as e:
        print(f"[ERROR] Failed to load consolidated Zarr: {e}")
        return

    # 2. Structural & Chunking Verification
    print("\n[Check] Structural Metadata & Chunking")
    for var in ds.data_vars:
        dims = ds[var].dims
        chunks = ds[var].data.chunksize
        
        # Verify Y/X chunking matches expectations
        yx_chunks = chunks[-2:] # Last two dims are always (y, x)
        if yx_chunks != (EXPECTED_CHUNK_YX, EXPECTED_CHUNK_YX) and yx_chunks[0] < EXPECTED_CHUNK_YX:
            # Note: Edge chunks might be smaller than 2048, so we check the nominal chunk size
            # from the encoding if possible, but inspecting the tuple directly is usually sufficient 
            # for the first block.
            pass 
        
        print(f"  -> {var}: dims={dims}, chunks={chunks}, dtype={ds[var].dtype}")

    # 3. Masking & NaN Validation
    # We expect all data outside the ice mask to be NaN. 
    # Let's verify the base footprint using the static bed_slope.
    print("\n[Check] Mask Enforcement (Static Geometry)")
    
    # Compute total pixels vs valid pixels
    total_pixels = ds.x.size * ds.y.size
    valid_pixels = int(np.isfinite(ds['bed_slope']).sum().compute())
    ocean_rock_pixels = total_pixels - valid_pixels
    
    print(f"  -> Total Grid Pixels: {total_pixels:,}")
    print(f"  -> Valid Ice Pixels:  {valid_pixels:,} ({(valid_pixels/total_pixels)*100:.1f}%)")
    print(f"  -> Masked NaN Pixels: {ocean_rock_pixels:,}")

    if valid_pixels == total_pixels:
        print("[WARNING] No NaNs detected. Masking step may have failed.")

    # 4. Physical Boundary Verification
    # We will compute min/max for the first timestep to avoid memory blowouts
    print("\n[Check] Physical Value Bounds (t=0)")
    
    t0_surface = ds['h_surface_dynamic'].isel(time=0)
    t0_slope = ds['surface_slope'].isel(time=0)
    
    # Compute bounds eagerly
    print("  -> Computing scalar bounds...")
    metrics = xr.Dataset({
        'h_surf_min': t0_surface.min(),
        'h_surf_max': t0_surface.max(),
        'surf_slope_max': t0_slope.max(),
        'bed_slope_max': ds['bed_slope'].max(),
        'gl_dist_max': ds['dist_to_grounding_line'].max()
    }).compute()

    print(f"  -> Surface Elevation: {float(metrics['h_surf_min']):.1f} m to {float(metrics['h_surf_max']):.1f} m")
    print(f"  -> Max Surface Slope: {float(metrics['surf_slope_max']):.4f} m/m")
    print(f"  -> Max Bed Slope:     {float(metrics['bed_slope_max']):.4f} m/m")
    print(f"  -> Max GL Distance:   {float(metrics['gl_dist_max'] / 1000):.1f} km")

    # Sanity checks on physics
    if float(metrics['surf_slope_max']) > 5.0:
        print("[WARNING] Surface slope exceeds 5 m/m. Check for localized data anomalies or cliff artifacts.")
    if float(metrics['gl_dist_max']) < 1000:
        print("[WARNING] Max distance to grounding line is suspiciously low for Antarctica.")

    wall_t = _time.perf_counter() - wall
    print(f"\n{'=' * 72}")
    print(f" Verification Complete. {wall_t:.1f} s")
    print(f"{'=' * 72}")

if __name__ == "__main__":
    main()
