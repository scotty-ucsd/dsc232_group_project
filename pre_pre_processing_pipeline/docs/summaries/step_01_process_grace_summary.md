# Step 01 — GRACE/GRACE-FO Mass Anomaly Processing

> **Script:** [`step_01_process_grace.py`](file:///home/scotty/dsc232_group_project/pre_pre_processing_pipeline/src/step_01_process_grace.py)
> **Output:** `data/processed/grace.zarr`

---

## What This Script Does

This script transforms raw **CSR GRACE/GRACE-FO mascon mass anomaly data** from its native NetCDF (EPSG:4326, 0.25° grid) into a reprojected, compressed Zarr store aligned to the pipeline's Antarctic Polar Stereographic CRS (EPSG:3031).

### Detailed Breakdown

#### 1. Lazy Loading & Manual Time Decoding
- Opens the raw mascon NetCDF **lazily** with `chunks={}` (respecting native disk chunking), then re-chunks to `{time: 1, lat: -1, lon: -1}` for one-slice-at-a-time processing.
- **Manual time decoding** is required because the raw time variable uses days since `2002-01-01` in a non-CF-compliant format. The script converts each raw float to `pd.Timestamp` manually via `timedelta(days=float(t))`.
- Filters to the date range `2002-04-01` → `2025-12-31`.
- Handles legacy naming: renames `lwe_thickness` → `lwe_length` if present (CSR naming inconsistency between releases).

#### 2. Mask Merge & Coordinate Cleaning
- Loads **Land Mask** and **Ocean Mask** (separate NetCDF files from CSR) and merges them with the main mascon dataset.
- **Longitude wrapping**: converts 0–360° → −180–180° and re-sorts coordinates. This is critical because `rioxarray.reproject()` assumes −180–180°.
- **Latitude cropping**: drops all data with `lat > −50°` to focus exclusively on the Antarctic region, reducing memory by ~75%.

#### 3. Unique Mascon ID Generation (Dask-Lazy)
- Generates a linear integer ID grid `mascon_id = [0, 1, 2, …, N_lat × N_lon]` using `dask.array.arange()`.
- This ID serves as the **join key** between pixel-level data and GRACE mascon-level aggregated mass anomalies in the downstream fusion step (step_08).
- The ID is generated **after** coordinate cleaning but **before** reprojection, so it carries the EPSG:4326 mascon identity through the CRS transform.

#### 4. Split-Domain Reprojection (EPSG:4326 → EPSG:3031)
This is the most nuanced part of the script. Rather than naively reprojecting the full dataset, it implements a **split-domain** strategy:

| Domain | Resampling | Rationale |
|---|---|---|
| Land LWE signal | Bilinear | Continuous field — bilinear preserves smooth gradients |
| Ocean LWE signal | Bilinear | Same reasoning |
| Land/Ocean Masks | Nearest Neighbour | Discrete/categorical — bilinear would create invalid fractional mask values |
| Mascon IDs | Nearest Neighbour | Integer IDs — bilinear would average IDs into meaningless floats |

- After reprojection, land and ocean LWE signals are recombined using `.combine_first()` (land takes priority in overlapping regions).
- Spatial alignment is enforced by `reindex_like(reproj_nearest, tolerance=1e-5)`.

#### 5. CRS Safety Fix
- The script documents a **bug fix**: `rio.write_crs("EPSG:4326")` must be called **after** all variables (including `mascon_id`) are added to the merged dataset. Xarray does not propagate CRS metadata to variables added post-CRS-assignment, causing silent reprojection failures for late-added variables.

#### 6. Zarr Output
- Chunks: `{time: 1, x: -1, y: -1}` — each time step is a single chunk (optimal for time-series slicing).
- Compression: Blosc Zstd (level 3, bitshuffle).
- Metadata is consolidated at the end for single-file `.zmetadata` reads.

---

## Critique of the Implementation

> [!WARNING]
> **Target resolution (`TARGET_RES = 27750` metres)** is dramatically coarser than the 500m master grid. This is the native GRACE mascon scale (~0.25°). While this preserves the actual GRACE spatial resolution, it means `mascon_id` and `lwe_length` will be very blocky when later resampled by step_02 onto the 500m grid. This is physically correct (GRACE cannot resolve 500m features) but should be clearly documented for downstream consumers.

> [!NOTE]
> The `warnings.filterwarnings('ignore')` is a pragmatic choice for production scripts, but it silences potentially important `rioxarray` and `Dask` warnings (e.g., chunk alignment mismatches). In a CI/CD environment, consider routing warnings to a log file instead.

---

## Why Pre-Process Here?

> [!IMPORTANT]
> **PySpark cannot reproject raster data across coordinate reference systems.**

1. **CRS Reprojection is a raster-native operation.** GRACE data arrives in EPSG:4326 (geographic lat/lon). Every other dataset in this pipeline uses EPSG:3031 (Antarctic Polar Stereographic). PySpark has no rasterio/rioxarray integration — reprojecting inside Spark would require loading entire spatial arrays into driver memory via `collect()`, defeating the purpose of distributed computing.

2. **Split-domain resampling requires n-dimensional awareness.** The decision to use bilinear for continuous fields and nearest-neighbour for discrete fields is a spatial data operation. PySpark does not distinguish between resampling strategies for different columns during a spatial join.

3. **Time decoding with non-CF-compliant formats.** The raw GRACE time encoding uses a non-standard reference date. While PySpark's `CAST(time AS TIMESTAMP)` handles standard Unix epochs, it cannot decode arbitrary "days since 2002-01-01" floats without a UDF.

4. **Zarr as an intermediate format.** Writing to Zarr (instead of directly to Parquet) at this stage preserves the 3D `(time, y, x)` structure needed for the mascon ID mapping in step_02 and for spatial analysis. Flattening to Parquet too early would lose the grid topology.
