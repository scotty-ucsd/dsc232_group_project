# Step 03b — ICESat-2 Regridding (1 km → 500m)

> **Script:** [`step_03b_regrid_icesat.py`](file:///home/scotty/dsc232_group_project/pre_pre_processing_pipeline/src/step_03b_regrid_icesat.py)
> **Output:** `data/processed/icesat2_500m_{deltah,lag1,lag4}.zarr`

---

## What This Script Does

This script upsamples the seamless ICESat-2 mosaics (from step_03a, native 1 km) onto the 500m master grid by performing **bilinear interpolation**. It also injects the `mascon_id` from the master mascon map (step_02) into every output, ensuring all downstream datasets share the same GRACE mascon identity per pixel.

### Detailed Breakdown

#### 1. Dask Cluster Tuning
- 6 workers × 1 thread × 8 GB each.
- Explicit Dask memory management thresholds:
  - Target: 70% (start spilling to disk)
  - Spill: 85% (actively evict data)
  - Pause: 95% (stop scheduling new tasks)
- Single-threaded workers are intentional: xarray's `interp()` with scipy backends is not thread-safe under the GIL for some interpolation paths.

#### 2. Mascon Map as Coordinate Master
- Opens the mascon map Zarr (`mascon_id_master_map.zarr`) from step_02.
- Uses its `x` and `y` coordinates as the **interpolation targets**:
  ```python
  ds_regridded = ds_subset.interp(x=ds_mascon_map.x, y=ds_mascon_map.y, method="linear")
  ```
- This guarantees that the output grid is **byte-identical** to all other 500m products (Bedmap3, ocean, spatial features).

#### 3. Bilinear Interpolation
- `xr.interp(method="linear")` performs bilinear interpolation in the projected (x, y) space.
- `fill_value=np.nan` ensures that pixels outside the source data extent get NaN (no extrapolation).
- This is a 2× upsampling: each 1 km pixel becomes four 500m pixels, with values smoothly interpolated from the surrounding 1 km observations.

#### 4. Mascon ID Injection
- After interpolation, the `mascon_id` from the master map is directly assigned to the output dataset.
- This is not interpolated (it's a static 2D integer field) — it's a direct array copy.

#### 5. Dtype Optimisation
- All `float64` variables are downcast to `float32` after interpolation.
- This halves memory and disk usage with negligible precision loss for glaciological measurements (sub-millimetre accuracy is sufficient).

#### 6. Output
- Three separate Zarr stores, one per data group:
  - `icesat2_500m_deltah.zarr` — elevation anomaly + sigma + ice_area + data_count + misfit_rms
  - `icesat2_500m_lag1.zarr` — 1-quarter dh/dt rate + sigma + ice_area
  - `icesat2_500m_lag4.zarr` — 4-quarter dh/dt rate + sigma + ice_area
- Chunks: `{time: 1, y: 2048, x: 2048}`.
- Compression: Blosc Zstd (level 3), Zarr Format 2 enforced.

### Tasks Matrix

| Task Key | Input Zarr | Output Zarr | Variables |
|---|---|---|---|
| `delta_h` | `icesat2_1km_seamless_deltah.zarr` | `icesat2_500m_deltah.zarr` | `delta_h`, `delta_h_sigma`, `ice_area`, `data_count`, `misfit_rms` |
| `lag1` | `icesat2_1km_seamless_lag1.zarr` | `icesat2_500m_lag1.zarr` | `dhdt`, `dhdt_sigma`, `ice_area` |
| `lag4` | `icesat2_1km_seamless_lag4.zarr` | `icesat2_500m_lag4.zarr` | `dhdt`, `dhdt_sigma`, `ice_area` |

---

## Why Pre-Process Here?

> [!IMPORTANT]
> **Bilinear interpolation on a 3D `(time, y, x)` array requires n-dimensional indexing and neighbourhood access — operations that are impossible in PySpark's row-based model.**

1. **Spatial interpolation is not a SQL expression.** Bilinear interpolation requires accessing the four nearest neighbours in a regular grid for each target point. PySpark operates on independent rows — it has no concept of "the pixel 500m to the north" without an explicit self-join, which would be $O(N^2)$.

2. **Grid upsampling changes the number of rows.** Going from 1 km → 500m quadruples the pixel count. In PySpark, this would require generating new rows via `explode()` and computing interpolated values via a UDF that accesses surrounding row values — an anti-pattern that forces all data through a single Python process.

3. **Mascon ID injection is a spatial lookup.** Copying the mascon_id from a Zarr store into the output requires coordinate-aligned array assignment. In PySpark, this would be a spatial join — feasible but slow without spatial indexing (which PySpark does not provide natively).

4. **Zarr's chunked format enables parallel downstream reads.** The `{time: 1, y: 2048, x: 2048}` chunking means step_07 (flattening) can read one time step at a time without loading the full dataset — essential for the 64 GB memory constraint.
