# Step 00 — Master Grid Construction

> **Script:** [`step_00_master_grid.py`](file:///home/scotty/dsc232_group_project/pre_pre_processing_pipeline/src/step_00_master_grid.py)
> **Output:** `processed_layers/master_grid_template.nc`

---

## What This Script Does

This script is the **foundational prerequisite** of the entire pipeline. It constructs a 2D coordinate template — the "Master Grid" — that defines the spatial skeleton onto which every subsequent dataset is aligned.

### Detailed Breakdown

1. **Define Geographic Bounds (Antarctic Polar Stereographic — EPSG:3031)**
   - The bounds `[-3072000, 3072000]` in both `x` and `y` are chosen to cover the **entire Antarctic continent** plus the surrounding shelf break, in metres from the South Pole.
   - The resolution is fixed at **500 metres**, yielding a grid of `12,288 × 12,288` pixels (~151 million cells).

2. **Generate Pixel-Centre Coordinates**
   - Coordinates are computed as the **centre** of each pixel — not the edge — using `np.arange(x_min + resolution/2, x_max, resolution)`.
   - `float64` precision is explicitly used to prevent floating-point drift that would compound across 12,000+ grid cells and cause sub-pixel misalignment later in the pipeline.
   - The `y` axis is generated in **descending** order (`y_max - res/2 → y_min`), preserving the raster convention where `y[0]` is the northernmost row.

3. **Create an Empty `xr.Dataset`**
   - The result is a coordinate-only dataset with `x` and `y` dimensions — no data variables yet.
   - This is intentional: it serves as a **reindexing template**. Every subsequent step (GRACE, ICESat-2, Bedmap3, Ocean) will `.reindex_like()` or `.interp()` onto this grid.

4. **Assign CRS via `rioxarray`**
   - `ds.rio.write_crs("EPSG:3031")` embeds the coordinate reference system metadata directly into the NetCDF.
   - This is **critical** because later reprojection steps (e.g., GRACE from EPSG:4326 → EPSG:3031) depend on rioxarray/rasterio knowing the target CRS.

5. **Save as NetCDF**
   - The output is a minimal ~1 KB file — just coordinates + CRS metadata.

### Critical Design Decisions

| Decision | Rationale |
|---|---|
| 500m resolution | Matches ICESat-2 ATL15 target after 2× upsampling from native 1 km. Finest resolution that fits in ~64 GB RAM for the full grid. |
| EPSG:3031 | Native CRS for Antarctic research. Avoids repeated reprojection. |
| `float64` coordinates | Prevents the "floating point nightmare" — sub-ULP drift across 12,288 cells would cause nearest-neighbour mismatches downstream. |
| Pixel-centre convention | Consistent with how `xr.interp()` and `rioxarray.reproject()` interpret coordinate values. Edge-convention would shift everything by 250m. |

---

## Why Pre-Process Here?

> [!IMPORTANT]
> **PySpark has no native understanding of coordinate reference systems, raster grids, or n-dimensional spatial alignment.**

This step establishes the **single source of truth** for the spatial coordinate system. Without it, every downstream dataset would carry its own coordinate grid, and aligning them inside PySpark would require:

1. **Custom UDFs for CRS reprojection** — PySpark cannot natively reproject coordinates from EPSG:4326 to EPSG:3031. You would need `pyproj` inside a UDF, which serialises Python objects per-row and destroys parallelism.
2. **Float-key equi-join failures** — PySpark uses strict equality for joins. IEEE 754 floating-point coordinates from independently generated grids will silently fail equi-joins due to sub-ULP differences (e.g., `500.0000000001` ≠ `500.0`). By defining a single template here, all downstream datasets share **exact** coordinate values.
3. **Grid expansion at join time** — Without a pre-defined grid, a PySpark `CROSS JOIN` between 4 datasets with different grids would produce a cartesian product, potentially billions of spurious rows.

By building this grid once in xarray/NumPy on the local machine, we guarantee that all downstream Zarr stores use **byte-identical** coordinate arrays — enabling exact, index-aligned joins in the downstream Parquet/PySpark pipeline.
