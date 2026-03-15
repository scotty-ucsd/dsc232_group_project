# Step 04 — Bedmap3 Regridding

> **Script:** [`step_04_process_bedmap.py`](file:///home/scotty/dsc232_group_project/pre_pre_processing_pipeline/src/step_04_process_bedmap.py)
> **Output:** `data/processed/bedmap3_500m.zarr`

---

## What This Script Does

This script regrids **Bedmap3** — the definitive Antarctic subsurface topography dataset — from its native resolution onto the pipeline's 500m master grid. It produces the **static geological/glaciological foundation** upon which all dynamic (time-varying) data is overlaid.

### Detailed Breakdown

#### 1. Input & Variable Mapping
Opens `bedmap3.nc` with `decode_cf=True` so that the `-9999` fill values are automatically converted to NaN by xarray's CF convention decoder.

The script regrids 5 variables, each with a specific resampling method:

| Bedmap3 Variable | Pipeline Name | Method | Dtype | Physical Meaning |
|---|---|---|---|---|
| `bed_topography` | `bed` | Bilinear | float32 | Bedrock elevation below the ice |
| `surface_topography` | `surface` | Bilinear | float32 | Ice surface elevation |
| `ice_thickness` | `thickness` | Bilinear | float32 | Depth of the ice column |
| `bed_uncertainty` | `errbed` | Bilinear | float32 | Uncertainty in bedrock elevation |
| `mask` | `mask` | Nearest | int8 | Land/ice/ocean classification |

- **Bilinear** for continuous fields — preserves smooth elevation gradients.
- **Nearest** for the mask — it is categorical (0=ocean, 1=grounded ice, 2=rock, 3=floating ice shelf). Bilinear would create invalid fractional categories.

#### 2. Coordinate Authority
- Opens the mascon map Zarr (`mascon_id_master_map.zarr`) from step_02.
- Uses its `x` and `y` coordinates as the interpolation targets, ensuring byte-identical coordinate arrays across all 500m products.
- Injects `mascon_id` directly from the mascon map into the output.

#### 3. Physics Cleanup
- **Negative thickness clipping**: Bilinear interpolation at the ice edge can produce small negative thickness values (e.g., −2m where the interpolant crosses zero between a thick-ice and no-ice pixel). The script clips these to 0.

#### 4. Zarr Output
- Chunks: `{y: 2048, x: 2048}` — no time dimension (these are static 2D fields).
- Compression: Blosc Zstd (level 3).
- Zarr Format 2 enforced.

### Role in the Pipeline

Bedmap3 is the **geometric anchor** of the entire pipeline:
- `surface` and `thickness` are used in step_05a to calculate **ice draft** (the depth of the ice base below sea level).
- `mask` is used in steps 05a, 06, and 07 to filter for ice-only pixels.
- `bed` is used in step_06 to compute **bed slope** (a key ML feature).
- All downstream analysis relies on this store for the static geological "background" upon which dynamic signals (elevation change, ocean temperature) are overlaid.

---

## Why Pre-Process Here?

> [!IMPORTANT]
> **Bedmap3 is a 2D NetCDF raster that must be spatially resampled and share exact coordinates with every other dataset. PySpark cannot read NetCDF, cannot perform spatial interpolation, and cannot enforce coordinate identity across datasets.**

1. **NetCDF incompatibility.** Bedmap3 is a standard NetCDF4/HDF5 file. PySpark's native readers handle Parquet, CSV, JSON, and ORC — not NetCDF. Reading it would require a custom Python UDF wrapping `xr.open_dataset()`, which loads the entire file into driver memory.

2. **Bilinear raster resampling.** Regridding from the Bedmap3 native resolution to 500m requires 2D spatial interpolation. PySpark processes data row-by-row — it has no concept of "the cell 500m to the east" without an expensive self-join or window function over spatial coordinates.

3. **Mascon ID injection.** Copying a 2D array (`mascon_id`) from one Zarr store to another is a trivial NumPy operation. In PySpark, it would require a spatial join between two DataFrames, adding overhead and risking coordinate misalignment.

4. **Physics enforcement (negative clipping).** The `.where(thickness >= 0, 0)` operation is element-wise and trivial in xarray. In PySpark it would be a simple `CASE WHEN`, but by the time you've ingested the NetCDF, reprojected it, and loaded it into a DataFrame, the overhead of the whole pipeline dwarfs the cost of a local clip.

5. **This creates the ice mask for all downstream filtering.** Steps 06 and 07 open this Zarr to read `mask.isin([1, 3])` for ice-only filtering. Having this precomputed avoids re-reading and re-filtering the raw NetCDF in every downstream step.
