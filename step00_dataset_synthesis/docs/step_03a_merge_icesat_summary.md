# Step 03a — ICESat-2 Tile Merge (Lowest-Uncertainty-First)

> **Script:** [`step_03a_merge_icesat.py`](file:///home/scotty/dsc232_group_project/pre_pre_processing_pipeline/src/step_03a_merge_icesat.py)
> **Output:** `data/processed/intermediate/icesat2_1km_seamless_{deltah,lag1,lag4}.zarr`

---

## What This Script Does

This script merges **four overlapping ICESat-2 ATL15 Antarctic tiles** (A1–A4) into seamless continental mosaics at the native 1 km resolution. It processes three distinct data groups — `delta_h` (elevation anomaly), `dhdt_lag1` (1-quarter rate), and `dhdt_lag4` (4-quarter rate) — each into its own Zarr store.

### Detailed Breakdown

#### 1. Dask Cluster Setup (Optimised for 55 GB / 12 Cores)
- Starts a `LocalCluster` with 4 workers × 2 threads × 12 GB memory limit each.
- This memory configuration is tuned for the specific hardware. Each worker handles one tile group at a time without spilling to disk.

#### 2. Master Canvas Construction
- Scans all four tile files to determine the **global bounding box** for the Antarctic continent.
- Coordinates are snapped to a strict 1 km grid by rounding to the nearest 1000m (`np.round(x / 1000) * 1000`).
- Generates a `canvas` dataset: a coordinate-only template spanning the full continental extent, used to align tile data onto a common grid.

#### 3. Lowest-Uncertainty-First Merge Logic
This is the algorithmic core of the script. The four ATL15 tiles overlap at their boundaries (by design). In overlap zones, the merge strategy selects the observation with the **lowest measurement uncertainty (sigma)**:

```
better_mask = new_valid & (~curr_valid | (new_sigma < curr_sigma))
```

- If a pixel exists in the new tile but not the current mosaic → **use the new tile** (fills gaps).
- If a pixel exists in both → **use whichever has the smaller sigma** (lowest-uncertainty-first).
- This is scientifically robust: uncertain observations at tile edges are replaced by higher-confidence overlapping observations.

#### 4. Physics Audit
The `verify_physics()` function runs mandatory sanity checks on every merged product:

| Check | Threshold | Failure Mode |
|---|---|---|
| `delta_h` or `dhdt` range | ±500 m | `FATAL` warning (physical impossibility — no ice change >500m exists) |
| `ice_area` sign | ≥ 0 | `ValueError` — negative area is physically impossible |
| `ice_area` scale detection | Auto-detect km² vs fraction | Validates against 1.01 (fraction) or 2.0e6 m² based on magnitude |

#### 5. Coordinate Snapping
- `fix_coordinates()` snaps each tile's `x` and `y` to the nearest multiples of 1000m.
- Validates monotonicity after snapping — non-monotonic coordinates would cause silent index errors in xarray.

#### 6. Zarr Output
- Chunks: `{time: 1, y: 1024, x: 1024}` — one time step per chunk, 1024×1024 spatial tiles.
- Compression: Blosc Zstd (level 3, byte shuffle).
- **Zarr Format 2 is explicitly forced** (`zarr_format=2`) because the zarr-python 3.x default (Format 3) introduces metadata incompatibilities with older xarray readers.
- Metadata is consolidated separately after the write completes.

### Data Variables Per Group

| Group | Variables | Dimensions |
|---|---|---|
| `delta_h` | `delta_h`, `delta_h_sigma`, `ice_area`, `misfit_rms`, `data_count` | `(time, y, x)` |
| `dhdt_lag1` | `dhdt`, `dhdt_sigma`, `ice_area` | `(time, y, x)` |
| `dhdt_lag4` | `dhdt`, `dhdt_sigma`, `ice_area` | `(time, y, x)` |

---

## Critique

> [!NOTE]
> The skipping logic `if os.path.exists(out_path): continue` is a simple resume mechanism but not atomic. If a previous run crashed mid-write, the output directory will exist but contain corrupted/incomplete data. A safer approach would be to write to a temporary path and atomically rename on completion, or check for a `.zmetadata` file (which is only created after consolidation).

---

## Why Pre-Process Here?

> [!IMPORTANT]
> **ICESat-2 ATL15 data is stored as four overlapping NetCDF tiles with hierarchical HDF5 groups. PySpark cannot read NetCDF groups, and it has no concept of "lowest-uncertainty-first" spatial merging.**

1. **HDF5/NetCDF group access.** ATL15 files use HDF5 groups (`/delta_h`, `/dhdt_lag1`, `/dhdt_lag4`). PySpark reads tabular formats (CSV, Parquet, ORC) — it cannot navigate HDF5 group hierarchies. Even with a custom data source, reading nested groups would require non-distributed I/O.

2. **Overlap resolution is a spatial algorithm.** The "lowest sigma wins" merge requires comparing 2D uncertainty fields from adjacent tiles at overlapping pixels. This is fundamentally a pixel-by-pixel comparison across aligned arrays — an operation native to xarray but requiring expensive UDFs in PySpark.

3. **Coordinate snapping.** The rounding/monotonicity validation ensures all tiles share exact coordinate values. Without this, PySpark joins on `(x, y)` would silently drop rows where tile A has `x = 1000.0000000001` and tile B has `x = 999.9999999999`.

4. **Zarr enables parallel I/O downstream.** By writing to Zarr (not NetCDF), subsequent steps can read individual spatial chunks in parallel without loading the entire file — a capability that PySpark's file-per-partition model cannot replicate for n-dimensional data.
