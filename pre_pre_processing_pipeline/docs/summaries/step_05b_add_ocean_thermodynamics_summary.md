# Step 05b — Ocean Thermodynamics (T_f, T*)

> **Script:** [`step_05b_add_ocean_thermodynamics.py`](file:///home/scotty/dsc232_group_project/pre_pre_processing_pipeline/src/step_05b_add_ocean_thermodynamics.py)
> **Output:** `data/processed/thermodynamic_ocean_grid.zarr`

---

## What This Script Does

This script computes two physically motivated derived variables — **freezing-point temperature (T_f)** and **thermal driving (T\*)** — from the matched ocean data produced by step_05a. These variables quantify the basal melt potential of ocean water at the ice-ocean interface.

### Detailed Breakdown

#### 1. Filesystem-Level Copy (Not Dask Copy)
- Instead of reading the 6-variable input store through Dask, decompressing, and recompressing, the script uses `shutil.copytree()` to **duplicate the raw Zarr chunks at the filesystem level**.
- This is ~10× faster for ~12 GB of zstd-compressed data: no decompression, no Python overhead — it's a `cp -r` equivalent.
- The stale `.zmetadata` file is deleted from the copy to prevent xarray from reading cached metadata that doesn't list the two new variables.

#### 2. Freezing-Point Temperature (Millero 1978 / Jenkins 1991)

$$T_f(S, P) = \lambda_1 \cdot S + \lambda_2 + \lambda_3 \cdot P$$

Where:
| Symbol | Value | Units | Meaning |
|---|---|---|---|
| $\lambda_1$ | −0.0575 | °C / PSU | Salinity depression coefficient |
| $\lambda_2$ | +0.0901 | °C | Constant offset |
| $\lambda_3$ | −7.61 × 10⁻⁴ | °C / dbar | Pressure depression coefficient |
| $S$ | `so` array | PSU | Practical salinity from GLORYS |
| $P$ | `max(0, -ice_draft)` | dbar ≈ m | Pressure at ice-ocean interface |

#### 3. Critical Bug Fix: Pressure Sign Convention
The original implementation used `ice_draft` directly as pressure. **This was wrong.**

- `ice_draft = surface - thickness` — this is **negative** when the ice base is below sea level (the normal case for floating ice shelves).
- Using a negative pressure in the freezing-point equation **reverses** the pressure-dependent depression:
  - Original: $T_f(P = -2483) = $ too warm by +3.78°C
  - Fixed: $T_f(P = +2483) = $ correct freezing-point depression
- The fix: `pressure_dbar = max(0, -ice_draft)` — negates the draft and clips to ≥ 0.

#### 4. Thermal Driving

$$T^* = \theta - T_f$$

- **Positive T\*** → ocean is warmer than local freezing → **melt potential**.
- **Negative T\*** → ocean is supercooled → **refreezing potential**.
- This is the primary input for basal melt rate parameterisations (Holland & Jenkins 1999).

#### 5. Append & Consolidate
- Only `T_f` and `T_star` are computed and appended to the copied store (`mode="a"`), leaving the original 6 variables untouched.
- Re-consolidates metadata to include the new variables.
- Sanity check: prints min/max of both T_f and T* for manual validation.

### Output Variables (8 total)

| Variable | Source Step | Dims |
|---|---|---|
| `dist_to_ocean` | 05a | `(y, x)` |
| `ice_draft` | 05a | `(y, x)` |
| `clamped_depth` | 05a | `(y, x)` |
| `mascon_id` | 05a | `(y, x)` |
| `thetao` | 05a | `(time, y, x)` |
| `so` | 05a | `(time, y, x)` |
| `T_f` | **05b** | `(time, y, x)` |
| `T_star` | **05b** | `(time, y, x)` |

---

## Why Pre-Process Here?

> [!IMPORTANT]
> **The freezing-point equation depends on a 2D pressure field derived from 3D ice geometry and requires physical understanding of sign conventions. Computing it pre-Spark ensures correctness and avoids replicating the physics in SQL.**

1. **The physics is sign-sensitive.** The pressure sign convention (`-ice_draft`, clipped to non-negative) is subtle and was already the source of a critical bug. Implementing this in a PySpark `CASE WHEN` statement risks repeating the same error without the benefit of xarray's named dimensions and the physicist's intuition about coordinate signs.

2. **Broadcasting 2D pressure across 3D `(time, y, x)` data.** Xarray automatically broadcasts `pressure_dbar(y, x)` against `so(time, y, x)`. In PySpark, you would need a self-join or a window function to combine a static 2D field with a dynamic 3D table — adding complexity and shuffle overhead.

3. **The filesystem copy trick.** Copying ~12 GB of compressed Zarr chunks at the OS level is vastly more efficient than reading, decompressing, and rewriting through Dask/PySpark. This optimisation is only available when working with chunk-based filesystems like Zarr.

4. **T* is a derived feature for the ML model.** By computing it during pre-processing, the downstream Parquet tables carry the ready-to-use feature. If computed in PySpark, every ML training run would need to re-derive it, wasting cluster compute time.
