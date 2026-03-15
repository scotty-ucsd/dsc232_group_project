# Step 06 — Spatial Feature Engineering

> **Script:** [`step_06_create_spatial_features.py`](file:///home/scotty/dsc232_group_project/pre_pre_processing_pipeline/src/step_06_create_spatial_features.py)
> **Output:** `data/processed/spatial_features_engineered.zarr`

---

## What This Script Does

This script computes **engineered spatial features** from the Bedmap3 and ICESat-2 500m stores. These features encode geometric and topographic properties that are physically meaningful for ice-sheet dynamics and form the spatial feature backbone for the downstream ML model.

### Detailed Breakdown

#### 1. Dynamic Surface Elevation

```
h_surface_dynamic = bedmap.surface + icesat2.delta_h
```

- Combines the **static Bedmap3 surface** with the **time-varying ICESat-2 elevation anomaly**.
- Result: a 3D `(time, y, x)` surface elevation field that captures temporal ice-level changes.
- Dimension ordering is enforced as `(time, y, x)` via explicit transpose (ICESat-2 stores `delta_h` as `(y, x, time)` which propagates through arithmetic).

#### 2. Bed Slope

$$|\nabla(bed)| = \sqrt{\left(\frac{\partial bed}{\partial y}\right)^2 + \left(\frac{\partial bed}{\partial x}\right)^2}$$

- **Physical meaning**: Steeper bed slopes beneath the ice correlate with basal drag, subglacial water routing, and potential ice-stream instability.
- **Implementation**: Uses `dask.array.map_overlap()` with a 1-pixel halo and `boundary='nearest'`. This:
  - Adds a 1-pixel halo from neighbouring chunks before calling `np.gradient`, then trims.
  - Ensures correct central differences at chunk boundaries (without overlap, chunk-edge gradients would use one-sided differences).
  - Guarantees every padded chunk has ≥ 3 elements (np.gradient's minimum), preventing crashes on small edge chunks.

#### 3. Surface Slope

$$|\nabla(h_{surface\_dynamic})| = \sqrt{\left(\frac{\partial h}{\partial y}\right)^2 + \left(\frac{\partial h}{\partial x}\right)^2}$$

- **Physical meaning**: Surface slope is a primary driver of ice flow velocity (via the shallow-ice approximation). Steeper slopes → faster flow.
- Same `map_overlap` implementation as bed slope, but applied to the dynamic surface (3D).

#### 4. Distance to Grounding Line

$$dist\_to\_grounding\_line = \text{EDT}(\neg \text{grounded\_ice})$$

- **Euclidean Distance Transform** from every pixel to the nearest grounded-ice pixel (mask == 1).
- **Physical meaning**: Proximity to the grounding line is the strongest single predictor of ice-sheet vulnerability. Floating ice shelves further from the grounding line are more exposed to ocean warming.
- **Implementation**: Computed eagerly on the full 12,288 × 12,288 mask (~150 MB as uint8) using `scipy.ndimage.distance_transform_edt`. The EDT is a global operation — it cannot be chunked because the distance at any pixel depends on the entire mask.
- Result is wrapped back into a Dask array for lazy downstream writes.

#### 5. Ice-Only Masking
- All features are masked to `(mask == 1 | mask == 3)` — grounded ice and floating ice shelves.
- Ocean, exposed rock, and other categories are set to NaN.

#### 6. Output
- Variables: `h_surface_dynamic` (3D), `bed_slope` (2D), `surface_slope` (3D), `dist_to_grounding_line` (2D).
- Chunks: `(1, 2048, 2048)` for 3D, `(2048, 2048)` for 2D.
- Compression: Blosc Zstd (level 3), Zarr Format 2.

---

## Critique

> [!NOTE]
> The `open_zarr()` calls enforce `chunks={"y": 2048, "x": 2048}` on both Bedmap and ICESat stores. This is a critical performance decision — without it, Dask inherits potentially misaligned native chunk sizes from the two stores, generating a massive rechunk graph (the script author notes this was causing a 1.13 GB task graph in an earlier version).

> [!TIP]
> The EDT could be accelerated using `scipy.ndimage.distance_transform_edt` with `sampling=500.0` to directly output distances in metres. The script already does this — good.

---

## Why Pre-Process Here?

> [!IMPORTANT]
> **Gradient computation, Euclidean distance transforms, and neighbourhood-based operations are fundamentally incompatible with PySpark's row-independent processing model.**

1. **Gradient is a neighbourhood operation.** `np.gradient` requires adjacent pixels to compute central differences. PySpark shuffles data across executors by partition — there is no guarantee that adjacent pixels will co-locate on the same executor. A PySpark implementation would require sorting by `(y, x)`, windowing with `LAG`/`LEAD`, and computing the gradient manually — resulting in massive shuffle and a fundamentally slower approach than in-memory NumPy.

2. **EDT is a global operation.** The distance to the nearest grounded-ice pixel at any point depends on the location of ALL grounded-ice pixels. There is no way to partition this computation across Spark executors without exchanging the full mask. In NumPy/SciPy, the EDT runs in $O(N)$ on the full array. In PySpark, the equivalent would be an $O(N^2)$ spatial self-join.

3. **The `map_overlap` pattern.** Dask's `map_overlap` fetches halo regions from neighbouring chunks, enabling correct edge computation. PySpark has no equivalent — its window functions work within partitions, and specifying "give me the row 500m to the north" requires knowing the physical layout of the data.

4. **These are ML features, not raw observations.** By engineering `bed_slope`, `surface_slope`, `h_surface_dynamic`, and `dist_to_grounding_line` locally, the downstream Parquet files carry ready-to-use features. If computed in PySpark on SDSC, every feature engineering iteration would require re-reading the full Zarr stores and re-computing gradients — wasting cluster time.
