"""
fuse_data.py
-------------------------------------------------------------------------------
COMPUTATIONAL GLACIOLOGY - ICESAT-2 ANCHORED FUSION
-------------------------------------------------------------------------------
Executes an observation-anchored left join.  Engineers quarterly lag features
for Ocean and GRACE data, aligning them to sparse ICESat-2 altimetry epochs
via a Year-Month temporal tolerance.


Fixes (2026-02-14)
-------------------
 1.  LWE fusion uses |delta_h| weighting to prevent sign-flip amplification
     in mascons with mixed thinning/thickening pixels.
 2.  Join keys ROUND(y, 1), ROUND(x, 1) to prevent float-equality drift
     across independently-flattened Parquet tables.
 3.  _parquet_glob() applied to all 4 source paths (directory auto-scan).
 4.  Full feature set: so, T_f, h_surface_dynamic, clamped_depth,
     dist_to_ocean, ice_draft, surface, bed now included.
 5.  GRACE: MAX(lwe_length) for consistency with build_lsp_duckdb.py.
 6.  COALESCE(STDDEV_SAMP(...), 0) prevents NULL for edge months where
     the rolling 3-month window has < 2 observations.
 7.  temp_directory configured for out-of-core spill safety.
 8.  Hive-partitioned output by month_idx for downstream predicate pushdown.
 9.  Join-hit-rate diagnostics printed after every key join.
10.  Indices on intermediate table join columns.
-------------------------------------------------------------------------------
"""


import os
import shutil
import time as _time
import duckdb


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
DIR_FLATTENED = "data/flattened"


STATIC_PATH = os.path.join(DIR_FLATTENED, "bedmap3_static.parquet")
ICESAT_PATH = os.path.join(DIR_FLATTENED, "icesat2_dynamic.parquet")
OCEAN_PATH  = os.path.join(DIR_FLATTENED, "ocean_dynamic.parquet")
GRACE_PATH  = os.path.join(DIR_FLATTENED, "grace.parquet")


ML_FEATURE_DIR = os.path.join(DIR_FLATTENED, "antarctica_sparse_features.parquet")


MEMORY_LIMIT            = "48GB"
THREADS                 = 8
TEMP_DIRECTORY          = os.path.join(DIR_FLATTENED, "_duckdb_fuse_tmp")
MAX_TEMP_DIRECTORY_SIZE = "500GiB"
FUSION_MIN_MEAN_DH_M   = 1e-4   # 0.1 mm noise floor for fusion division




# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------


def _parquet_glob(path: str) -> str:
   """Return a DuckDB-compatible glob for a Parquet source.


   Handles both:
     - Single Parquet file:  path.parquet   -> 'path.parquet'
     - Directory of parts:   path/          -> 'path/**/*.parquet'
   """
   if os.path.isdir(path):
       return os.path.join(path, "**", "*.parquet")
   return path




# ---------------------------------------------------------
# PIPELINE
# ---------------------------------------------------------


def build_observation_anchored_dataset():
   wall_t = _time.perf_counter()


   con = duckdb.connect(database=":memory:")
   con.execute(f"SET memory_limit = '{MEMORY_LIMIT}'")
   con.execute(f"SET threads = {THREADS}")
   con.execute(f"SET temp_directory = '{TEMP_DIRECTORY}'")
   con.execute(f"PRAGMA max_temp_directory_size='{MAX_TEMP_DIRECTORY_SIZE}'")


   print("=" * 72)
   print(" BUILDING ICESAT-ANCHORED ML FEATURE STORE")
   print("=" * 72)


   # ==================================================================
   # 1. REGISTER SOURCE VIEWS
   #    - _parquet_glob() handles both file and directory layouts.
   #    - ROUND(y, 1), ROUND(x, 1) normalise float join keys so that
   #      sub-ULP drift between independently flattened tables never
   #      causes a silent equi-join miss.
   #    - CAST(mascon_id AS INTEGER) converts NaN -> NULL and aligns
   #      the type for clean integer equi-joins with GRACE.
   # ==================================================================
   print("\n[1/6] Registering Views (ROUND join keys, CAST mascon_id) ...")


   static_glob = _parquet_glob(STATIC_PATH)
   icesat_glob = _parquet_glob(ICESAT_PATH)
   ocean_glob  = _parquet_glob(OCEAN_PATH)
   grace_glob  = _parquet_glob(GRACE_PATH)


   con.execute(f"""
       CREATE VIEW static AS
       SELECT ROUND(y, 1) AS y, ROUND(x, 1) AS x,
              CAST(mascon_id AS INTEGER) AS mascon_id,
              * EXCLUDE (y, x, mascon_id)
       FROM read_parquet('{static_glob}')
   """)
   con.execute(f"""
       CREATE VIEW icesat AS
       SELECT ROUND(y, 1) AS y, ROUND(x, 1) AS x,
              * EXCLUDE (y, x)
       FROM read_parquet('{icesat_glob}')
   """)
   con.execute(f"""
       CREATE VIEW ocean AS
       SELECT ROUND(y, 1) AS y, ROUND(x, 1) AS x,
              * EXCLUDE (y, x)
       FROM read_parquet('{ocean_glob}')
   """)
   con.execute(f"""
       CREATE VIEW grace AS
       SELECT CAST(mascon_id AS INTEGER) AS mascon_id,
              * EXCLUDE (mascon_id)
       FROM read_parquet('{grace_glob}')
   """)


   for v in ("static", "icesat", "ocean", "grace"):
       cols = [r[0] for r in con.execute(
           f"SELECT column_name FROM information_schema.columns "
           f"WHERE table_name = '{v}' ORDER BY ordinal_position"
       ).fetchall()]
       print(f"  -> {v:8s}  {len(cols)} cols  {cols}")


   # ==================================================================
   # 2. OCEAN FEATURE ENGINEERING
   #    Full thermodynamic suite: thetao, T_star, so, T_f.
   #    Quarterly rolling stats with COALESCE(STDDEV_SAMP, 0) so the
   #    first 1-2 months never propagate NULL into ML features.
   # ==================================================================
   print("\n[2/6] Engineering Ocean Lags (Quarterly T*, thetao, so, T_f) ...")
   t0 = _time.perf_counter()


   con.execute("""
       CREATE TABLE ocean_features AS
       WITH monthly_ocean AS (
           SELECT
               y, x,
               EXTRACT(YEAR FROM time) * 12 + EXTRACT(MONTH FROM time) AS month_idx,
               AVG(thetao)   AS thetao_mo,
               AVG("T_star") AS t_star_mo,
               AVG(so)       AS so_mo,
               AVG("T_f")    AS t_f_mo
           FROM ocean
           GROUP BY y, x, month_idx
       )
       SELECT
           y, x, month_idx,
           thetao_mo,
           t_star_mo,
           so_mo,
           t_f_mo,
           -- T* quarterly
           AVG(t_star_mo) OVER w3   AS t_star_quarterly_avg,
           COALESCE(STDDEV_SAMP(t_star_mo) OVER w3, 0) AS t_star_quarterly_std,
           -- thetao quarterly
           AVG(thetao_mo) OVER w3   AS thetao_quarterly_avg,
           COALESCE(STDDEV_SAMP(thetao_mo) OVER w3, 0) AS thetao_quarterly_std
       FROM monthly_ocean
       WINDOW w3 AS (
           PARTITION BY y, x
           ORDER BY month_idx
           RANGE BETWEEN 2 PRECEDING AND CURRENT ROW
       )
   """)
   con.execute("CREATE INDEX idx_ocean_yx_mo ON ocean_features(y, x, month_idx)")
   print(f"         [{_time.perf_counter() - t0:.1f} s]")


   # ==================================================================
   # 3. GRACE FEATURE ENGINEERING
   #    MAX(lwe_length) for consistency with build_lsp_duckdb.py.
   #    COALESCE(STDDEV_SAMP, 0) for edge months.
   # ==================================================================
   print("[3/6] Engineering GRACE Lags (Quarterly LWE) ...")
   t0 = _time.perf_counter()


   con.execute("""
       CREATE TABLE grace_features AS
       WITH monthly_grace AS (
           SELECT
               mascon_id,
               EXTRACT(YEAR FROM time) * 12 + EXTRACT(MONTH FROM time) AS month_idx,
               MAX(lwe_length) AS lwe_mo
           FROM grace
           GROUP BY mascon_id, month_idx
       )
       SELECT
           mascon_id, month_idx,
           lwe_mo,
           AVG(lwe_mo) OVER w3   AS lwe_quarterly_avg,
           COALESCE(STDDEV_SAMP(lwe_mo) OVER w3, 0) AS lwe_quarterly_std
       FROM monthly_grace
       WINDOW w3 AS (
           PARTITION BY mascon_id
           ORDER BY month_idx
           RANGE BETWEEN 2 PRECEDING AND CURRENT ROW
       )
   """)
   con.execute("CREATE INDEX idx_grace_mid_mo ON grace_features(mascon_id, month_idx)")
   print(f"         [{_time.perf_counter() - t0:.1f} s]")


   # ==================================================================
   # 4. ICESAT-2 BACKBONE  +  STATIC JOIN
   #    Full feature set from both tables:
   #      ICESat-2: delta_h, ice_area, surface_slope, h_surface_dynamic
   #      Static:   surface, bed, thickness, bed_slope,
   #                dist_to_grounding_line, clamped_depth,
   #                dist_to_ocean, ice_draft, mascon_id
   # ==================================================================
   print("[4/6] Establishing ICESat-2 Sparse Backbone (full feature set) ...")
   t0 = _time.perf_counter()


   con.execute("""
       CREATE TABLE sparse_backbone AS
       SELECT
           i.y, i.x, i.time AS exact_time,
           (EXTRACT(YEAR FROM i.time) * 12
            + EXTRACT(MONTH FROM i.time)) AS month_idx,
           -- ICESat-2 dynamic
           i.delta_h, i.ice_area, i.surface_slope, i.h_surface_dynamic,
           -- Static topography & geometry
           s.mascon_id,
           s.surface, s.bed, s.thickness,
           s.bed_slope, s.dist_to_grounding_line,
           s.clamped_depth, s.dist_to_ocean, s.ice_draft
       FROM icesat i
       LEFT JOIN static s ON i.y = s.y AND i.x = s.x
   """)
   print(f"         [{_time.perf_counter() - t0:.1f} s]")


   # -- Join-hit-rate diagnostic (backbone) --
   hit = con.execute("""
       SELECT
           COUNT(*)          AS total,
           COUNT(mascon_id)  AS with_mascon,
           COUNT(thickness)  AS with_static,
           ROUND(COUNT(thickness) * 100.0 / NULLIF(COUNT(*), 0), 1) AS static_hit_pct
       FROM sparse_backbone
   """).fetchone()
   print(f"  -> Backbone: {hit[0]:,} rows | "
         f"static join: {hit[3]}% | mascon: {hit[1]:,}")
   if hit[3] is not None and hit[3] < 50:
       print("  ** WARNING: Static join hit rate < 50% "
             "-- check (y, x) coordinate alignment!")


   # ==================================================================
   # 5. ENRICHED TABLE  (4-way join)
   #    mascon_aggregates uses SUM(ABS(delta_h)) so the proportional
   #    distribution never flips sign in mixed thinning/thickening mascons.
   # ==================================================================
   print("[5/6] Enriching backbone (ocean + GRACE + mascon aggregates) ...")
   t0 = _time.perf_counter()


   con.execute("""
       CREATE TABLE enriched AS
       WITH mascon_aggregates AS (
           SELECT
               mascon_id, month_idx,
               COUNT(delta_h)      AS n_pix,
               SUM(ABS(delta_h))   AS sum_abs_dh
           FROM sparse_backbone
           WHERE mascon_id IS NOT NULL
           GROUP BY mascon_id, month_idx
       )
       SELECT
           b.*,
           -- Ocean monthly + quarterly
           o.thetao_mo,  o.t_star_mo,  o.so_mo,  o.t_f_mo,
           o.t_star_quarterly_avg,  o.t_star_quarterly_std,
           o.thetao_quarterly_avg,  o.thetao_quarterly_std,
           -- GRACE monthly + quarterly
           g.lwe_mo,  g.lwe_quarterly_avg,  g.lwe_quarterly_std,
           -- Mascon aggregates (internal, dropped before write)
           ma.n_pix,  ma.sum_abs_dh
       FROM sparse_backbone b
       LEFT JOIN ocean_features o
           ON b.y = o.y AND b.x = o.x AND b.month_idx = o.month_idx
       LEFT JOIN grace_features g
           ON b.mascon_id = g.mascon_id AND b.month_idx = g.month_idx
       LEFT JOIN mascon_aggregates ma
           ON b.mascon_id = ma.mascon_id AND b.month_idx = ma.month_idx
   """)
   print(f"         [{_time.perf_counter() - t0:.1f} s]")


   # -- Join-hit-rate diagnostics (enriched) --
   diag = con.execute("""
       SELECT
           COUNT(*)           AS total,
           COUNT(thetao_mo)   AS with_ocean,
           COUNT(lwe_mo)      AS with_grace,
           ROUND(COUNT(thetao_mo) * 100.0 / NULLIF(COUNT(*), 0), 1) AS ocean_pct,
           ROUND(COUNT(lwe_mo)    * 100.0 / NULLIF(COUNT(*), 0), 1) AS grace_pct
       FROM enriched
   """).fetchone()
   print(f"  -> Enriched: {diag[0]:,} rows | "
         f"ocean: {diag[3]}% | grace: {diag[4]}%")


   # Free intermediates before the heavy COPY.
   con.execute("DROP TABLE IF EXISTS sparse_backbone")
   con.execute("DROP TABLE IF EXISTS ocean_features")
   con.execute("DROP TABLE IF EXISTS grace_features")


   # ==================================================================
   # 6. COMPUTE lwe_fused  &  WRITE HIVE-PARTITIONED PARQUET
   #
   #    ABS-weighted Signal Fusion
   #    --------------------------
   #    Distributes mascon-level LWE to pixels proportionally to
   #    |delta_h|.  Every pixel in the mascon receives the same sign
   #    as lwe_mo, preventing the physical sign-flip that occurs when
   #    a mascon contains both thinning and thickening pixels.
   #
   #    Conservation:  SUM(lwe_fused) = n_pix * lwe_mo  (mass-preserving)
   #    Average:       AVG(lwe_fused) = lwe_mo
   #
   #    Fallback: when mean(|delta_h|) < 0.1 mm (below altimetric noise),
   #    ICESat-2 cannot resolve the spatial pattern -> uniform distribution.
   # ==================================================================
   print("[6/6] Writing Hive-Partitioned Parquet ...")
   t0 = _time.perf_counter()


   if os.path.exists(ML_FEATURE_DIR):
       shutil.rmtree(ML_FEATURE_DIR)


   query = f"""
       COPY (
           SELECT
               y, x, exact_time, month_idx,
               -- Static topography & geometry
               mascon_id,
               surface, bed, thickness, bed_slope,
               dist_to_grounding_line, clamped_depth,
               dist_to_ocean, ice_draft,
               -- ICESat-2 kinematics
               delta_h, ice_area, surface_slope, h_surface_dynamic,
               -- Ocean thermodynamics (monthly)
               thetao_mo, t_star_mo, so_mo, t_f_mo,
               -- Ocean thermodynamics (quarterly rolling)
               t_star_quarterly_avg, t_star_quarterly_std,
               thetao_quarterly_avg, thetao_quarterly_std,
               -- GRACE mass (monthly + quarterly)
               lwe_mo, lwe_quarterly_avg, lwe_quarterly_std,


               -- ABS-weighted Constrained Forward Modeling
               CASE
                   WHEN mascon_id IS NOT NULL
                    AND delta_h   IS NOT NULL
                    AND lwe_mo    IS NOT NULL
                    AND n_pix     > 0
                   THEN
                       CASE
                           WHEN (sum_abs_dh / n_pix) > {FUSION_MIN_MEAN_DH_M}
                            AND sum_abs_dh > 0
                           THEN (ABS(delta_h) / sum_abs_dh) * n_pix * lwe_mo
                           ELSE lwe_mo
                       END
                   ELSE NULL
               END AS lwe_fused


           FROM enriched
       ) TO '{ML_FEATURE_DIR}' (
           FORMAT PARQUET,
           COMPRESSION ZSTD,
           PARTITION_BY (month_idx)
       )
   """


   con.execute(query)


   elapsed = _time.perf_counter() - t0
   print(f"  -> Fusion + write complete in {elapsed:.1f} seconds.")


   # -- Cleanup --
   con.execute("DROP TABLE IF EXISTS enriched")
   con.close()


   wall = _time.perf_counter() - wall_t
   print("\n" + "=" * 72)
   print(f" SPARSE ML FEATURE STORE COMPLETE [{wall:.1f} s  ({wall / 60:.1f} min)]")
   print("=" * 72)




if __name__ == "__main__":
   build_observation_anchored_dataset()
