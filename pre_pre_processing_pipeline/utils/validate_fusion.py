"""
validate_fusion.py
-------------------------------------------------------------------------------
COMPUTATIONAL GLACIOLOGY - POST-FUSION VALIDATION
-------------------------------------------------------------------------------
Validates the hive-partitioned Parquet feature store written by fuse_data.py.
All analytics run through DuckDB out-of-core (memory_limit deliberately low)
so the validator never needs to fit the dataset in RAM.


Check Catalogue
---------------
 1. Read & Structure    : hive partition scan, row count, file/partition census
 2. Schema              : all 31 expected columns present
 3. Anchor Null Integrity: (y, x, exact_time, month_idx, delta_h) never NULL
 4. Feature Sparsity    : per-column non-NULL coverage for all features
 5. COALESCE Integrity  : quarterly-std columns have no NULL where parent present
 6. LWE Fused Bounds    : min/max/mean, percentiles, amplification vs lwe_mo
 7. Sign-Flip           : SIGN(lwe_fused) == SIGN(lwe_mo) for every fused pixel
 8. Mass Conservation   : SUM(lwe_fused) == n_pix * lwe_mo per mascon-month


Exit Codes
----------
 0  All checks passed.
 1  One or more checks failed (or fatal error).
-------------------------------------------------------------------------------
"""


import os
import sys
import time as _time
import duckdb


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
DIR_FLATTENED = "data/flattened"
ML_FEATURE_DIR = os.path.join(DIR_FLATTENED, "antarctica_sparse_features.parquet")


MEMORY_LIMIT            = "8GB"   # low on purpose → forces out-of-core
THREADS                 = 8
TEMP_DIRECTORY          = os.path.join(DIR_FLATTENED, "_duckdb_validate_tmp")
MAX_TEMP_DIRECTORY_SIZE = "100GiB"


# Float-precision tolerance for mass conservation.
# float64 accumulation across ~10k pixels with values in [-500, 500] mm
# yields worst-case ULP drift of O(1e-9).  1e-3 gives 6 orders of headroom.
MASS_TOLERANCE = 1e-3


# Expected output columns (must match fuse_data.py final SELECT exactly).
EXPECTED_COLUMNS = [
   # Coordinates & time
   "y", "x", "exact_time", "month_idx",
   # Static topography & geometry
   "mascon_id",
   "surface", "bed", "thickness", "bed_slope",
   "dist_to_grounding_line", "clamped_depth",
   "dist_to_ocean", "ice_draft",
   # ICESat-2 kinematics
   "delta_h", "ice_area", "surface_slope", "h_surface_dynamic",
   # Ocean thermodynamics (monthly)
   "thetao_mo", "t_star_mo", "so_mo", "t_f_mo",
   # Ocean thermodynamics (quarterly rolling)
   "t_star_quarterly_avg", "t_star_quarterly_std",
   "thetao_quarterly_avg", "thetao_quarterly_std",
   # GRACE mass (monthly + quarterly)
   "lwe_mo", "lwe_quarterly_avg", "lwe_quarterly_std",
   # Fusion target
   "lwe_fused",
]


# Columns that must NEVER be NULL (ICESat-2 observation backbone).
ANCHOR_COLUMNS = ["y", "x", "exact_time", "month_idx", "delta_h"]


# Quarterly-std columns that were COALESCED to 0 in fuse_data.py.
# Each maps to the parent column whose non-NULL rows must also have
# a non-NULL std (otherwise the COALESCE failed).
COALESCED_STD = {
   "t_star_quarterly_std":  "t_star_mo",
   "thetao_quarterly_std":  "thetao_mo",
   "lwe_quarterly_std":     "lwe_mo",
}




# ---------------------------------------------------------
# VALIDATION
# ---------------------------------------------------------


def run_validation():
   wall_t = _time.perf_counter()
   failures = []                       # accumulate all check failures


   # ── Pre-flight ────────────────────────────────────────────────────
   if not os.path.exists(ML_FEATURE_DIR):
       print(f"[FATAL] Feature store not found: {ML_FEATURE_DIR}")
       return 1


   con = duckdb.connect(database=":memory:")
   try:
       con.execute(f"SET memory_limit = '{MEMORY_LIMIT}'")
       con.execute(f"SET threads = {THREADS}")
       con.execute(f"SET temp_directory = '{TEMP_DIRECTORY}'")
       con.execute(f"PRAGMA max_temp_directory_size='{MAX_TEMP_DIRECTORY_SIZE}'")


       print("=" * 72)
       print(" POST-FUSION VALIDATION: ANTARCTICA ML FEATURE STORE")
       print(f" Target: {ML_FEATURE_DIR}")
       print("=" * 72)


       # ==============================================================
       # 1. READ & STRUCTURE
       # ==============================================================
       print("\n[1/8] Registering Partitioned Parquet View ...")
       con.execute(f"""
           CREATE VIEW ml_data AS
           SELECT *
           FROM read_parquet(
               '{ML_FEATURE_DIR}/**/*.parquet',
               hive_partitioning = true
           )
       """)


       total_rows = con.execute("SELECT COUNT(*) FROM ml_data").fetchone()[0]
       print(f"  -> {total_rows:,} rows scanned.")


       if total_rows == 0:
           print("[FATAL] Feature store contains 0 rows.")
           return 1


       # Partition & file census
       n_partitions = sum(
           1 for d in os.listdir(ML_FEATURE_DIR)
           if os.path.isdir(os.path.join(ML_FEATURE_DIR, d))
       )
       n_files = sum(
           len(files)
           for _, _, files in os.walk(ML_FEATURE_DIR)
           if files
       )
       total_bytes = sum(
           os.path.getsize(os.path.join(dp, f))
           for dp, _, fnames in os.walk(ML_FEATURE_DIR)
           for f in fnames if f.endswith(".parquet")
       )
       print(f"  -> {n_partitions} hive partitions, "
             f"{n_files} parquet files, "
             f"{total_bytes / 1e9:.2f} GB on disk.")


       # ==============================================================
       # 2. SCHEMA VALIDATION
       # ==============================================================
       print("\n[2/8] Validating Schema ...")
       actual_cols = {
           r[0] for r in con.execute(
               "SELECT column_name FROM information_schema.columns "
               "WHERE table_name = 'ml_data'"
           ).fetchall()
       }
       expected_set = set(EXPECTED_COLUMNS)
       missing = sorted(expected_set - actual_cols)
       extra   = sorted(actual_cols - expected_set)


       if missing:
           msg = f"Missing columns: {missing}"
           failures.append(msg)
           print(f"  [✗] {msg}")
       else:
           print(f"  [✓] All {len(EXPECTED_COLUMNS)} expected columns present.")


       if extra:
           print(f"  [i] Extra columns (not a failure): {extra}")


       # If critical columns are missing, remaining checks would error.
       critical_missing = {"lwe_fused", "lwe_mo", "delta_h", "mascon_id"} & set(missing)
       if critical_missing:
           print(f"[FATAL] Cannot continue — critical columns missing: "
                 f"{sorted(critical_missing)}")
           _print_summary(failures, wall_t)
           return 1


       # ==============================================================
       # 3. ANCHOR NULL INTEGRITY
       # ==============================================================
       print("\n[3/8] Checking Anchor Column Null Integrity ...")
       for col in ANCHOR_COLUMNS:
           if col not in actual_cols:
               msg = f"Anchor column '{col}' missing from schema"
               failures.append(msg)
               print(f"  [✗] {msg}")
               continue
           null_n = con.execute(
               f"SELECT COUNT(*) FROM ml_data WHERE \"{col}\" IS NULL"
           ).fetchone()[0]
           if null_n > 0:
               msg = f"Anchor column '{col}' has {null_n:,} NULLs (must be 0)"
               failures.append(msg)
               print(f"  [✗] {msg}")
           else:
               print(f"  [✓] {col}: 0 NULLs")


       # ==============================================================
       # 4. FEATURE SPARSITY & COVERAGE
       # ==============================================================
       print("\n[4/8] Validating Feature Sparsity ...")
       t0 = _time.perf_counter()


       # Build COUNT(col) for every expected column that exists
       countable = [c for c in EXPECTED_COLUMNS if c in actual_cols]
       count_exprs = ", ".join(f'COUNT("{c}") AS "{c}"' for c in countable)
       counts_row = con.execute(
           f"SELECT {count_exprs} FROM ml_data"
       ).fetchone()


       pct = lambda n: n / total_rows * 100
       for col, cnt in zip(countable, counts_row):
           marker = "✓" if cnt == total_rows else "~"
           print(f"  [{marker}] {col:30s}  {cnt:>14,}  ({pct(cnt):5.1f}%)")


       print(f"      [{_time.perf_counter() - t0:.1f} s]")


       # ==============================================================
       # 5. COALESCE(STDDEV_SAMP, 0) INTEGRITY
       # ==============================================================
       print("\n[5/8] Verifying COALESCE Integrity on Quarterly Std Columns ...")
       for std_col, parent_col in COALESCED_STD.items():
           if std_col not in actual_cols or parent_col not in actual_cols:
               print(f"  [i] Skipped {std_col} (column missing)")
               continue
           null_n = con.execute(f"""
               SELECT COUNT(*) FROM ml_data
               WHERE "{parent_col}" IS NOT NULL
                 AND "{std_col}" IS NULL
           """).fetchone()[0]
           if null_n > 0:
               msg = (f"COALESCE failure: '{std_col}' has {null_n:,} NULLs "
                      f"where '{parent_col}' is non-NULL")
               failures.append(msg)
               print(f"  [✗] {msg}")
           else:
               print(f"  [✓] {std_col}: properly COALESCED "
                     f"(0 NULLs where {parent_col} present)")


       # ==============================================================
       # 6. LWE FUSED PHYSICS BOUNDS
       # ==============================================================
       print("\n[6/8] Validating LWE Fused Physics Bounds ...")
       t0 = _time.perf_counter()


       fused_n = con.execute(
           "SELECT COUNT(lwe_fused) FROM ml_data"
       ).fetchone()[0]


       if fused_n == 0:
           print("  [!] No lwe_fused values — skipping bounds & downstream.")
       else:
           bounds = con.execute("""
               SELECT
                   MIN(lwe_fused),
                   MAX(lwe_fused),
                   AVG(lwe_fused),
                   PERCENTILE_CONT(0.01) WITHIN GROUP (ORDER BY lwe_fused),
                   PERCENTILE_CONT(0.99) WITHIN GROUP (ORDER BY lwe_fused),
                   MIN(lwe_mo),
                   MAX(lwe_mo)
               FROM ml_data
               WHERE lwe_fused IS NOT NULL
           """).fetchone()
           f_min, f_max, f_avg, p01, p99, lm_min, lm_max = bounds


           def _fmt(v):
               return f"{v:.4f}" if v is not None else "NULL"


           print(f"  -> lwe_fused range:   [{_fmt(f_min)}, {_fmt(f_max)}] mm")
           print(f"  -> lwe_fused mean:    {_fmt(f_avg)} mm")
           print(f"  -> lwe_fused p01-p99: [{_fmt(p01)}, {_fmt(p99)}] mm")
           print(f"  -> lwe_mo    range:   [{_fmt(lm_min)}, {_fmt(lm_max)}] mm")


           # Amplification ratio: max |fused| relative to max |lwe_mo|
           max_abs_fused = max(abs(f_min), abs(f_max))
           max_abs_lwe   = max(abs(lm_min), abs(lm_max)) if lm_min is not None else 1
           amp = max_abs_fused / max(max_abs_lwe, 1e-12)
           print(f"  -> Amplification:     {amp:.1f}x  "
                 f"(max |fused| / max |lwe_mo|)")


       print(f"      [{_time.perf_counter() - t0:.1f} s]")


       # ==============================================================
       # 7. SIGN-FLIP VERIFICATION
       #    After ABS-weighting, every pixel's lwe_fused must share
       #    the sign of lwe_mo.  Zero values are excluded (no sign).
       # ==============================================================
       print("\n[7/8] Verifying No Sign-Flip Anomalies ...")
       t0 = _time.perf_counter()


       if fused_n == 0:
           print("  [i] Skipped (no fused values).")
       else:
           sign_violations = con.execute("""
               SELECT COUNT(*) FROM ml_data
               WHERE lwe_fused IS NOT NULL
                 AND lwe_mo    IS NOT NULL
                 AND lwe_fused != 0
                 AND lwe_mo    != 0
                 AND SIGN(lwe_fused) != SIGN(lwe_mo)
           """).fetchone()[0]


           if sign_violations > 0:
               msg = (f"Sign-flip: {sign_violations:,} pixels where "
                      f"SIGN(lwe_fused) ≠ SIGN(lwe_mo)")
               failures.append(msg)
               print(f"  [✗] {msg}")
           else:
               print(f"  [✓] Zero sign-flip anomalies across "
                     f"{fused_n:,} fused pixels.")


       print(f"      [{_time.perf_counter() - t0:.1f} s]")


       # ==============================================================
       # 8. STRICT MASS CONSERVATION
       #    Invariant:  SUM(lwe_fused) = COUNT(*) * lwe_mo
       #    per (mascon_id, month_idx) group, within float tolerance.
       #
       #    Derivation (ABS-weighted path):
       #      lwe_fused_i = (|dh_i| / Σ|dh|) * n_pix * lwe_mo
       #      Σ lwe_fused = n_pix * lwe_mo * Σ|dh_i| / Σ|dh| = n_pix * lwe_mo
       #
       #    Derivation (fallback path, mean|dh| < threshold):
       #      lwe_fused_i = lwe_mo   (uniform distribution)
       #      Σ lwe_fused = n_pix * lwe_mo
       #
       #    Since all rows where lwe_fused IS NOT NULL have both
       #    mascon_id and lwe_mo non-NULL, and lwe_mo is constant
       #    within a (mascon_id, month_idx) group:
       #      expected = MAX(lwe_mo) * COUNT(*)
       # ==============================================================
       print("\n[8/8] Executing Strict Mass Conservation Proof ...")
       t0 = _time.perf_counter()


       if fused_n == 0:
           print("  [i] Skipped (no fused values).")
       else:
           mass_check = con.execute(f"""
               WITH mascon_totals AS (
                   SELECT
                       mascon_id,
                       month_idx,
                       COUNT(*)                     AS n_rows,
                       SUM(lwe_fused)               AS sum_fused,
                       MAX(lwe_mo) * COUNT(delta_h) AS expected_total
                   FROM ml_data
                   WHERE lwe_fused IS NOT NULL
                   GROUP BY mascon_id, month_idx
               )
               SELECT
                   COUNT(*)  AS total_mascon_months,
                   SUM(n_rows) AS total_fused_pixels,
                   COUNT(*) FILTER (
                       WHERE ABS(sum_fused - expected_total) > {MASS_TOLERANCE}
                   ) AS violations,
                   MAX(ABS(sum_fused - expected_total)) AS max_drift,
                   AVG(ABS(sum_fused - expected_total)) AS avg_drift
               FROM mascon_totals
           """).fetchone()


           mm_total, mm_pix, mm_viol, mm_max, mm_avg = mass_check


           print(f"  -> Mascon-Months evaluated:  {mm_total:,}")
           print(f"  -> Total fused pixels:       {mm_pix:,}")
           if mm_max is not None:
               print(f"  -> Max float drift:          {mm_max:.9f} mm")
               print(f"  -> Avg float drift:          {mm_avg:.9f} mm")
           else:
               print(f"  -> Drift: N/A (single-row groups)")


           if mm_viol > 0:
               msg = (f"Mass conservation violated in {mm_viol:,} / "
                      f"{mm_total:,} mascon-months (tol={MASS_TOLERANCE})")
               failures.append(msg)
               print(f"  [✗] CRITICAL: {msg}")


               # Show worst offenders for debugging
               worst = con.execute(f"""
                   WITH mascon_totals AS (
                       SELECT
                           mascon_id, month_idx,
                           COUNT(*) AS n_pix,
                           SUM(lwe_fused) AS sum_fused,
                           MAX(lwe_mo) * COUNT(delta_h) AS expected
                       FROM ml_data
                       WHERE lwe_fused IS NOT NULL
                       GROUP BY mascon_id, month_idx
                   )
                   SELECT mascon_id, month_idx, n_pix,
                          sum_fused, expected,
                          ABS(sum_fused - expected) AS drift
                   FROM mascon_totals
                   WHERE ABS(sum_fused - expected) > {MASS_TOLERANCE}
                   ORDER BY drift DESC
                   LIMIT 5
               """).fetchall()
               print("  -> Worst offenders:")
               for r in worst:
                   print(f"     mascon={r[0]}  month={r[1]}  n={r[2]}  "
                         f"Σfused={r[3]:.4f}  expected={r[4]:.4f}  "
                         f"drift={r[5]:.6f}")
           else:
               print(f"  [✓] PHYSICS VERIFIED: 100% of {mm_total:,} mascon-months "
                     f"conserve mass within {MASS_TOLERANCE} tolerance.")


       print(f"      [{_time.perf_counter() - t0:.1f} s]")


   finally:
       con.close()


   # ── Summary ───────────────────────────────────────────────────────
   _print_summary(failures, wall_t)
   return 1 if failures else 0




def _print_summary(failures, wall_t):
   """Print final pass/fail summary."""
   wall = _time.perf_counter() - wall_t
   print("\n" + "=" * 72)
   if failures:
       print(f" VALIDATION FAILED  [{wall:.1f} s]")
       print(f" {len(failures)} failure(s):")
       for i, f in enumerate(failures, 1):
           print(f"   {i}. {f}")
   else:
       print(f" VALIDATION PASSED  [{wall:.1f} s]")
   print("=" * 72)




if __name__ == "__main__":
   sys.exit(run_validation())
