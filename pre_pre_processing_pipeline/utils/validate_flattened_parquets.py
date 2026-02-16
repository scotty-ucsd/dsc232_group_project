"""
validate_flattened_parquet.py


-------------------------------------------------------------------------------
COMPUTATIONAL GLACIOLOGY - OPTIMISED PIPELINE
-------------------------------------------------------------------------------
DATE:   2026-02-14
STATUS: PRODUCTION


Pre-flight validation of the four flat Parquet tables produced by
flatten_to_parquet.py, run BEFORE build_lsp.py / build_lsp_duckdb.py.


Targets
-------
 1. bedmap3_static.parquet     (static 2D topography + mascon + ocean 2D)
 2. icesat2_dynamic.parquet    (3D elevation change kinematics)
 3. ocean_dynamic.parquet      (3D ocean T, S, T_f, T*)
 4. grace.parquet              (mascon mass anomalies)


Check Catalogue (per table)
---------------------------
 A. Existence & readability
 B. Schema:  required columns present, data types correct
 C. Null integrity:  join keys (y, x, time, mascon_id) non-null
 D. No duplicates on join keys
 E. CRS plausibility:  y, x within EPSG:3031 Antarctic bounds
 F. Temporal plausibility:  time ∈ [2018, 2026)
 G. Physical ranges:  elevation, temperature, salinity, etc.
 H. Cross-table compatibility:
      - static (y, x) ⊇ icesat2 (y, x)  and  ocean (y, x)
      - icesat2 and ocean share overlapping time domain
      - GRACE mascon_ids referenced by static exist in grace
 I. Row-count and file-size diagnostics


Architecture
------------
DuckDB in-process for all analytics — zero-copy Parquet memory-mapping,
out-of-core if needed, and SQL aggregations are faster than Pandas on
multi-GB tables.  PyArrow is used only for lightweight schema introspection
(reading Parquet metadata without scanning row groups).


Exit Codes
----------
 0  All checks passed.
 1  One or more checks failed.
 2  Fatal error (missing files, unreadable Parquet, etc.).
-------------------------------------------------------------------------------
"""


import os
import sys
import time as _time
import argparse


import duckdb
import pyarrow.parquet as pq


# ── Configuration ───────────────────────────────────────────────────────────


DIR_FLATTENED = "data/flattened"


STATIC_PATH = os.path.join(DIR_FLATTENED, "bedmap3_static.parquet")
ICESAT_PATH = os.path.join(DIR_FLATTENED, "icesat2_dynamic.parquet")
OCEAN_PATH  = os.path.join(DIR_FLATTENED, "ocean_dynamic_clean.parquet")
GRACE_PATH  = os.path.join(DIR_FLATTENED, "grace_clean.parquet")


# ── Expected Schemas ────────────────────────────────────────────────────────
#
# Maps each table to its REQUIRED columns.  Extra columns are tolerated
# (e.g. spatial_ref), but missing required columns are fatal.
#
# Type families:  'float' matches float32/float64/double,
#                 'int'   matches int8..int64/uint8..uint64,
#                 'ts'    matches timestamp[*],
#                 'any'   matches anything.


SCHEMA_STATIC = {
   "y":                       "float",
   "x":                       "float",
   "surface":                 "float",
   "bed":                     "float",
   "thickness":               "float",
   "mask":                    "any",     # int or float depending on source
   "mascon_id":               "float",
   "bed_slope":               "float",
   "dist_to_grounding_line":  "float",
   "clamped_depth":           "float",
   "dist_to_ocean":           "float",
   "ice_draft":               "float",
}


SCHEMA_ICESAT = {
   "y":                  "float",
   "x":                  "float",
   "time":               "ts",
   "delta_h":            "float",
   "ice_area":           "float",
   "h_surface_dynamic":  "float",
   "surface_slope":      "float",
}


SCHEMA_OCEAN = {
   "y":      "float",
   "x":      "float",
   "time":   "ts",
   "thetao": "float",
   "so":     "float",
   "T_f":    "float",
   "T_star": "float",
}


SCHEMA_GRACE = {
   "mascon_id":  "any",   # float or int depending on flatten stage
   "time":       "ts",
   "lwe_length": "float",
}


# ── EPSG:3031 coordinate bounds (Antarctic Polar Stereographic) ─────────
#
# The Bedmap3 500 m grid spans roughly:
#   y ∈ [−3_333_000,  3_333_000]   x ∈ [−3_333_000,  3_333_000]
# We use generous bounds to catch gross CRS errors.
EPSG3031_MIN = -3_500_000.0
EPSG3031_MAX =  3_500_000.0


# ── Temporal bounds ─────────────────────────────────────────────────────
TIME_MIN = "2018-01-01"
TIME_MAX = "2026-07-01"


# ── Physical range bounds ───────────────────────────────────────────────
#
# Antarctic-specific plausible ranges.  Exceeding these does not
# necessarily mean the data is wrong, but flags it for human review.
#
PHYS_RANGES = {
   # (min, max, units, description)
   "surface":                 (-500,   5000,  "m",   "Ice surface elevation"),
   "bed":                     (-3000,  5000,  "m",   "Bed topography"),
   "thickness":               (0,      5000,  "m",   "Ice thickness"),
   "bed_slope":               (0,      10,    "m/m", "Bed slope magnitude"),
   "dist_to_grounding_line":  (0,      2e6,   "m",   "Distance to GL"),
   "clamped_depth":           (0,      6000,  "m",   "Clamped ocean depth"),
   "dist_to_ocean":           (0,      1.5e6, "m",   "Distance to ocean pixel"),
   "ice_draft":               (-3000,  500,   "m",   "Ice draft"),
   "delta_h":                 (-200,   200,   "m",   "Elevation change"),
   "ice_area":                (0,      1e12,  "m²",  "Ice area"),
   "h_surface_dynamic":       (-500,   5000,  "m",   "Dynamic surface elev"),
   "surface_slope":           (0,      10,    "m/m", "Surface slope magnitude"),
   "thetao":                  (-3.5,   6.0,   "°C",  "Potential temperature"),
   "so":                      (25.0,   38.0,  "PSU", "Practical salinity"),
   "T_f":                     (-5.0,   1.0,   "°C",  "Freezing temperature"),
   "T_star":                  (-3.0,   10.0,  "°C",  "Thermal driving"),
   "lwe_length":              (-500,   500,   "mm",  "LWE anomaly"),
}




# ── Result tracker ──────────────────────────────────────────────────────────


class CheckTracker:
   """Accumulates pass / fail / warn counts and provides a verdict."""


   def __init__(self):
       self.n_pass = 0
       self.n_fail = 0
       self.n_warn = 0
       self.n_skip = 0


   def ok(self, label: str, passed: bool, detail: str = "") -> bool:
       if passed:
           self.n_pass += 1
           sym = "✓"
       else:
           self.n_fail += 1
           sym = "✗"
       line = f"  [{sym}]  {label}"
       if detail:
           line += f"  —  {detail}"
       print(line)
       return passed


   def warn(self, label: str, detail: str = ""):
       self.n_warn += 1
       line = f"  [⚠]  {label}"
       if detail:
           line += f"  —  {detail}"
       print(line)


   def skip(self, label: str, reason: str = ""):
       self.n_skip += 1
       line = f"  [-]  {label} SKIPPED"
       if reason:
           line += f"  —  {reason}"
       print(line)


   @property
   def total(self):
       return self.n_pass + self.n_fail


   @property
   def passed(self):
       return self.n_fail == 0




T = CheckTracker()




def _hdr(title: str) -> None:
   print(f"\n{'─' * 72}")
   print(f"  {title}")
   print(f"{'─' * 72}")




# ── Parquet path helpers ────────────────────────────────────────────────────


def _parquet_glob(path: str) -> str:
   """Return a DuckDB-compatible glob for a Parquet source."""
   if os.path.isdir(path):
       return os.path.join(path, "**", "*.parquet")
   return path




def _count_parquet_files(path: str) -> int:
   """Count .parquet files under a path (file or directory)."""
   if os.path.isfile(path) and path.endswith(".parquet"):
       return 1
   if os.path.isdir(path):
       n = 0
       for root, _, files in os.walk(path):
           for f in files:
               if f.endswith(".parquet"):
                   n += 1
       return n
   return 0




def _dir_size_bytes(path: str) -> int:
   """Total bytes of all files under path."""
   if os.path.isfile(path):
       return os.path.getsize(path)
   total = 0
   for root, _, files in os.walk(path):
       for f in files:
           total += os.path.getsize(os.path.join(root, f))
   return total




def _human_size(nbytes: int) -> str:
   for unit in ("B", "KB", "MB", "GB", "TB"):
       if abs(nbytes) < 1024:
           return f"{nbytes:.1f} {unit}"
       nbytes /= 1024
   return f"{nbytes:.1f} PB"




# ── Type-family matching ────────────────────────────────────────────────────


def _type_family(arrow_type) -> str:
   """Classify a PyArrow type into a family string."""
   import pyarrow as pa
   t = arrow_type
   if pa.types.is_floating(t) or pa.types.is_decimal(t):
       return "float"
   if pa.types.is_integer(t):
       return "int"
   if pa.types.is_timestamp(t) or pa.types.is_date(t):
       return "ts"
   return str(t)




def _family_matches(actual_family: str, expected_family: str) -> bool:
   if expected_family == "any":
       return True
   if expected_family == "float":
       return actual_family in ("float", "int")  # int → float is safe
   if expected_family == "int":
       return actual_family == "int"
   if expected_family == "ts":
       return actual_family == "ts"
   return actual_family == expected_family




# ── Per-table validation ────────────────────────────────────────────────────


def validate_existence(name: str, path: str) -> bool:
   """Check A: file/dir exists and contains ≥ 1 .parquet file."""
   exists = os.path.exists(path)
   T.ok(f"{name}: path exists", exists, path)
   if not exists:
       return False
   n_files = _count_parquet_files(path)
   T.ok(f"{name}: contains Parquet files", n_files > 0,
        f"{n_files} file(s)")
   return n_files > 0




def validate_schema(name: str, path: str,
                   expected: dict[str, str]) -> bool:
   """Check B: required columns present with correct type families."""
   all_ok = True
   try:
       # Read schema from the first Parquet file (no row-group scan).
       if os.path.isdir(path):
           first = None
           for root, _, files in os.walk(path):
               for f in sorted(files):
                   if f.endswith(".parquet"):
                       first = os.path.join(root, f)
                       break
               if first:
                   break
           if first is None:
               T.ok(f"{name}: schema readable", False, "no .parquet files")
               return False
       else:
           first = path


       schema = pq.read_schema(first)
       col_names = set(schema.names)
   except Exception as e:
       T.ok(f"{name}: schema readable", False, str(e))
       return False


   T.ok(f"{name}: schema readable", True, f"{len(schema)} columns")


   # Check each required column.
   for col, exp_family in expected.items():
       present = col in col_names
       if not present:
           T.ok(f"{name}: column '{col}' present", False)
           all_ok = False
           continue
       actual_family = _type_family(schema.field(col).type)
       fam_ok = _family_matches(actual_family, exp_family)
       T.ok(
           f"{name}: '{col}' type",
           fam_ok,
           f"expected={exp_family}, got={actual_family} "
           f"({schema.field(col).type})",
       )
       if not fam_ok:
           all_ok = False


   # Report extra columns (informational, not a failure).
   extra = col_names - set(expected.keys())
   if extra:
       print(f"    [info]  Extra columns in {name}: {sorted(extra)}")


   return all_ok




def validate_table(
   con: duckdb.DuckDBPyConnection,
   name: str,
   path: str,
   expected_schema: dict[str, str],
   join_keys: list[str],
   has_yx: bool = True,
   has_time: bool = False,
   phys_cols: list[str] | None = None,
) -> dict:
   """
   Run checks C-G on a single table.


   Returns a dict with diagnostic metadata (row_count, columns, etc.)
   for cross-table checks.
   """
   meta = {"name": name, "readable": False, "row_count": 0, "columns": []}


   glob = _parquet_glob(path)


   # ── Register view ────────────────────────────────────────────────
   try:
       con.execute(
           f"CREATE OR REPLACE VIEW {name} AS "
           f"SELECT * FROM read_parquet('{glob}', union_by_name=true)"
       )
   except Exception as e:
       T.ok(f"{name}: DuckDB readable", False, str(e))
       return meta


   meta["readable"] = True


   # Row count
   row_count = con.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
   meta["row_count"] = row_count
   T.ok(f"{name}: row count > 0", row_count > 0, f"{row_count:,} rows")
   if row_count == 0:
       return meta


   # Column names
   cols_info = con.execute(
       f"SELECT column_name, data_type "
       f"FROM information_schema.columns "
       f"WHERE table_name = '{name}' "
       f"ORDER BY ordinal_position"
   ).fetchall()
   meta["columns"] = [r[0] for r in cols_info]
   meta["types"] = {r[0]: r[1] for r in cols_info}


   # ── C. Null integrity on join keys ───────────────────────────────
   for key in join_keys:
       if key not in meta["columns"]:
           continue
       n_null = con.execute(
           f'SELECT COUNT(*) FROM {name} WHERE "{key}" IS NULL'
       ).fetchone()[0]
       T.ok(
           f"{name}: '{key}' no nulls",
           n_null == 0,
           f"{n_null:,} null(s)" if n_null else "",
       )


   # ── D. Duplicate check on composite join key ─────────────────────
   key_cols_present = [k for k in join_keys if k in meta["columns"]]
   if key_cols_present:
       key_list = ", ".join(f'"{k}"' for k in key_cols_present)
       n_dup = con.execute(f"""
           SELECT COUNT(*) FROM (
               SELECT {key_list}, COUNT(*) AS cnt
               FROM {name}
               GROUP BY {key_list}
               HAVING cnt > 1
           ) dups
       """).fetchone()[0]
       T.ok(
           f"{name}: no duplicates on ({', '.join(key_cols_present)})",
           n_dup == 0,
           f"{n_dup:,} duplicate key group(s)" if n_dup else "",
       )


   # ── E. CRS plausibility (y, x in EPSG:3031 range) ───────────────
   if has_yx:
       for coord in ("y", "x"):
           if coord not in meta["columns"]:
               continue
           row = con.execute(
               f'SELECT MIN("{coord}"), MAX("{coord}") FROM {name}'
           ).fetchone()
           cmin, cmax = float(row[0]), float(row[1])
           in_range = cmin >= EPSG3031_MIN and cmax <= EPSG3031_MAX
           T.ok(
               f"{name}: '{coord}' within EPSG:3031 bounds",
               in_range,
               f"[{cmin:,.0f}, {cmax:,.0f}]",
           )


   # ── F. Temporal plausibility ─────────────────────────────────────
   if has_time and "time" in meta["columns"]:
       row = con.execute(
           f"SELECT MIN(time), MAX(time) FROM {name}"
       ).fetchone()
       t_min, t_max = row[0], row[1]


       # DuckDB may return datetime or string; normalise to string
       t_min_s = str(t_min)[:19]
       t_max_s = str(t_max)[:19]


       # Check against bounds
       time_ok = str(t_min) >= TIME_MIN and str(t_max) < TIME_MAX
       T.ok(
           f"{name}: time within [{TIME_MIN}, {TIME_MAX})",
           time_ok,
           f"[{t_min_s}, {t_max_s}]",
       )


       # Count distinct time steps
       n_times = con.execute(
           f"SELECT COUNT(DISTINCT time) FROM {name}"
       ).fetchone()[0]
       meta["n_times"] = n_times
       print(f"    [info]  {n_times} distinct time steps")


   # ── G. Physical range checks ─────────────────────────────────────
   check_cols = phys_cols or []
   for col in check_cols:
       if col not in meta["columns"]:
           continue
       if col not in PHYS_RANGES:
           continue


       pmin, pmax, units, desc = PHYS_RANGES[col]
       row = con.execute(
           f'SELECT MIN("{col}"), MAX("{col}"), '
           f'AVG("{col}"), COUNT("{col}") FROM {name}'
       ).fetchone()
       vmin, vmax, vmean, vcount = (
           float(row[0]) if row[0] is not None else float("nan"),
           float(row[1]) if row[1] is not None else float("nan"),
           float(row[2]) if row[2] is not None else float("nan"),
           int(row[3]),
       )
       in_range = vmin >= pmin and vmax <= pmax
       T.ok(
           f"{name}: '{col}' ∈ [{pmin}, {pmax}] {units}",
           in_range,
           f"[{vmin:.4g}, {vmax:.4g}]  μ={vmean:.4g}  n={vcount:,}",
       )


   return meta




# ── Cross-table checks ──────────────────────────────────────────────────────


def validate_cross_table(
   con: duckdb.DuckDBPyConnection,
   meta_static: dict,
   meta_icesat: dict,
   meta_ocean: dict,
   meta_grace: dict,
):
   """Check H: cross-table join-key compatibility."""
   _hdr("H. Cross-Table Compatibility")


   # ── H1. static (y,x) ⊇ icesat2 (y,x) ───────────────────────────
   # Every ICESat-2 pixel should have a corresponding static row.
   if meta_static["readable"] and meta_icesat["readable"]:
       n_orphan = con.execute("""
           SELECT COUNT(*) FROM (
               SELECT DISTINCT y, x FROM icesat2
               EXCEPT
               SELECT DISTINCT y, x FROM static
           ) orphans
       """).fetchone()[0]
       T.ok(
           "static(y,x) ⊇ icesat2(y,x)",
           n_orphan == 0,
           f"{n_orphan:,} ICESat-2 pixels without static match"
           if n_orphan else "",
       )
   else:
       T.skip("static ⊇ icesat2", "one or both tables unreadable")


   # ── H2. static (y,x) ⊇ ocean (y,x) ─────────────────────────────
   if meta_static["readable"] and meta_ocean["readable"]:
       n_orphan = con.execute("""
           SELECT COUNT(*) FROM (
               SELECT DISTINCT y, x FROM ocean
               EXCEPT
               SELECT DISTINCT y, x FROM static
           ) orphans
       """).fetchone()[0]
       T.ok(
           "static(y,x) ⊇ ocean(y,x)",
           n_orphan == 0,
           f"{n_orphan:,} ocean pixels without static match"
           if n_orphan else "",
       )
   else:
       T.skip("static ⊇ ocean", "one or both tables unreadable")


   # ── H3. icesat2 and ocean overlap in time ────────────────────────
   if meta_icesat["readable"] and meta_ocean["readable"]:
       n_shared = con.execute("""
           SELECT COUNT(*) FROM (
               SELECT DISTINCT time FROM icesat2
               INTERSECT
               SELECT DISTINCT time FROM ocean
           ) shared
       """).fetchone()[0]
       n_ice_times = meta_icesat.get("n_times", 0)
       n_oc_times  = meta_ocean.get("n_times", 0)
       T.ok(
           "icesat2 ∩ ocean temporal overlap",
           n_shared > 0,
           f"{n_shared} shared times  "
           f"(icesat2={n_ice_times}, ocean={n_oc_times})",
       )
   else:
       T.skip("icesat2 ∩ ocean time", "one or both tables unreadable")


   # ── H4. GRACE mascon_ids referenced by static exist in grace ─────
   if meta_static["readable"] and meta_grace["readable"]:
       n_missing = con.execute("""
           SELECT COUNT(DISTINCT CAST(s.mascon_id AS INTEGER)) FROM (
               SELECT DISTINCT CAST(mascon_id AS INTEGER) AS mascon_id
               FROM static
               WHERE mascon_id IS NOT NULL
           ) s
           LEFT JOIN (
               SELECT DISTINCT CAST(mascon_id AS INTEGER) AS mascon_id
               FROM grace
           ) g ON s.mascon_id = g.mascon_id
           WHERE g.mascon_id IS NULL
       """).fetchone()[0]
       T.ok(
           "static mascon_ids ⊆ grace mascon_ids",
           n_missing == 0,
           f"{n_missing} mascon_id(s) in static but not in GRACE"
           if n_missing else "",
       )
   else:
       T.skip("mascon_id coverage", "one or both tables unreadable")


   # ── H5. GRACE covers the full icesat2/ocean time span ────────────
   if meta_grace["readable"] and (meta_icesat["readable"]
                                   or meta_ocean["readable"]):
       grace_range = con.execute(
           "SELECT MIN(time), MAX(time) FROM grace"
       ).fetchone()
       dynamic_min_q = []
       dynamic_max_q = []
       if meta_icesat["readable"]:
           dynamic_min_q.append("SELECT MIN(time) FROM icesat2")
           dynamic_max_q.append("SELECT MAX(time) FROM icesat2")
       if meta_ocean["readable"]:
           dynamic_min_q.append("SELECT MIN(time) FROM ocean")
           dynamic_max_q.append("SELECT MAX(time) FROM ocean")


       d_min = con.execute(
           " UNION ALL ".join(dynamic_min_q)
       ).fetchall()
       d_max = con.execute(
           " UNION ALL ".join(dynamic_max_q)
       ).fetchall()


       dyn_earliest = min(r[0] for r in d_min)
       dyn_latest   = max(r[0] for r in d_max)


       covers = (grace_range[0] <= dyn_earliest
                 and grace_range[1] >= dyn_latest)
       T.ok(
           "GRACE time span ⊇ dynamic tables",
           covers,
           f"GRACE=[{str(grace_range[0])[:10]}, {str(grace_range[1])[:10]}]  "
           f"dynamic=[{str(dyn_earliest)[:10]}, {str(dyn_latest)[:10]}]",
       )
   else:
       T.skip("GRACE time coverage", "tables unreadable")




# ── Summary diagnostics ─────────────────────────────────────────────────────


def print_summary_table(metas: list[dict]):
   """Print a row-count / file-size summary for all tables."""
   _hdr("I. Summary Diagnostics")


   header = (
       f"  {'Table':>25s}  {'Rows':>14s}  {'Columns':>8s}  "
       f"{'Files':>6s}  {'Size':>10s}"
   )
   print(f"\n{header}")
   print(f"  {'─' * 25}  {'─' * 14}  {'─' * 8}  {'─' * 6}  {'─' * 10}")


   for m in metas:
       name = m["name"]
       path = m.get("path", "")
       rows = m.get("row_count", 0)
       ncol = len(m.get("columns", []))
       nfiles = _count_parquet_files(path) if path else 0
       size = _human_size(_dir_size_bytes(path)) if path and os.path.exists(path) else "-"
       print(
           f"  {name:>25s}  {rows:>14,}  {ncol:>8}  "
           f"{nfiles:>6}  {size:>10s}"
       )
   print()




# ── Main ────────────────────────────────────────────────────────────────────


def main(base_dir: str = DIR_FLATTENED):
   global T
   T = CheckTracker()
   wall = _time.perf_counter()


   static_path = os.path.join(base_dir, "bedmap3_static.parquet")
   icesat_path = os.path.join(base_dir, "icesat2_dynamic.parquet")
   ocean_path  = os.path.join(base_dir, "ocean_dynamic.parquet")
   grace_path  = os.path.join(base_dir, "grace.parquet")


   print("=" * 72)
   print(" VALIDATE FLATTENED PARQUET TABLES")
   print(f" Base directory: {os.path.abspath(base_dir)}")
   print("=" * 72)


   # ── DuckDB session ───────────────────────────────────────────────
   con = duckdb.connect(database=":memory:")
   con.execute("SET memory_limit = '8GB'")
   con.execute("SET threads = 4")


   # ================================================================
   # A. EXISTENCE
   # ================================================================
   _hdr("A. File Existence & Readability")


   s_ok = validate_existence("static",  static_path)
   i_ok = validate_existence("icesat2", icesat_path)
   o_ok = validate_existence("ocean",   ocean_path)
   g_ok = validate_existence("grace",   grace_path)


   if not (s_ok or i_ok or o_ok or g_ok):
       print("\n[FATAL] No readable Parquet sources found. Aborting.")
       return 2


   # ================================================================
   # B. SCHEMA VALIDATION
   # ================================================================
   _hdr("B. Schema Validation")


   if s_ok:
       validate_schema("static",  static_path, SCHEMA_STATIC)
   if i_ok:
       validate_schema("icesat2", icesat_path, SCHEMA_ICESAT)
   if o_ok:
       validate_schema("ocean",   ocean_path,  SCHEMA_OCEAN)
   if g_ok:
       validate_schema("grace",   grace_path,  SCHEMA_GRACE)


   # ================================================================
   # C-G. PER-TABLE VALIDATION
   # ================================================================
   meta_static = {"name": "static", "readable": False, "row_count": 0,
                  "columns": [], "path": static_path}
   meta_icesat = {"name": "icesat2", "readable": False, "row_count": 0,
                  "columns": [], "path": icesat_path}
   meta_ocean  = {"name": "ocean", "readable": False, "row_count": 0,
                  "columns": [], "path": ocean_path}
   meta_grace  = {"name": "grace", "readable": False, "row_count": 0,
                  "columns": [], "path": grace_path}


   if s_ok:
       _hdr("C-G. Static Table: bedmap3_static.parquet")
       m = validate_table(
           con, "static", static_path, SCHEMA_STATIC,
           join_keys=["y", "x"],
           has_yx=True, has_time=False,
           phys_cols=[
               "surface", "bed", "thickness", "bed_slope",
               "dist_to_grounding_line", "clamped_depth",
               "dist_to_ocean", "ice_draft",
           ],
       )
       meta_static.update(m)


       # Static-specific: mask values should be only 1 or 3 (ice)
       if "mask" in meta_static["columns"]:
           distinct_mask = con.execute(
               "SELECT DISTINCT mask FROM static ORDER BY mask"
           ).fetchall()
           mask_vals = sorted([int(r[0]) for r in distinct_mask])
           T.ok(
               "static: mask ∈ {1, 3}",
               set(mask_vals) <= {1, 3},
               f"values: {mask_vals}",
           )


   if i_ok:
       _hdr("C-G. ICESat-2 Table: icesat2_dynamic.parquet")
       m = validate_table(
           con, "icesat2", icesat_path, SCHEMA_ICESAT,
           join_keys=["y", "x", "time"],
           has_yx=True, has_time=True,
           phys_cols=[
               "delta_h", "ice_area", "h_surface_dynamic", "surface_slope",
           ],
       )
       meta_icesat.update(m)


   if o_ok:
       _hdr("C-G. Ocean Table: ocean_dynamic.parquet")
       m = validate_table(
           con, "ocean", ocean_path, SCHEMA_OCEAN,
           join_keys=["y", "x", "time"],
           has_yx=True, has_time=True,
           phys_cols=["thetao", "so", "T_f", "T_star"],
       )
       meta_ocean.update(m)


       # Ocean-specific: T_f should be negative (sub-zero freezing)
       if "T_f" in meta_ocean["columns"] and meta_ocean["row_count"] > 0:
           pct_pos_tf = con.execute(
               'SELECT 100.0 * COUNT(*) FILTER (WHERE "T_f" > 0) '
               '/ COUNT(*) FROM ocean WHERE "T_f" IS NOT NULL'
           ).fetchone()[0]
           T.ok(
               "ocean: T_f predominantly ≤ 0 °C",
               pct_pos_tf < 5.0,
               f"{pct_pos_tf:.2f}% positive (expect <5%)",
           )


       # Ocean-specific: T_star (thermal driving) — flag if > 50% negative
       if "T_star" in meta_ocean["columns"] and meta_ocean["row_count"] > 0:
           pct_pos_ts = con.execute(
               'SELECT 100.0 * COUNT(*) FILTER (WHERE "T_star" > 0) '
               '/ COUNT(*) FROM ocean WHERE "T_star" IS NOT NULL'
           ).fetchone()[0]
           print(f"    [info]  T_star: {pct_pos_ts:.1f}% positive (melt potential)")


   if g_ok:
       _hdr("C-G. GRACE Table: grace.parquet")
       m = validate_table(
           con, "grace", grace_path, SCHEMA_GRACE,
           join_keys=["mascon_id", "time"],
           has_yx=False, has_time=True,
           phys_cols=["lwe_length"],
       )
       meta_grace.update(m)


       # GRACE-specific: mascon_id should have reasonable cardinality
       if meta_grace["row_count"] > 0:
           n_mascons = con.execute(
               "SELECT COUNT(DISTINCT CAST(mascon_id AS INTEGER)) FROM grace"
           ).fetchone()[0]
           print(f"    [info]  {n_mascons} distinct mascon_ids")


   # ================================================================
   # H. CROSS-TABLE COMPATIBILITY
   # ================================================================
   validate_cross_table(con, meta_static, meta_icesat, meta_ocean, meta_grace)


   # ================================================================
   # I. SUMMARY
   # ================================================================
   print_summary_table([meta_static, meta_icesat, meta_ocean, meta_grace])


   # ── Cleanup ──────────────────────────────────────────────────────
   con.close()


   # ── Verdict ──────────────────────────────────────────────────────
   elapsed = _time.perf_counter() - wall
   print("=" * 72)
   if T.passed:
       print(
           f" ✓  ALL {T.n_pass} CHECKS PASSED"
           + (f"  ({T.n_warn} warning(s))" if T.n_warn else "")
           + f"  [{elapsed:.1f} s]"
       )
   else:
       print(
           f" ✗  {T.n_fail} FAILED  /  {T.total} total"
           + (f"  ({T.n_warn} warning(s))" if T.n_warn else "")
           + f"  [{elapsed:.1f} s]"
       )
   if T.n_skip:
       print(f"    {T.n_skip} check(s) skipped due to missing prerequisites.")
   print("=" * 72)


   return 0 if T.passed else 1




# ── Entry point ─────────────────────────────────────────────────────────────


if __name__ == "__main__":
   parser = argparse.ArgumentParser(
       description=(
           "Validate the four flattened Parquet tables "
           "(bedmap3_static, icesat2_dynamic, ocean_dynamic, grace) "
           "before LSP assembly."
       ),
   )
   parser.add_argument(
       "--dir",
       default=DIR_FLATTENED,
       help=f"Base directory containing the Parquet files (default: {DIR_FLATTENED})",
   )
   args = parser.parse_args()
   sys.exit(main(base_dir=args.dir))
