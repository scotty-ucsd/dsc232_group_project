"""
extract_ml_subset.py
-------------------------------------------------------------------------------
COMPUTATIONAL GLACIOLOGY - ML SUBSET GENERATOR
-------------------------------------------------------------------------------
Extracts a spatially contiguous, full-time-series subset of the 39GB Feature 
Store for local ML model prototyping. Targets the highly dynamic Amundsen Sea 
Embayment (Thwaites / Pine Island).
"""


import os
import time as _time
import duckdb


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
DIR_FLATTENED = "data/flattened"
ML_FEATURE_DIR = os.path.join(DIR_FLATTENED,
                              "antarctica_sparse_features.parquet")
OUTPUT_SUBSET = os.path.join(DIR_FLATTENED, "ml_subset_amundsen_sea.parquet")


MEMORY_LIMIT = "8GB"
THREADS = 8


# EPSG:3031 Bounding Box for the Amundsen Sea Embayment (Approximate)
# Captures Thwaites, Pine Island, Crosson, and Dotson ice shelves & catchments
ASE_BOUNDS = {
    "x_min": -1800000.0,
    "x_max": -1100000.0,
    "y_min": -800000.0,
    "y_max": -100000.0
}


def create_local_ml_subset():
    wall_t = _time.perf_counter()
    
    if not os.path.exists(ML_FEATURE_DIR):
        print(f"[FATAL] Source directory not found: {ML_FEATURE_DIR}")
        return


    con = duckdb.connect(database=":memory:")
    con.execute(f"SET memory_limit = '{MEMORY_LIMIT}'")
    con.execute(f"SET threads = {THREADS}")


    print("=" * 72)
    print(" EXTRACTING LOCAL ML SUBSET: AMUNDSEN SEA EMBAYMENT")
    print("=" * 72)


    print("\n[1/2] Scanning Hive Partitions & Applying Spatial Bounds...")
    t0 = _time.perf_counter()
    
    # We use predicate pushdown to let DuckDB filter the files efficiently
    query = f"""
        COPY (
            SELECT * FROM read_parquet('{ML_FEATURE_DIR}/**/*.parquet', hive_partitioning=true)
            WHERE x >= {ASE_BOUNDS['x_min']} AND x <= {ASE_BOUNDS['x_max']}
              AND y >= {ASE_BOUNDS['y_min']} AND y <= {ASE_BOUNDS['y_max']}
            ORDER BY y, x, month_idx
        ) TO '{OUTPUT_SUBSET}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """
    
    try:
        con.execute(query)
    except Exception as e:
        print(f"\n[FATAL] Extraction failed: {e}")
        con.close()
        return


    elapsed = _time.perf_counter() - t0
    
    # ---------------------------------------------------------
    # VALIDATE SUBSET
    # ---------------------------------------------------------
    print(f"  -> Extraction complete in {elapsed:.1f} seconds.")
    print("\n[2/2] Validating Subset Properties...")
    
    subset_info = con.execute(f"""
        SELECT 
            COUNT(*),
            COUNT(DISTINCT month_idx),
            COUNT(DISTINCT mascon_id)
        FROM read_parquet('{OUTPUT_SUBSET}')
    """).fetchone()
    
    file_size_mb = os.path.getsize(OUTPUT_SUBSET) / (1024 * 1024)
    
    print(f"  -> Rows Extracted:     {subset_info[0]:,}")
    print(f"  -> Temporal Depth:     {subset_info[1]} distinct months (Full Time Series)")
    print(f"  -> Spatial Breadth:    {subset_info[2]} unique GRACE Mascons")
    print(f"  -> File Size on Disk:  {file_size_mb:.1f} MB")


    con.close()
    
    print("\n" + "=" * 72)
    print(f" ML SUBSET READY [{_time.perf_counter() - wall_t:.1f} s]")
    print(f" Path: {OUTPUT_SUBSET}")
    print("=" * 72)


if __name__ == "__main__":
    create_local_ml_subset()

