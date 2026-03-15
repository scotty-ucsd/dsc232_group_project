# =====================================================================
# data_exploration.py
#
# Antarctic Basal Melt Prediction -- Step 01: EDA Pipeline
#
# Converted from sdsc_eda.ipynb.
# Runs on SDSC Expanse via:  sbatch run_data_exploration.sh
#
# All Step 01 requirements satisfied:
#   1. SparkSession config with formula justification
#   2. df.count(), df.show(), .summary(), .dtypes, groupBy().agg(),
#      df.select().distinct().count()
#   3. Categorical vs continuous variable classification
#   4. Target column (delta_h) description
#   5. Missing value ratios + duplicate key check
#   6. Plots: histograms, bar charts, correlation heatmap, range chart,
#             completeness heatmap, null structure chart
#   7. Preprocessing plan (documented, not executed)
#   8. All plots use plot_config.py Deep Field style
# =====================================================================

from __future__ import annotations

import os
import sys
import math
import time as _time
from collections import OrderedDict

import matplotlib
os.environ['MPLCONFIGDIR'] = os.path.join(os.getcwd(), '.matplotlib_cache')
os.makedirs(os.environ['MPLCONFIGDIR'], exist_ok=True)
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from pyspark.sql import SparkSession
from pyspark.sql.types import NumericType
from pyspark.sql import functions as F

from plot_config import apply_style, STYLE, new_fig, save_fig, style_colorbar
apply_style()


# =====================================================================
# CONFIGURATION
# =====================================================================

# -- HPC Resource Allocation ------------------------------------------
# Formula (Step 01 requirements):
#   executor_instances = total_cores - 1   (practical: reserve for driver)
#   executor_memory    = (total_memory - driver_memory) / executor_instances
#
# SDSC Expanse shared partition: 32 cores / 128 GB
#   executor_cores     = 5   (Spark best-practice for HDFS throughput)
#   executor_instances = (32 - 1) // 5 = 6
#   driver_memory      = 10 GB
#   executor_memory    = floor((128 - 10) / 6) = 19 GB
TOTAL_CORES        = 32
TOTAL_MEMORY_GB    = 128
EXECUTOR_CORES     = 5
EXECUTOR_INSTANCES = max((TOTAL_CORES - 1) // EXECUTOR_CORES, 1)   # 6
DRIVER_MEMORY_GB   = 10
EXECUTOR_MEMORY_GB = max(
    math.floor((TOTAL_MEMORY_GB - DRIVER_MEMORY_GB) / EXECUTOR_INSTANCES), 1
)
SHUFFLE_PARTITIONS = 2 * TOTAL_CORES  # 64

# -- Data paths -------------------------------------------------------
DATA_DIRS  = [
    os.path.join("data", "indiv_data"),
    os.path.join("data", "fused_data"),
]
OUTPUT_DIR = os.path.join("data", "eda_plots")
SAMPLE_DIR = os.path.join("data", "sample_data")

# -- Plot / histogram config ------------------------------------------
HISTOGRAM_BINS     = 50
FIG_DPI            = 150
TARGET_SAMPLE_ROWS = 250_000
OVERSAMPLE_FACTOR  = 1.2

# -- Resume config ----------------------------------------------------
RESUME_DATASET_IDX = 0
RESUME_PHASE       = 1

# -- Physical columns per dataset -------------------------------------
PHYS_COLUMNS = {
    "bedmap3_static.parquet": [
        "surface", "bed", "thickness", "bed_slope",
        "dist_to_grounding_line", "clamped_depth", "ice_draft",
    ],
    "grace.parquet": ["lwe_length"],
    "icesat2_dynamic.parquet": [
        "delta_h", "ice_area", "h_surface_dynamic", "surface_slope",
    ],
    "ocean_dynamic.parquet": ["thetao", "so", "T_f", "T_star"],
    "antarctica_sparse_features.parquet": [
        "surface", "bed", "thickness", "lwe_length",
        "delta_h", "thetao", "so", "T_f", "T_star",
    ],
}

DATASET_LABELS = {
    "bedmap3_static.parquet":             "Bedmap3 Static",
    "grace.parquet":                      "GRACE",
    "icesat2_dynamic.parquet":            "ICESat-2 Dynamic",
    "ocean_dynamic.parquet":              "Ocean Dynamic",
    "antarctica_sparse_features.parquet": "Fused (Sparse)",
}

DATASET_COLOURS = {
    "bedmap3_static.parquet":             STYLE["PURPLE"],
    "grace.parquet":                      STYLE["BLUE"],
    "icesat2_dynamic.parquet":            STYLE["GREEN"],
    "ocean_dynamic.parquet":              STYLE["AMBER"],
    "antarctica_sparse_features.parquet": STYLE["RED"],
}


# =====================================================================
# SPARK SESSION
# =====================================================================

def build_spark_session() -> SparkSession:
    """
    HPC-optimised SparkSession for 32 cores / 128 GB.

    Resource allocation (formula from Step 01 requirements):
      driver.memory      = 10g
      executor.instances = (32-1) // 5 = 6
      executor.cores     = 5
      executor.memory    = floor((128-10) / 6) = 19g

    AQE enabled for runtime coalesce of post-shuffle partitions.
    shuffle.partitions = 2 x 32 = 64.
    """
    if EXECUTOR_MEMORY_GB < 1:
        raise RuntimeError(
            f"Derived executor memory is {EXECUTOR_MEMORY_GB}g -- "
            f"allocation too small for {EXECUTOR_INSTANCES} executors."
        )

    driver_mem = f"{DRIVER_MEMORY_GB}g"
    exec_mem   = f"{EXECUTOR_MEMORY_GB}g"

    print("=" * 72)
    print("  HPC SparkSession Configuration")
    print("=" * 72)
    print(f"  TOTAL_CORES ............. {TOTAL_CORES}")
    print(f"  TOTAL_MEMORY_GB ......... {TOTAL_MEMORY_GB}")
    print(f"  DRIVER_MEMORY ........... {driver_mem}")
    print(f"  EXECUTOR_CORES .......... {EXECUTOR_CORES}")
    print(f"  EXECUTOR_INSTANCES ...... {EXECUTOR_INSTANCES}")
    print(f"  EXECUTOR_MEMORY ......... {exec_mem}")
    print(f"  SHUFFLE_PARTITIONS ...... {SHUFFLE_PARTITIONS}")
    print("=" * 72)

    spark = (
        SparkSession.builder
        .appName("HPC_Antarctic_EDA_Pipeline")
        .config("spark.driver.memory",           driver_mem)
        .config("spark.executor.instances",       str(EXECUTOR_INSTANCES))
        .config("spark.executor.cores",           str(EXECUTOR_CORES))
        .config("spark.executor.memory",          exec_mem)
        .config("spark.sql.shuffle.partitions",   str(SHUFFLE_PARTITIONS))
        .config("spark.driver.maxResultSize",     "4g")
        .config("spark.network.timeout",          "1200s")
        .config("spark.sql.sources.parallelPartitionDiscovery.threshold",   "32")
        .config("spark.sql.sources.parallelPartitionDiscovery.parallelism", "64")
        .config("spark.sql.adaptive.enabled",                     "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled",  "true")
        .config("spark.sql.adaptive.advisoryPartitionSizeInBytes","128m")
        .config("spark.sql.parquet.filterPushdown",               "true")
        .config("spark.sql.parquet.mergeSchema",                  "false")
        .config("spark.local.dir",
                os.environ.get("SPARK_LOCAL_DIRS",
                               os.path.join(os.getcwd(), "spark_scratch")))
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    scratch = spark.conf.get("spark.local.dir")
    os.makedirs(scratch, exist_ok=True)
    return spark


# =====================================================================
# DATASET DISCOVERY + LOADING
# =====================================================================

def discover_parquet_datasets(data_dirs: list) -> list:
    datasets = []
    for base_dir in data_dirs:
        abs_dir = os.path.abspath(base_dir)
        if not os.path.isdir(abs_dir):
            print(f"  [WARN] Directory not found, skipping: {abs_dir}")
            continue
        for entry in sorted(os.listdir(abs_dir)):
            if entry.endswith(".parquet"):
                datasets.append(os.path.join(abs_dir, entry))
    if not datasets:
        print("[WARNING] No .parquet entries found.")
    return datasets


def load_dataset(spark: SparkSession, path: str):
    """
    Read Parquet with recursiveFileLookup to handle sub-directory
    layouts (e.g. icesat2_dynamic.parquet/step_015.parquet/).
    mergeSchema unifies columns across sub-files.
    """
    return (
        spark.read
        .option("recursiveFileLookup", "true")
        .option("mergeSchema", "true")
        .parquet(path)
    )


# =====================================================================
# DISTRIBUTED METRIC COMPUTATION
# =====================================================================

def compute_histogram_spark(df, col_name: str, n_bins: int = HISTOGRAM_BINS):
    """
    Executor-side histogram via RDD.histogram().
    Only O(n_bins) floats returned to driver -- no raw rows collected.
    """
    try:
        rdd = df.select(col_name).na.drop().rdd.map(lambda r: float(r[0]))
        if rdd.isEmpty():
            return None, None
        return rdd.histogram(n_bins)
    except Exception as exc:
        print(f"      [WARN] Histogram failed for '{col_name}': {exc}")
        return None, None


def compute_correlation_matrix_spark(df, cols: list) -> np.ndarray:
    """
    Pairwise Pearson correlations via df.stat.corr() -- executor-side.
    Each call is one single-pass distributed aggregate.
    """
    n   = len(cols)
    mat = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            try:
                r = df.stat.corr(cols[i], cols[j])
                r = 0.0 if r is None or math.isnan(r) else r
            except Exception:
                r = 0.0
            mat[i, j] = mat[j, i] = r
    return mat


def compute_range_stats_spark(df, cols: list) -> dict:
    """
    Single .agg() call for min/max/mean/std across all columns.
    One Spark job, no shuffle.
    """
    agg_exprs = []
    for c in cols:
        agg_exprs += [
            F.min(c).alias(f"{c}__min"),
            F.max(c).alias(f"{c}__max"),
            F.mean(c).alias(f"{c}__mean"),
            F.stddev(c).alias(f"{c}__std"),
        ]
    row = df.agg(*agg_exprs).head()
    return {
        c: {
            "min":  float(row[f"{c}__min"])  if row[f"{c}__min"]  is not None else float("nan"),
            "max":  float(row[f"{c}__max"])  if row[f"{c}__max"]  is not None else float("nan"),
            "mean": float(row[f"{c}__mean"]) if row[f"{c}__mean"] is not None else float("nan"),
            "std":  float(row[f"{c}__std"])  if row[f"{c}__std"]  is not None else float("nan"),
        }
        for c in cols
    }


def collect_dataset_metadata(df, dataset_name: str) -> dict:
    """
    Two-action metadata collection.

    Action 1: df.count()
    Action 2: per-column non-null counts via single .agg()

    Also demonstrates:
      - df.groupBy("mask").agg() for categorical distribution
      - df.select("x","y").distinct().count() for unique spatial pixels
    """
    columns = df.columns

    print(f"      [Step 1/2] Global row count ...")
    row_count = df.count()
    print(f"      -> {row_count:,} rows")

    print(f"      [Step 2/2] Column completeness ...")
    agg_exprs = [
        F.count(F.when(F.col(c).isNotNull(), c)).alias(c)
        for c in columns
    ]
    completeness = {}
    null_counts  = {}
    try:
        non_null_row = df.agg(*agg_exprs).head()
        for c in columns:
            nn = int(non_null_row[c]) if non_null_row[c] is not None else 0
            completeness[c] = (nn / row_count * 100) if row_count > 0 else 0.0
            null_counts[c]  = row_count - nn
    except Exception as exc:
        print(f"      [WARNING] Completeness check failed: {exc}")
        for c in columns:
            completeness[c] = 0.0
            null_counts[c]  = 0

    # groupBy().agg() -- categorical distribution
    if "mask" in columns:
        print(f"      Mask category distribution (groupBy/agg):")
        grp_cols = [F.count("*").alias("count")]
        if "delta_h" in columns:
            grp_cols.append(F.avg("delta_h").alias("avg_delta_h"))
        df.groupBy("mask").agg(*grp_cols).orderBy("mask").show()

    # distinct().count() -- unique spatial pixels
    if "x" in columns and "y" in columns:
        n_pixels = df.select("x", "y").distinct().count()
        print(f"      Distinct spatial pixels (x, y): {n_pixels:,}")

    # distinct().count() -- unique time steps
    if "month_idx" in columns:
        n_times = df.select("month_idx").distinct().count()
        print(f"      Distinct time steps (month_idx): {n_times:,}")

    return {
        "row_count":    row_count,
        "col_count":    len(columns),
        "columns":      columns,
        "completeness": completeness,
        "null_counts":  null_counts,
    }


# =====================================================================
# PER-DATASET PLOTS
# =====================================================================

def fig_histograms_for_dataset(df, dataset_name: str,
                                phys_cols: list, colour: str):
    """
    Multi-panel histogram figure.
    Histograms computed via Spark RDD.histogram() on executors.
    Only bin arrays returned to driver.
    """
    available    = set(df.columns)
    cols_to_plot = [c for c in phys_cols if c in available]
    if not cols_to_plot:
        print(f"      [SKIP] No plottable columns for {dataset_name}")
        return

    n          = len(cols_to_plot)
    ncols_grid = min(n, 3)
    nrows_grid = math.ceil(n / ncols_grid)

    fig, axes = new_fig(
        nrows=nrows_grid, ncols=ncols_grid,
        figsize=(5 * ncols_grid, 4 * nrows_grid),
    )
    if n == 1:
        axes = np.array([axes])
    axes_flat = np.array(axes).flatten()

    label = DATASET_LABELS.get(dataset_name, dataset_name)
    fig.suptitle(f"Distributions : {label}",
                 color=STYLE["TEXT"], fontsize=14, fontweight="bold", y=1.02)

    for idx, col_name in enumerate(cols_to_plot):
        ax = axes_flat[idx]
        print(f"      Computing histogram: {col_name} ...")
        edges, counts = compute_histogram_spark(df, col_name, HISTOGRAM_BINS)

        if edges is None:
            ax.text(0.5, 0.5, "No Data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12, color=STYLE["GRID"])
            ax.set_title(col_name, color=STYLE["TEXT"])
            continue

        centres = [(edges[i] + edges[i+1]) / 2 for i in range(len(counts))]
        widths  = [edges[i+1] - edges[i]       for i in range(len(counts))]

        ax.bar(centres, counts, width=widths, color=colour,
               edgecolor=STYLE["BG"], alpha=0.85, linewidth=0.3)
        ax.set_title(col_name, color=STYLE["TEXT"], fontsize=11)
        ax.set_ylabel("Count", color=STYLE["TEXT"], fontsize=9)
        ax.yaxis.set_major_formatter(mticker.EngFormatter())
        ax.tick_params(colors=STYLE["TEXT"])

    for idx in range(len(cols_to_plot), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.tight_layout()
    safe_name = dataset_name.replace(".parquet", "")
    save_fig(fig, OUTPUT_DIR, f"fig_03_histograms_{safe_name}.png")


def fig_correlation_for_dataset(df, dataset_name: str, phys_cols: list):
    """
    Correlation heatmap using df.stat.corr() on executors.
    """
    available   = set(df.columns)
    cols_to_use = [c for c in phys_cols if c in available]
    if len(cols_to_use) < 2:
        print(f"      [SKIP] Need >= 2 numeric columns for correlation")
        return

    print(f"      Computing {len(cols_to_use)}x{len(cols_to_use)} "
          f"correlation matrix ...")
    corr = compute_correlation_matrix_spark(df, cols_to_use)

    fig, ax = new_fig(
        figsize=(max(7, len(cols_to_use) * 0.9),
                 max(6, len(cols_to_use) * 0.8))
    )
    label = DATASET_LABELS.get(dataset_name, dataset_name)
    ax.set_title(f"Correlation Matrix : {label}",
                 color=STYLE["TEXT"], fontsize=13, fontweight="bold")

    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1, aspect="equal")
    ax.set_xticks(range(len(cols_to_use)))
    ax.set_xticklabels(cols_to_use, rotation=45, ha="right",
                       fontsize=9, color=STYLE["TEXT"])
    ax.set_yticks(range(len(cols_to_use)))
    ax.set_yticklabels(cols_to_use, fontsize=9, color=STYLE["TEXT"])

    for i in range(len(cols_to_use)):
        for j in range(len(cols_to_use)):
            val = corr[i, j]
            tc  = STYLE["TEXT"] if abs(val) < 0.6 else "#ffffff"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color=tc)

    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    style_colorbar(cb, "Pearson r")
    ax.grid(False)
    fig.tight_layout()
    safe_name = dataset_name.replace(".parquet", "")
    save_fig(fig, OUTPUT_DIR, f"fig_04_correlation_{safe_name}.png")


# =====================================================================
# CROSS-DATASET PLOTS
# =====================================================================

def fig_dataset_overview(dataset_meta: dict):
    """
    Two-panel bar chart: row counts (log) and column counts per dataset.
    Requirement: bar chart type.
    """
    names  = list(dataset_meta.keys())
    labels = [DATASET_LABELS.get(n, n) for n in names]
    rows   = [dataset_meta[n]["row_count"] for n in names]
    cols   = [dataset_meta[n]["col_count"]  for n in names]
    colors = [DATASET_COLOURS.get(n, STYLE["BLUE"]) for n in names]

    fig, (ax1, ax2) = new_fig(nrows=1, ncols=2, figsize=(14, 5))
    fig.suptitle("Dataset Overview",
                 color=STYLE["TEXT"], fontsize=16, fontweight="bold", y=1.02)

    bars1 = ax1.barh(labels, rows, color=colors,
                     edgecolor=STYLE["BG"], alpha=0.85)
    ax1.set_xscale("log")
    ax1.set_xlabel("Row Count (log scale)", color=STYLE["TEXT"], fontsize=9)
    ax1.set_title("Rows per Dataset", color=STYLE["TEXT"])
    ax1.tick_params(colors=STYLE["TEXT"])
    ax1.xaxis.grid(True, color=STYLE["GRID"], lw=0.5, alpha=0.6)
    ax1.yaxis.grid(False)
    for bar, val in zip(bars1, rows):
        ax1.text(bar.get_width() * 1.1, bar.get_y() + bar.get_height() / 2,
                 f"{val:,.0f}", va="center", fontsize=9, color=STYLE["TEXT"])

    bars2 = ax2.barh(labels, cols, color=colors,
                     edgecolor=STYLE["BG"], alpha=0.85)
    ax2.set_xlabel("Column Count", color=STYLE["TEXT"], fontsize=9)
    ax2.set_title("Columns per Dataset", color=STYLE["TEXT"])
    ax2.tick_params(colors=STYLE["TEXT"])
    ax2.xaxis.grid(True, color=STYLE["GRID"], lw=0.5, alpha=0.6)
    ax2.yaxis.grid(False)
    for bar, val in zip(bars2, cols):
        ax2.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                 str(val), va="center", fontsize=9, color=STYLE["TEXT"])

    fig.tight_layout()
    save_fig(fig, OUTPUT_DIR, "fig_01_dataset_overview.png")


def fig_data_completeness(dataset_meta: dict):
    """Heatmap of percent non-null per column per dataset."""
    all_cols = []
    seen = set()
    for name in dataset_meta:
        for c in dataset_meta[name]["columns"]:
            if c not in seen:
                all_cols.append(c); seen.add(c)

    names  = list(dataset_meta.keys())
    labels = [DATASET_LABELS.get(n, n) for n in names]
    matrix = np.full((len(names), len(all_cols)), np.nan)
    for i, name in enumerate(names):
        comp = dataset_meta[name].get("completeness", {})
        for j, c in enumerate(all_cols):
            if c in comp:
                matrix[i, j] = comp[c]

    fig, ax = new_fig(figsize=(max(14, len(all_cols) * 0.9), 5))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=100)
    ax.set_xticks(range(len(all_cols)))
    ax.set_xticklabels(all_cols, rotation=55, ha="right",
                       fontsize=8, color=STYLE["TEXT"])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10, color=STYLE["TEXT"])
    ax.set_title("Data Completeness (Percent Non-Null)", color=STYLE["TEXT"])
    ax.grid(False)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if np.isnan(val):
                ax.text(j, i, "N/A", ha="center", va="center",
                        fontsize=7, color=STYLE["GRID"])
            else:
                tc = STYLE["TEXT"] if val > 50 else "#ffffff"
                ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                        fontsize=7, fontweight="bold", color=tc)

    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    style_colorbar(cb, "% Non-Null")
    fig.tight_layout()
    save_fig(fig, OUTPUT_DIR, "fig_02_data_completeness.png")


def fig_physical_ranges(all_range_stats: dict):
    """Horizontal range bars: [min, max] with mean +/- std per variable."""
    entries = []
    for ds_name, stats in all_range_stats.items():
        colour = DATASET_COLOURS.get(ds_name, STYLE["BLUE"])
        label  = DATASET_LABELS.get(ds_name, ds_name)
        for col_name, vals in stats.items():
            entries.append({
                "label":  f"{col_name}\n({label})",
                "min":    vals["min"],  "max":  vals["max"],
                "mean":   vals["mean"], "std":  vals["std"],
                "colour": colour,
            })

    if not entries:
        print("    [SKIP] No range stats to plot.")
        return

    n   = len(entries)
    fig, ax = new_fig(figsize=(14, max(6, n * 0.45)))
    ax.set_title("Physical Variable Ranges (Min / Max / Mean +/- Std)",
                 color=STYLE["TEXT"], fontsize=14, fontweight="bold")

    for i, e in enumerate(entries):
        ax.barh(i, e["max"] - e["min"], left=e["min"], height=0.6,
                color=e["colour"], alpha=0.3, edgecolor=e["colour"])
        ax.plot(e["mean"], i, "D", color=e["colour"],
                markersize=6, markeredgecolor="#ffffff", markeredgewidth=0.5)
        if not np.isnan(e["std"]):
            ax.plot([e["mean"] - e["std"], e["mean"] + e["std"]], [i, i],
                    color=e["colour"], linewidth=2, solid_capstyle="round")

    ax.set_yticks(range(n))
    ax.set_yticklabels([e["label"] for e in entries],
                       fontsize=8, color=STYLE["TEXT"])
    ax.set_xlabel("Value", color=STYLE["TEXT"], fontsize=9)
    ax.tick_params(colors=STYLE["TEXT"])
    ax.xaxis.grid(True, color=STYLE["GRID"], lw=0.5, alpha=0.5)
    ax.yaxis.grid(False)
    ax.invert_yaxis()
    fig.tight_layout()
    save_fig(fig, OUTPUT_DIR, "fig_05_physical_ranges.png")


def fig_null_structure(dataset_meta: dict):
    """Grouped horizontal bar chart of null counts per column."""
    all_cols = []
    seen = set()
    for name in dataset_meta:
        for c in dataset_meta[name]["columns"]:
            if c not in seen:
                all_cols.append(c); seen.add(c)

    names = list(dataset_meta.keys())
    fig, ax = new_fig(figsize=(14, max(5, len(all_cols) * 0.35)))
    ax.set_title("Null Counts per Column",
                 color=STYLE["TEXT"], fontsize=14, fontweight="bold")

    y_pos      = np.arange(len(all_cols))
    bar_height = 0.7 / max(len(names), 1)

    for ds_idx, ds_name in enumerate(names):
        nulls  = dataset_meta[ds_name].get("null_counts", {})
        label  = DATASET_LABELS.get(ds_name, ds_name)
        colour = DATASET_COLOURS.get(ds_name, STYLE["BLUE"])
        vals   = [nulls.get(c, 0) for c in all_cols]

        ax.barh(y_pos + ds_idx * bar_height, vals, height=bar_height,
                color=colour, alpha=0.8, label=label,
                edgecolor=STYLE["BG"], linewidth=0.3)

    ax.set_yticks(y_pos + bar_height * (len(names) - 1) / 2)
    ax.set_yticklabels(all_cols, fontsize=8, color=STYLE["TEXT"])
    ax.set_xlabel("Null Count", color=STYLE["TEXT"], fontsize=9)
    ax.tick_params(colors=STYLE["TEXT"])
    ax.xaxis.set_major_formatter(mticker.EngFormatter())
    ax.xaxis.grid(True, color=STYLE["GRID"], lw=0.5, alpha=0.5)
    ax.yaxis.grid(False)
    ax.legend(fontsize=8, facecolor=STYLE["BG"], edgecolor=STYLE["GRID"],
              labelcolor=STYLE["TEXT"])
    ax.invert_yaxis()
    fig.tight_layout()
    save_fig(fig, OUTPUT_DIR, "fig_06_null_structure.png")


# =====================================================================
# SAMPLING
# =====================================================================

def generate_sample(df, dataset_name: str, row_count: int):
    """
    Fractional sample targeting TARGET_SAMPLE_ROWS rows.
    1.2x oversample factor compensates for Bernoulli variance in df.sample().
    coalesce(1) produces a single output file -- safe at this scale.
    """
    if row_count == 0:
        print(f"      [SKIP] Empty dataset.")
        return

    raw_frac = TARGET_SAMPLE_ROWS / row_count
    if raw_frac >= 1.0:
        print(f"      Source ({row_count:,}) <= target. Writing full copy.")
        sample_df = df
    else:
        frac = min(raw_frac * OVERSAMPLE_FACTOR, 1.0)
        print(f"      Sampling fraction: {frac:.6f}")
        sample_df = df.sample(withReplacement=False, fraction=frac, seed=42)

    safe_name   = dataset_name.replace(".parquet", "")
    output_path = os.path.join(SAMPLE_DIR, f"{safe_name}_sample.parquet")
    sample_df.coalesce(1).write.mode("overwrite").parquet(output_path)
    print(f"      -> Sample written: {output_path}")


# =====================================================================
# PREPROCESSING PLAN
# Requirement: describe planned preprocessing -- explain only, no execution
# =====================================================================

def print_preprocessing_plan():
    print(f"\n{'=' * 72}")
    print(f"  PREPROCESSING PLAN")
    print(f"  (Planned -- actual execution in feature_engineering_pipeline.py)")
    print(f"{'=' * 72}")
    print("""
  1. MISSING VALUES
     - ICESat-2 (delta_h, ~8%): impute with pixel expanding-window mean
       via Spark Window.partitionBy("x","y").orderBy("month_idx")
       .rowsBetween(Window.unboundedPreceding, 0).
     - Ocean columns (thetao, so, ~70%): leave null for grounded ice.
       Physically undefined outside floating shelves (mask != 3).
     - All remaining nulls: pyspark.ml.feature.Imputer, strategy="median",
       fit on training split only to prevent leakage into val/test.

  2. CLASS IMBALANCE
     - Global positive rate: ~3% (basal_loss_agreement = 1).
     - Region-stratified undersampling via pipeline_config.undersample():
         amundsen_sea:      1:5   (smallest positive pool)
         totten_and_aurora: 1:5
         antarctic_peninsula: 1:8
         lambert_amery, ross, ronne: 1:12
     - Additional: REGION_WEIGHT_MAP sample weights via weightCol.
       XGBoost scale_pos_weight = N_neg/N_pos from training fold.

  3. FEATURE TRANSFORMATIONS
     Scaling:
       - MinMaxScaler for XGBoost pipeline (bounded [0,1]).
       - StandardScaler (withMean=False) for SVD pipeline.
     Encoding:
       - StringIndexer + OneHotEncoder for region_id.
       - Bucketizer for dist_to_grounding_line proximity zones.
     Feature engineering via Spark SQL functions:
       - thermal_driving = thetao_mo - t_f_mo
       - grounding_line_vulnerability = thickness / dist_to_grounding_line
       - retrograde_flag = (bed_slope < 0).cast(int)
       - sin_month, cos_month = cyclical month encoding
       - 6-month rolling averages: Window.rowsBetween(-5, 0)
       - Regional residuals: ocean features minus per-region train mean
     Dimensionality reduction (Model 2):
       - RowMatrix.computeSVD(k=15, computeU=False)
       - pyspark.ml.feature.PCA(k=15) for efficient .transform()
       - KMeans(k=8) on SVD components

  4. SPARK OPERATIONS USED
     - Imputer, StandardScaler, MinMaxScaler, VectorAssembler,
       StringIndexer, OneHotEncoder, Bucketizer, PCA
     - Window.partitionBy(x,y).orderBy(month_idx) for lag/rolling features
     - df.groupBy().agg() for regional mean residualization
     - df.write.parquet() checkpoints to break DAG lineage (OOM prevention)
""")
    print(f"{'=' * 72}")


# =====================================================================
# PERFORMANCE REPORT
# =====================================================================

def print_performance_report(dataset_times: dict, total_wall: float):
    print(f"\n{'=' * 72}")
    print(f"  PERFORMANCE REPORT")
    print(f"{'=' * 72}")
    print(f"\n  {'Dataset':<40s}  {'Time (s)':>10s}")
    print(f"  {'-' * 40}  {'-' * 10}")
    for name, t in dataset_times.items():
        print(f"  {name:<40s}  {t:>10.2f}")
    print(f"  {'-' * 40}  {'-' * 10}")
    print(f"  {'TOTAL':<40s}  {total_wall:>10.2f}")
    print(f"""
  Executor configuration:
    Executor instances = {EXECUTOR_INSTANCES}
    Executor cores     = {EXECUTOR_CORES}
    Executor memory    = {EXECUTOR_MEMORY_GB}g
    Driver memory      = {DRIVER_MEMORY_GB}g
    Tn (this run)      = {total_wall:.2f}s

  To compute Speedup and Efficiency:
    1. Run with TOTAL_CORES=6, EXECUTOR_CORES=5 (1 executor). Record T1.
    2. Speedup    = T1 / Tn
    3. Efficiency = Speedup / n   where n = EXECUTOR_INSTANCES
""")


# =====================================================================
# PER-DATASET PIPELINE
# =====================================================================

def run_eda_for_dataset(
    spark: SparkSession,
    dataset_path: str,
    dataset_meta_accumulator: OrderedDict,
    all_range_stats_accumulator: OrderedDict,
    start_phase: int = 1,
) -> float:
    """
    8-phase EDA pipeline for one Parquet dataset.

    Phase 1: Ingest + schema (df.dtypes)
    Phase 2: Row count (df.count()) + column count
    Phase 3: Numeric summary statistics (.summary())
    Phase 4: Metadata -- completeness, nulls, groupBy/agg, distinct/count
    Phase 5: Distributed histograms (RDD.histogram() on executors)
    Phase 6: Distributed correlation heatmap (stat.corr() on executors)
    Phase 7: Range statistics (single .agg())
    Phase 8: Representative sample generation

    No raw data is ever collected to the driver.
    """
    dataset_name    = os.path.basename(dataset_path)
    t_dataset_start = _time.perf_counter()

    print(f"\n{'=' * 72}")
    print(f"  DATASET : {dataset_name}")
    if start_phase > 1:
        print(f"  RESUMING from Phase {start_phase}")
    print(f"  PATH    : {dataset_path}")
    print(f"{'=' * 72}")

    # Phase 1: Ingest + Schema
    print(f"\n  [Phase 1] Ingesting + reading schema ...")
    t0 = _time.perf_counter()
    try:
        df = load_dataset(spark, dataset_path)
    except Exception as exc:
        print(f"  [ERROR] Failed to read {dataset_name}: {exc}")
        return _time.perf_counter() - t_dataset_start

    num_cols = len(df.columns)
    print(f"  Schema ({num_cols} columns):")
    print(f"  {'Column Name':<30s}  {'Data Type':<20s}")
    print(f"  {'-' * 52}")
    for col_name, col_type in df.dtypes:
        print(f"  {col_name:<30s}  {col_type:<20s}")
    print(f"  [{_time.perf_counter() - t0:.1f}s]")

    # Phase 2: Row count
    print(f"\n  [Phase 2] Counting rows ...")
    t0 = _time.perf_counter()
    try:
        row_count = df.count()
    except Exception as exc:
        print(f"  [ERROR] Row count failed: {exc}")
        return _time.perf_counter() - t_dataset_start
    print(f"  Total rows    : {row_count:>14,}")
    print(f"  Total columns : {num_cols:>14,}")
    print(f"  [{_time.perf_counter() - t0:.1f}s]")

    if row_count == 0:
        print(f"  [WARNING] Dataset is empty -- skipping analysis.")
        return _time.perf_counter() - t_dataset_start

    # Phase 3: Summary statistics
    if start_phase <= 3:
        print(f"\n  [Phase 3] Summary statistics (.summary()) ...")
        t0 = _time.perf_counter()
        numeric_cols = [
            f.name for f in df.schema.fields
            if isinstance(f.dataType, NumericType)
        ]
        if not numeric_cols:
            print(f"  [WARNING] No numeric columns -- skipping summary.")
        else:
            print(f"  Numeric columns ({len(numeric_cols)}): {numeric_cols}")
            df.select(numeric_cols).summary(
                "count", "min", "max", "mean", "stddev"
            ).show(truncate=False, vertical=True)
        print(f"  [{_time.perf_counter() - t0:.1f}s]")
    else:
        print(f"\n  [Phase 3] SKIPPED (resume)")

    # Phase 4: Metadata + groupBy/agg + distinct/count
    if start_phase <= 4:
        print(f"\n  [Phase 4] Collecting metadata ...")
        t0 = _time.perf_counter()
        meta = collect_dataset_metadata(df, dataset_name)
        dataset_meta_accumulator[dataset_name] = meta
        print(f"  [{_time.perf_counter() - t0:.1f}s]")
    else:
        print(f"\n  [Phase 4] SKIPPED (resume)")

    phys_cols = PHYS_COLUMNS.get(dataset_name, [])
    colour    = DATASET_COLOURS.get(dataset_name, STYLE["BLUE"])

    # Phase 5: Histograms
    if start_phase <= 5:
        if phys_cols:
            print(f"\n  [Phase 5] Building histograms ...")
            t0 = _time.perf_counter()
            fig_histograms_for_dataset(df, dataset_name, phys_cols, colour)
            print(f"  [{_time.perf_counter() - t0:.1f}s]")
        else:
            print(f"\n  [Phase 5] No PHYS_COLUMNS configured -- skipping.")
    else:
        print(f"\n  [Phase 5] SKIPPED (resume)")

    # Phase 6: Correlation heatmap
    if start_phase <= 6:
        if phys_cols and len(phys_cols) >= 2:
            print(f"\n  [Phase 6] Building correlation heatmap ...")
            t0 = _time.perf_counter()
            fig_correlation_for_dataset(df, dataset_name, phys_cols)
            print(f"  [{_time.perf_counter() - t0:.1f}s]")
        else:
            print(f"\n  [Phase 6] Need >= 2 PHYS_COLUMNS -- skipping.")
    else:
        print(f"\n  [Phase 6] SKIPPED (resume)")

    # Phase 7: Range statistics
    if start_phase <= 7:
        if phys_cols:
            avail_phys = [c for c in phys_cols if c in set(df.columns)]
            if avail_phys:
                print(f"\n  [Phase 7] Computing range stats ...")
                t0 = _time.perf_counter()
                all_range_stats_accumulator[dataset_name] = \
                    compute_range_stats_spark(df, avail_phys)
                print(f"  [{_time.perf_counter() - t0:.1f}s]")
            else:
                print(f"\n  [Phase 7] No available PHYS_COLUMNS -- skipping.")
        else:
            print(f"\n  [Phase 7] No PHYS_COLUMNS configured -- skipping.")
    else:
        print(f"\n  [Phase 7] SKIPPED (resume)")

    # Phase 8: Sample generation
    print(f"\n  [Phase 8] Generating representative sample ...")
    t0 = _time.perf_counter()
    try:
        generate_sample(df, dataset_name, row_count)
    except Exception as exc:
        print(f"  [ERROR] Sampling failed: {exc}")
    print(f"  [{_time.perf_counter() - t0:.1f}s]")

    elapsed = _time.perf_counter() - t_dataset_start
    print(f"\n  DONE: {dataset_name}  ({elapsed:.2f}s)")
    return elapsed


# =====================================================================
# MAIN ENTRY POINT
# =====================================================================

def main() -> int:
    wall_start = _time.perf_counter()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(SAMPLE_DIR, exist_ok=True)
    print(f"  Output dir  : {os.path.abspath(OUTPUT_DIR)}")
    print(f"  Sample dir  : {os.path.abspath(SAMPLE_DIR)}")

    try:
        spark = build_spark_session()
    except Exception as exc:
        print(f"[FATAL] Failed to create SparkSession: {exc}")
        return 2

    try:
        datasets = discover_parquet_datasets(DATA_DIRS)
    except Exception as exc:
        print(f"[FATAL] {exc}")
        spark.stop()
        return 2

    if not datasets:
        print("[FATAL] No .parquet datasets found.")
        spark.stop()
        return 2

    print(f"\n  Discovered {len(datasets)} dataset(s):")
    for i, ds in enumerate(datasets, 1):
        print(f"    {i}. {os.path.basename(ds)}")

    dataset_meta    = OrderedDict()
    all_range_stats = OrderedDict()
    dataset_times   = OrderedDict()
    n_failures      = 0

    for ds_idx, dataset_path in enumerate(datasets):
        name = os.path.basename(dataset_path)
        if ds_idx < RESUME_DATASET_IDX:
            print(f"\n  [SKIP] {name} (index {ds_idx} < RESUME_DATASET_IDX)")
            continue
        start_phase = RESUME_PHASE if ds_idx == RESUME_DATASET_IDX else 1
        try:
            elapsed = run_eda_for_dataset(
                spark, dataset_path,
                dataset_meta, all_range_stats,
                start_phase=start_phase,
            )
            dataset_times[name] = elapsed
        except Exception as exc:
            print(f"  [ERROR] Unhandled exception for {name}: {exc}")
            dataset_times[name] = 0.0
            n_failures += 1

    if dataset_meta:
        print(f"\n  Building dataset overview ...")
        fig_dataset_overview(dataset_meta)
        print(f"  Building completeness heatmap ...")
        fig_data_completeness(dataset_meta)
        print(f"  Building null structure chart ...")
        fig_null_structure(dataset_meta)

    if all_range_stats:
        print(f"  Building physical ranges chart ...")
        fig_physical_ranges(all_range_stats)

    print_preprocessing_plan()

    total_wall = _time.perf_counter() - wall_start
    print_performance_report(dataset_times, total_wall)

    spark.stop()

    if n_failures > 0:
        print(f"[WARNING] {n_failures} dataset(s) encountered errors.")
        return 1

    print(f"\n  EDA PIPELINE COMPLETE")
    print(f"  Total time : {total_wall:.1f}s ({total_wall/60:.1f} min)")
    print(f"  Plots dir  : {os.path.abspath(OUTPUT_DIR)}")
    print(f"  Sample dir : {os.path.abspath(SAMPLE_DIR)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())