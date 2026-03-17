# =====================================================================
# 02_data_exploration_plots_only.py

# Antarctic Basal Melt Prediction -- Step 02: Plots-Only EDA Pipeline

# Regenerates the EDA figure set without rerunning the full Step 01
# profiling workflow. Uses saved sample parquet files for sample-based
# figures and the fused sparse feature table for the full monthly bonus
# time-series plots.

# This version includes:
# 1. Core EDA figures fig_01 through fig_06
# 2. Spatial bonus figures fig_07 and fig_08 from sample parquet files
# 3. Full-data monthly time-series figures fig_09 and fig_10
# 4. Formatting aligned with the attached project pipeline scripts

# Runs on SDSC Expanse via: sbatch 02_run_data_exploration_plots_only.sh
# =====================================================================

from __future__ import annotations

import os
import sys
import math
import time as _time
from collections import OrderedDict

import matplotlib
os.environ["MPLCONFIGDIR"] = os.path.join(os.getcwd(), ".matplotlib_cache")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from plot_config import apply_style, STYLE, new_fig, save_fig, style_colorbar
apply_style()

# =====================================================================
# 1. CONFIGURATION
# =====================================================================

# -- HPC Resource Allocation ------------------------------------------
# Formula mirrors the main Step 01 EDA pipeline.
# executor_instances = (total_cores - 1) // executor_cores
# executor_memory = floor((total_memory - driver_memory) / executor_instances)

TOTAL_CORES = 32
TOTAL_MEMORY_GB = 128
EXECUTOR_CORES = 5
EXECUTOR_INSTANCES = max((TOTAL_CORES - 1) // EXECUTOR_CORES, 1)
DRIVER_MEMORY_GB = 10
EXECUTOR_MEMORY_GB = max(
    math.floor((TOTAL_MEMORY_GB - DRIVER_MEMORY_GB) / EXECUTOR_INSTANCES),
    1,
)
SHUFFLE_PARTITIONS = 2 * TOTAL_CORES

# -- Paths ------------------------------------------------------------
SAMPLE_DIR = os.path.join("data", "sample_data")
FUSED_PATH = os.path.join(
    "data", "fused_data", "antarctica_sparse_features.parquet"
)
OUTPUT_DIR = os.path.join("data", "eda_plots")
SCRATCH = os.environ.get(
    "SPARK_LOCAL_DIRS",
    os.path.join(os.getcwd(), "spark_scratch"),
)

# -- Plot config ------------------------------------------------------
HISTOGRAM_BINS = 50
FIG_DPI = 150

# -- Dataset labels / colours ----------------------------------------
DATASET_LABELS = {
    "bedmap3_static": "Bedmap3 Static",
    "grace": "GRACE",
    "icesat2_dynamic": "ICESat-2 Dynamic",
    "ocean_dynamic": "Ocean Dynamic",
    "antarctica_sparse_features": "Fused (Sparse)",
}

DATASET_COLOURS = {
    "bedmap3_static": STYLE["PURPLE"],
    "grace": STYLE["BLUE"],
    "icesat2_dynamic": STYLE["GREEN"],
    "ocean_dynamic": STYLE["AMBER"],
    "antarctica_sparse_features": STYLE["RED"],
}

PHYS_COLUMNS = {
    "bedmap3_static": [
        "surface", "bed", "thickness", "bed_slope",
        "dist_to_grounding_line", "clamped_depth", "ice_draft",
    ],
    "grace": ["lwe_length"],
    "icesat2_dynamic": [
        "delta_h", "ice_area", "h_surface_dynamic", "surface_slope",
    ],
    "ocean_dynamic": ["thetao", "so", "T_f", "T_star"],
    "antarctica_sparse_features": [
        "surface", "bed", "thickness", "lwe_length",
        "delta_h", "thetao", "so", "T_f", "T_star",
    ],
}

SAMPLE_FILE_MAP = {
    "bedmap3_static": "bedmap3_static_sample.parquet",
    "antarctica_sparse_features": "antarctica_sparse_features_sample.parquet",
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================================
# 2. SPARK SESSION
# =====================================================================


def build_spark_session() -> SparkSession:
    """HPC-optimised SparkSession for plots-only regeneration."""
    if EXECUTOR_MEMORY_GB < 1:
        raise RuntimeError(
            f"Derived executor memory is {EXECUTOR_MEMORY_GB}g -- "
            f"allocation too small for {EXECUTOR_INSTANCES} executors."
        )

    driver_mem = f"{DRIVER_MEMORY_GB}g"
    exec_mem = f"{EXECUTOR_MEMORY_GB}g"

    print("=" * 72)
    print(" HPC SparkSession Configuration")
    print("=" * 72)
    print(f" TOTAL_CORES ............. {TOTAL_CORES}")
    print(f" TOTAL_MEMORY_GB ......... {TOTAL_MEMORY_GB}")
    print(f" DRIVER_MEMORY ........... {driver_mem}")
    print(f" EXECUTOR_CORES .......... {EXECUTOR_CORES}")
    print(f" EXECUTOR_INSTANCES ...... {EXECUTOR_INSTANCES}")
    print(f" EXECUTOR_MEMORY ......... {exec_mem}")
    print(f" SHUFFLE_PARTITIONS ...... {SHUFFLE_PARTITIONS}")
    print("=" * 72)

    spark = (
        SparkSession.builder
        .appName("HPC_Antarctic_EDA_Plots_Only")
        .config("spark.driver.memory", driver_mem)
        .config("spark.executor.instances", str(EXECUTOR_INSTANCES))
        .config("spark.executor.cores", str(EXECUTOR_CORES))
        .config("spark.executor.memory", exec_mem)
        .config("spark.sql.shuffle.partitions", str(SHUFFLE_PARTITIONS))
        .config("spark.driver.maxResultSize", "4g")
        .config("spark.network.timeout", "1200s")
        .config("spark.sql.sources.parallelPartitionDiscovery.threshold", "32")
        .config("spark.sql.sources.parallelPartitionDiscovery.parallelism", "64")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .config("spark.sql.adaptive.advisoryPartitionSizeInBytes", "128m")
        .config("spark.sql.parquet.filterPushdown", "true")
        .config("spark.sql.parquet.mergeSchema", "false")
        .config("spark.local.dir", SCRATCH)
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")
    os.makedirs(spark.conf.get("spark.local.dir"), exist_ok=True)
    return spark


# =====================================================================
# 3. SAMPLE DISCOVERY + LOADING
# =====================================================================


def load_samples(spark: SparkSession) -> OrderedDict:
    """Load all Spark sample parquet directories from SAMPLE_DIR."""
    samples = OrderedDict()

    if not os.path.isdir(SAMPLE_DIR):
        print(f" [WARN] Sample dir not found: {os.path.abspath(SAMPLE_DIR)}")
        return samples

    for entry in sorted(os.listdir(SAMPLE_DIR)):
        if not entry.endswith("_sample.parquet"):
            continue
        key = entry.replace("_sample.parquet", "")
        path = os.path.join(SAMPLE_DIR, entry)
        try:
            df = spark.read.option("recursiveFileLookup", "true").parquet(path)
            samples[key] = df
            print(f" Loaded {key}: {df.count():,} rows, {len(df.columns)} cols")
        except Exception as exc:
            print(f" [WARN] Could not load {entry}: {exc}")

    return samples


def load_sample_pandas(name: str) -> pd.DataFrame | None:
    """Load a sample parquet directory with pandas for spatial bonus plots."""
    path = os.path.join(SAMPLE_DIR, name)
    if not os.path.exists(path):
        print(f" [WARN] Sample not found: {os.path.abspath(path)}")
        return None

    try:
        pdf = pd.read_parquet(path)
        print(f" Loaded {len(pdf):,} rows from {name}")
        return pdf
    except Exception as exc:
        print(f" [WARN] Failed to read {name}: {exc}")
        return None


# =====================================================================
# 4. DISTRIBUTED METRIC COMPUTATION
# =====================================================================


def compute_histogram_spark(df, col_name: str, n_bins: int = HISTOGRAM_BINS):
    """Executor-side histogram via RDD.histogram()."""
    try:
        rdd = df.select(col_name).na.drop().rdd.map(lambda r: float(r[0]))
        if rdd.isEmpty():
            return None, None
        return rdd.histogram(n_bins)
    except Exception as exc:
        print(f" [WARN] Histogram failed for '{col_name}': {exc}")
        return None, None


def compute_correlation_matrix_spark(df, cols: list) -> np.ndarray:
    """Pairwise Pearson correlations via df.stat.corr()."""
    n = len(cols)
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


def compute_range_stats(df, cols: list) -> dict:
    """Single .agg() call for min/max/mean/std across all columns."""
    available = [c for c in cols if c in df.columns]
    if not available:
        return {}

    agg_exprs = []
    for c in available:
        agg_exprs += [
            F.min(c).alias(f"{c}__min"),
            F.max(c).alias(f"{c}__max"),
            F.mean(c).alias(f"{c}__mean"),
            F.stddev(c).alias(f"{c}__std"),
        ]

    row = df.agg(*agg_exprs).head()
    return {
        c: {
            "min": float(row[f"{c}__min"]) if row[f"{c}__min"] is not None else float("nan"),
            "max": float(row[f"{c}__max"]) if row[f"{c}__max"] is not None else float("nan"),
            "mean": float(row[f"{c}__mean"]) if row[f"{c}__mean"] is not None else float("nan"),
            "std": float(row[f"{c}__std"]) if row[f"{c}__std"] is not None else float("nan"),
        }
        for c in available
    }


def compute_completeness(samples: OrderedDict) -> dict:
    """Percent non-null by column for each sample dataset."""
    completeness = {}
    for key, df in samples.items():
        total = df.count()
        exprs = [F.count(F.when(F.col(c).isNotNull(), c)).alias(c) for c in df.columns]
        row = df.agg(*exprs).head()
        completeness[key] = {
            c: (int(row[c]) / total * 100 if total > 0 and row[c] is not None else 0.0)
            for c in df.columns
        }
    return completeness


def compute_null_counts(df, total: int) -> dict:
    """Null counts per column in one aggregate pass."""
    exprs = [F.count(F.when(F.col(c).isNotNull(), c)).alias(c) for c in df.columns]
    row = df.agg(*exprs).head()
    return {
        c: max(0, total - (int(row[c]) if row[c] is not None else 0))
        for c in df.columns
    }


# =====================================================================
# 5. FIGURE 01 -- DATASET OVERVIEW
# =====================================================================


def fig_dataset_overview(samples: OrderedDict):
    """Two-panel bar chart: row counts and column counts per dataset."""
    names = list(samples.keys())
    labels = [DATASET_LABELS.get(n, n) for n in names]
    rows = [samples[n].count() for n in names]
    cols = [len(samples[n].columns) for n in names]
    colors = [DATASET_COLOURS.get(n, STYLE["BLUE"]) for n in names]

    fig, (ax1, ax2) = new_fig(nrows=1, ncols=2, figsize=(14, 5))
    fig.suptitle(
        "Dataset Overview",
        color=STYLE["TEXT"],
        fontsize=16,
        fontweight="bold",
        y=1.02,
    )

    bars1 = ax1.barh(labels, rows, color=colors, edgecolor=STYLE["BG"], alpha=0.85)
    ax1.set_xscale("log")
    ax1.set_xlabel("Row Count (log scale)", color=STYLE["TEXT"], fontsize=9)
    ax1.set_title("Rows per Dataset", color=STYLE["TEXT"])
    ax1.tick_params(colors=STYLE["TEXT"])
    ax1.xaxis.grid(True, color=STYLE["GRID"], lw=0.5, alpha=0.6)
    ax1.yaxis.grid(False)
    for bar, val in zip(bars1, rows):
        ax1.text(
            bar.get_width() * 1.1,
            bar.get_y() + bar.get_height() / 2,
            f"{val:,.0f}",
            va="center",
            fontsize=9,
            color=STYLE["TEXT"],
        )

    bars2 = ax2.barh(labels, cols, color=colors, edgecolor=STYLE["BG"], alpha=0.85)
    ax2.set_xlabel("Column Count", color=STYLE["TEXT"], fontsize=9)
    ax2.set_title("Columns per Dataset", color=STYLE["TEXT"])
    ax2.tick_params(colors=STYLE["TEXT"])
    ax2.xaxis.grid(True, color=STYLE["GRID"], lw=0.5, alpha=0.6)
    ax2.yaxis.grid(False)
    for bar, val in zip(bars2, cols):
        ax2.text(
            bar.get_width() + 0.2,
            bar.get_y() + bar.get_height() / 2,
            str(val),
            va="center",
            fontsize=9,
            color=STYLE["TEXT"],
        )

    fig.tight_layout()
    save_fig(fig, OUTPUT_DIR, "fig_01_dataset_overview.png")


# =====================================================================
# 6. FIGURE 02 -- DATA COMPLETENESS
# =====================================================================


def fig_data_completeness(samples: OrderedDict):
    """Heatmap of percent non-null per column per dataset."""
    completeness = compute_completeness(samples)

    all_cols = []
    seen = set()
    for key in samples:
        for c in samples[key].columns:
            if c not in seen:
                all_cols.append(c)
                seen.add(c)

    names = list(samples.keys())
    labels = [DATASET_LABELS.get(n, n) for n in names]
    matrix = np.full((len(names), len(all_cols)), np.nan)
    for i, name in enumerate(names):
        comp = completeness.get(name, {})
        for j, c in enumerate(all_cols):
            if c in comp:
                matrix[i, j] = comp[c]

    fig_w = max(20, len(all_cols) * 0.55)
    fig_h = max(4, len(names) * 1.2)
    fig, ax = new_fig(figsize=(fig_w, fig_h))

    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=100)
    ax.set_xticks(range(len(all_cols)))
    ax.set_xticklabels(all_cols, rotation=50, ha="right", fontsize=9, color=STYLE["TEXT"])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=11, color=STYLE["TEXT"])
    ax.set_title(
        "Data Completeness (Percent Non-Null)",
        color=STYLE["TEXT"],
        fontsize=14,
        fontweight="bold",
        pad=12,
    )
    ax.grid(False)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            if np.isnan(val):
                ax.text(j, i, "N/A", ha="center", va="center", fontsize=7, color=STYLE["GRID"])
            else:
                tc = STYLE["TEXT"] if val > 50 else "#ffffff"
                ax.text(
                    j,
                    i,
                    f"{val:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=8,
                    fontweight="bold",
                    color=tc,
                )

    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    style_colorbar(cb, "% Non-Null")
    fig.tight_layout()
    save_fig(fig, OUTPUT_DIR, "fig_02_data_completeness.png")


# =====================================================================
# 7. FIGURE 03 -- HISTOGRAMS
# =====================================================================


def fig_histograms_for_dataset(df, key: str, colour: str):
    """Multi-panel histogram figure for configured physical columns."""
    phys_cols = PHYS_COLUMNS.get(key, [])
    available = set(df.columns)
    cols_to_plot = [c for c in phys_cols if c in available]
    if not cols_to_plot:
        print(f" [SKIP] No plottable columns for {key}")
        return

    n = len(cols_to_plot)
    ncols_grid = min(n, 3)
    nrows_grid = math.ceil(n / ncols_grid)

    fig, axes = new_fig(
        nrows=nrows_grid,
        ncols=ncols_grid,
        figsize=(6 * ncols_grid, 4 * nrows_grid),
    )
    axes_flat = np.array(axes).flatten() if n > 1 else np.array([axes])

    label = DATASET_LABELS.get(key, key)
    fig.suptitle(
        f"Distributions : {label}",
        color=STYLE["TEXT"],
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    for idx, col_name in enumerate(cols_to_plot):
        ax = axes_flat[idx]
        print(f" Computing histogram: {col_name} ...")
        edges, counts = compute_histogram_spark(df, col_name, HISTOGRAM_BINS)

        if edges is None:
            ax.text(0.5, 0.5, "No Data", ha="center", va="center", transform=ax.transAxes, color=STYLE["GRID"])
            ax.set_title(col_name, color=STYLE["TEXT"])
            continue

        centres = [(edges[i] + edges[i + 1]) / 2 for i in range(len(counts))]
        widths = [edges[i + 1] - edges[i] for i in range(len(counts))]
        ax.bar(
            centres,
            counts,
            width=widths,
            color=colour,
            edgecolor=STYLE["BG"],
            alpha=0.85,
            linewidth=0.3,
        )
        ax.set_title(col_name, color=STYLE["TEXT"], fontsize=11)
        ax.set_ylabel("Count", color=STYLE["TEXT"], fontsize=9)
        ax.yaxis.set_major_formatter(mticker.EngFormatter())
        ax.tick_params(colors=STYLE["TEXT"])

    for idx in range(len(cols_to_plot), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.tight_layout()
    save_fig(fig, OUTPUT_DIR, f"fig_03_histograms_{key}.png")


# =====================================================================
# 8. FIGURE 04 -- CORRELATION HEATMAP
# =====================================================================


def fig_correlation_for_dataset(df, key: str):
    """Correlation heatmap using df.stat.corr()."""
    phys_cols = PHYS_COLUMNS.get(key, [])
    available = set(df.columns)
    cols_to_use = [c for c in phys_cols if c in available]
    if len(cols_to_use) < 2:
        print(f" [SKIP] Need >= 2 numeric columns for correlation: {key}")
        return

    print(f" Computing {len(cols_to_use)}x{len(cols_to_use)} correlation matrix ...")
    corr = compute_correlation_matrix_spark(df, cols_to_use)

    fig, ax = new_fig(figsize=(max(7, len(cols_to_use) * 0.9), max(6, len(cols_to_use) * 0.8)))
    label = DATASET_LABELS.get(key, key)
    ax.set_title(
        f"Correlation Matrix : {label}",
        color=STYLE["TEXT"],
        fontsize=13,
        fontweight="bold",
    )

    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1, aspect="equal")
    ax.set_xticks(range(len(cols_to_use)))
    ax.set_xticklabels(cols_to_use, rotation=45, ha="right", fontsize=9, color=STYLE["TEXT"])
    ax.set_yticks(range(len(cols_to_use)))
    ax.set_yticklabels(cols_to_use, fontsize=9, color=STYLE["TEXT"])

    for i in range(len(cols_to_use)):
        for j in range(len(cols_to_use)):
            val = corr[i, j]
            tc = STYLE["TEXT"] if abs(val) < 0.6 else "#ffffff"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color=tc)

    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    style_colorbar(cb, "Pearson r")
    ax.grid(False)
    fig.tight_layout()
    save_fig(fig, OUTPUT_DIR, f"fig_04_correlation_{key}.png")


# =====================================================================
# 9. FIGURE 05 -- PHYSICAL RANGES
# =====================================================================


def fig_physical_ranges(samples: OrderedDict):
    """Horizontal range bars: min/max with mean and std overlay."""
    entries = []
    for key, df in samples.items():
        phys_cols = PHYS_COLUMNS.get(key, [])
        if not phys_cols:
            continue
        colour = DATASET_COLOURS.get(key, STYLE["BLUE"])
        label = DATASET_LABELS.get(key, key)
        stats = compute_range_stats(df, phys_cols)
        for col_name, vals in stats.items():
            entries.append(
                {
                    "label": f"{col_name}\n({label})",
                    "min": vals["min"],
                    "max": vals["max"],
                    "mean": vals["mean"],
                    "std": vals["std"],
                    "colour": colour,
                }
            )

    if not entries:
        print(" [SKIP] No range statistics available.")
        return

    n = len(entries)
    fig, ax = new_fig(figsize=(14, max(6, n * 0.5)))
    ax.set_title(
        "Physical Variable Ranges (Min / Max / Mean +/- Std)",
        color=STYLE["TEXT"],
        fontsize=14,
        fontweight="bold",
    )

    for i, e in enumerate(entries):
        mn, mx, mu, sd = e["min"], e["max"], e["mean"], e["std"]
        if any(math.isnan(v) for v in [mn, mx, mu]):
            continue
        rng = abs(mx - mn)
        if rng > 0:
            ax.barh(i, rng, left=mn, height=0.6, color=e["colour"], alpha=0.3, edgecolor=e["colour"])
        ax.plot(mu, i, "D", color=e["colour"], markersize=6, markeredgecolor="#ffffff", markeredgewidth=0.5)
        if not math.isnan(sd):
            ax.plot([mu - sd, mu + sd], [i, i], color=e["colour"], linewidth=2, solid_capstyle="round")

    ax.set_yticks(range(n))
    ax.set_yticklabels([e["label"] for e in entries], fontsize=8, color=STYLE["TEXT"])
    ax.set_xlabel("Value (note: scale varies per variable)", color=STYLE["TEXT"], fontsize=9)
    ax.tick_params(colors=STYLE["TEXT"])
    ax.xaxis.grid(True, color=STYLE["GRID"], lw=0.5, alpha=0.5)
    ax.yaxis.grid(False)
    ax.invert_yaxis()
    ax.annotate(
        "Note: ice_area extends to large values, so linear readability is prioritised",
        xy=(0.01, 0.01),
        xycoords="axes fraction",
        fontsize=8,
        color=STYLE["AMBER"],
        ha="left",
        va="bottom",
    )

    fig.tight_layout()
    save_fig(fig, OUTPUT_DIR, "fig_05_physical_ranges.png")


# =====================================================================
# 10. FIGURE 06 -- NULL STRUCTURE
# =====================================================================


def fig_null_structure(samples: OrderedDict):
    """Grouped horizontal bar chart of null counts per column."""
    all_cols = []
    seen = set()
    for key in samples:
        for c in samples[key].columns:
            if c not in seen:
                all_cols.append(c)
                seen.add(c)

    names = list(samples.keys())
    null_data = {}
    for key, df in samples.items():
        total = df.count()
        null_data[key] = compute_null_counts(df, total)

    fig, ax = new_fig(figsize=(16, max(6, len(all_cols) * 0.38)))
    ax.set_title(
        "Null Counts per Column (log scale)",
        color=STYLE["TEXT"],
        fontsize=14,
        fontweight="bold",
    )

    y_pos = np.arange(len(all_cols))
    bar_height = 0.7 / max(len(names), 1)

    for ds_idx, key in enumerate(names):
        nulls = null_data.get(key, {})
        label = DATASET_LABELS.get(key, key)
        colour = DATASET_COLOURS.get(key, STYLE["BLUE"])
        vals = [max(nulls.get(c, 0), 0) for c in all_cols]
        ax.barh(
            y_pos + ds_idx * bar_height,
            vals,
            height=bar_height,
            color=colour,
            alpha=0.8,
            label=label,
            edgecolor=STYLE["BG"],
            linewidth=0.3,
        )

    ax.set_yticks(y_pos + bar_height * (len(names) - 1) / 2)
    ax.set_yticklabels(all_cols, fontsize=8, color=STYLE["TEXT"])
    ax.set_xlabel("Null Count (log scale)", color=STYLE["TEXT"], fontsize=9)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(mticker.EngFormatter())
    ax.tick_params(colors=STYLE["TEXT"])
    ax.xaxis.grid(True, color=STYLE["GRID"], lw=0.5, alpha=0.5)
    ax.yaxis.grid(False)
    ax.legend(fontsize=8, facecolor=STYLE["BG"], edgecolor=STYLE["GRID"], labelcolor=STYLE["TEXT"])
    ax.invert_yaxis()
    fig.tight_layout()
    save_fig(fig, OUTPUT_DIR, "fig_06_null_structure.png")


# =====================================================================
# 11. MONTHLY AGGREGATION
# =====================================================================


def compute_monthly_aggregates(spark: SparkSession) -> pd.DataFrame | None:
    """Aggregate full fused dataset by month for delta_h and lwe_fused."""
    fused_abs = os.path.abspath(FUSED_PATH)
    if not os.path.exists(fused_abs):
        print(f" [SKIP] Fused dataset not found: {fused_abs}")
        return None

    print("\n Computing monthly aggregates from full fused dataset ...")
    print(f" Path: {fused_abs}")
    t0 = _time.perf_counter()

    df = (
        spark.read
        .option("recursiveFileLookup", "true")
        .option("mergeSchema", "true")
        .parquet(fused_abs)
    )

    lwe_col = None
    for candidate in ["lwe_fused", "lwe_mo", "lwe_length"]:
        if candidate in df.columns:
            lwe_col = candidate
            break

    agg_exprs = [F.mean("delta_h").alias("mean_delta_h"), F.count("*").alias("n_rows")]
    if lwe_col is not None:
        agg_exprs.append(F.mean(lwe_col).alias("mean_lwe_fused"))
    else:
        agg_exprs.append(F.lit(None).cast("double").alias("mean_lwe_fused"))

    monthly = (
        df.withColumn("month_year", F.date_trunc("month", "exact_time"))
        .groupBy("month_year")
        .agg(*agg_exprs)
        .orderBy("month_year")
    )

    pdf = monthly.toPandas()
    pdf["month_year"] = pd.to_datetime(pdf["month_year"])
    pdf = pdf.dropna(subset=["month_year"]).sort_values("month_year")

    elapsed = _time.perf_counter() - t0
    print(f" -> {len(pdf)} monthly records aggregated in {elapsed:.1f}s")
    if not pdf.empty:
        print(f" Date range: {pdf['month_year'].min()} to {pdf['month_year'].max()}")

    return pdf


# =====================================================================
# 12. FIGURE 07 -- ICE MASK + OCEAN COVERAGE
# =====================================================================


def fig_07_ice_mask_ocean_coverage():
    """2x1 spatial scatter: Bedmap mask and fused ocean-data coverage."""
    print("\n Building fig_07: Ice Mask + Ocean Coverage ...")

    pdf_bedmap = load_sample_pandas(SAMPLE_FILE_MAP["bedmap3_static"])
    pdf_fused = load_sample_pandas(SAMPLE_FILE_MAP["antarctica_sparse_features"])

    if pdf_bedmap is None or pdf_fused is None:
        print(" [SKIP] Required sample data not available.")
        return

    fig, axes = new_fig(nrows=2, ncols=1, figsize=(12, 18))
    ax1, ax2 = axes
    fig.suptitle(
        "Ice Mask and Ocean Data Coverage",
        color=STYLE["TEXT"],
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    mask_palette = {
        1: STYLE["GREEN"],
        2: STYLE["BLUE"],
        3: STYLE["AMBER"],
    }
    mask_labels = {
        1: "Grounded Ice",
        2: "Floating Ice",
        3: "Ocean",
    }

    if {"x", "y", "mask"}.issubset(pdf_bedmap.columns):
        mask_values = sorted(pd.Series(pdf_bedmap["mask"]).dropna().unique())
        for mask_val in mask_values:
            mv = int(mask_val)
            subset = pdf_bedmap[pdf_bedmap["mask"] == mask_val]
            ax1.scatter(
                subset["x"],
                subset["y"],
                s=0.15,
                alpha=0.35,
                rasterized=True,
                c=mask_palette.get(mv, "#888888"),
                label=mask_labels.get(mv, f"Mask={mv}"),
            )
        ax1.legend(
            markerscale=25,
            loc="upper right",
            fontsize=9,
            facecolor=STYLE["BG"],
            edgecolor=STYLE["GRID"],
            labelcolor=STYLE["TEXT"],
        )
    else:
        ax1.text(0.5, 0.5, "Required columns missing", ha="center", va="center", transform=ax1.transAxes, color=STYLE["TEXT"])

    ax1.set_title("Bedmap3 Ice Mask Classification", color=STYLE["TEXT"])
    ax1.set_xlabel("x (EPSG:3031, m)", color=STYLE["TEXT"])
    ax1.set_ylabel("y (EPSG:3031, m)", color=STYLE["TEXT"])
    ax1.set_aspect("equal")
    ax1.grid(alpha=0.2)
    ax1.xaxis.set_major_formatter(mticker.EngFormatter())
    ax1.yaxis.set_major_formatter(mticker.EngFormatter())

    ocean_col = None
    for candidate in ["thetao_mo", "thetao", "T_star", "t_star_mo"]:
        if candidate in pdf_fused.columns:
            ocean_col = candidate
            break

    if ocean_col is None or not {"x", "y"}.issubset(pdf_fused.columns):
        ax2.text(0.5, 0.5, "No ocean coverage column found", ha="center", va="center", transform=ax2.transAxes, color=STYLE["TEXT"])
    else:
        has_ocean = pdf_fused[ocean_col].notna()
        no_ocean = ~has_ocean
        ax2.scatter(
            pdf_fused.loc[no_ocean, "x"],
            pdf_fused.loc[no_ocean, "y"],
            s=0.1,
            alpha=0.12,
            c="#aeb7c4",
            label="No Ocean Data",
            rasterized=True,
        )
        ax2.scatter(
            pdf_fused.loc[has_ocean, "x"],
            pdf_fused.loc[has_ocean, "y"],
            s=0.35,
            alpha=0.55,
            c=STYLE["RED"],
            label="Valid Ocean Data",
            rasterized=True,
        )
        ax2.legend(
            markerscale=20,
            loc="upper right",
            fontsize=9,
            facecolor=STYLE["BG"],
            edgecolor=STYLE["GRID"],
            labelcolor=STYLE["TEXT"],
        )

    ax2.set_title(
        f"Ocean Data Coverage ({ocean_col if ocean_col is not None else 'missing'})",
        color=STYLE["TEXT"],
    )
    ax2.set_xlabel("x (EPSG:3031, m)", color=STYLE["TEXT"])
    ax2.set_ylabel("y (EPSG:3031, m)", color=STYLE["TEXT"])
    ax2.set_aspect("equal")
    ax2.grid(alpha=0.2)
    ax2.xaxis.set_major_formatter(mticker.EngFormatter())
    ax2.yaxis.set_major_formatter(mticker.EngFormatter())

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save_fig(fig, OUTPUT_DIR, "fig_07_ice_mask_ocean_coverage.png")


# =====================================================================
# 13. FIGURE 08 -- DELTA_H + LWE SPATIAL
# =====================================================================


def fig_08_delta_h_vs_lwe_spatial():
    """2x1 spatial scatter for delta_h and LWE-like fused mass anomaly."""
    print("\n Building fig_08: delta_h vs LWE Fused (spatial) ...")

    pdf = load_sample_pandas(SAMPLE_FILE_MAP["antarctica_sparse_features"])
    if pdf is None:
        print(" [SKIP] Fused sample not available.")
        return

    fig, axes = new_fig(nrows=2, ncols=1, figsize=(12, 18))
    ax1, ax2 = axes
    fig.suptitle(
        "Height Change and Mass Anomaly : Spatial Distribution",
        color=STYLE["TEXT"],
        fontsize=16,
        fontweight="bold",
        y=0.98,
    )

    valid_dh = pdf.dropna(subset=["delta_h", "x", "y"]) if {"delta_h", "x", "y"}.issubset(pdf.columns) else pd.DataFrame()
    if valid_dh.empty:
        ax1.text(0.5, 0.5, "No delta_h Data", ha="center", va="center", transform=ax1.transAxes, color=STYLE["TEXT"])
    else:
        vmin = valid_dh["delta_h"].quantile(0.01)
        vmax = valid_dh["delta_h"].quantile(0.99)
        sc1 = ax1.scatter(
            valid_dh["x"],
            valid_dh["y"],
            s=0.2,
            alpha=0.45,
            rasterized=True,
            c=valid_dh["delta_h"],
            cmap="RdBu_r",
            vmin=vmin,
            vmax=vmax,
        )
        cb1 = fig.colorbar(sc1, ax=ax1, shrink=0.7, pad=0.04)
        style_colorbar(cb1, "delta_h (m)")

    ax1.set_title("Ice Height Change (delta_h)", color=STYLE["TEXT"])
    ax1.set_xlabel("x (EPSG:3031, m)", color=STYLE["TEXT"])
    ax1.set_ylabel("y (EPSG:3031, m)", color=STYLE["TEXT"])
    ax1.set_aspect("equal")
    ax1.grid(alpha=0.2)
    ax1.xaxis.set_major_formatter(mticker.EngFormatter())
    ax1.yaxis.set_major_formatter(mticker.EngFormatter())

    lwe_col = None
    for candidate in ["lwe_fused", "lwe_mo", "lwe_length"]:
        if candidate in pdf.columns:
            lwe_col = candidate
            break

    if lwe_col is None or not {"x", "y"}.issubset(pdf.columns):
        ax2.text(0.5, 0.5, "No LWE Column Found", ha="center", va="center", transform=ax2.transAxes, color=STYLE["TEXT"])
    else:
        valid_lwe = pdf.dropna(subset=[lwe_col, "x", "y"])
        if valid_lwe.empty:
            ax2.text(0.5, 0.5, f"No valid {lwe_col} Data", ha="center", va="center", transform=ax2.transAxes, color=STYLE["TEXT"])
        else:
            vmin = valid_lwe[lwe_col].quantile(0.01)
            vmax = valid_lwe[lwe_col].quantile(0.99)
            sc2 = ax2.scatter(
                valid_lwe["x"],
                valid_lwe["y"],
                s=0.2,
                alpha=0.45,
                rasterized=True,
                c=valid_lwe[lwe_col],
                cmap="BrBG",
                vmin=vmin,
                vmax=vmax,
            )
            cb2 = fig.colorbar(sc2, ax=ax2, shrink=0.7, pad=0.04)
            style_colorbar(cb2, f"{lwe_col}")

    ax2.set_title(f"Liquid Water Equivalent ({lwe_col if lwe_col is not None else 'missing'})", color=STYLE["TEXT"])
    ax2.set_xlabel("x (EPSG:3031, m)", color=STYLE["TEXT"])
    ax2.set_ylabel("y (EPSG:3031, m)", color=STYLE["TEXT"])
    ax2.set_aspect("equal")
    ax2.grid(alpha=0.2)
    ax2.xaxis.set_major_formatter(mticker.EngFormatter())
    ax2.yaxis.set_major_formatter(mticker.EngFormatter())

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save_fig(fig, OUTPUT_DIR, "fig_08_delta_h_vs_lwe_spatial.png")


# =====================================================================
# 14. TIME-SERIES HELPER
# =====================================================================


def plot_timeseries_with_trend(
    ax,
    dates: pd.Series,
    values: pd.Series,
    line_colour: str,
    ylabel: str,
    title: str,
    annotation_y_factor: float = 0.92,
):
    """Shared monthly time-series plot with seasonal shading and trend."""
    ax.plot(
        dates,
        values,
        color=line_colour,
        linewidth=1.8,
        marker="o",
        markersize=3,
        label="Monthly Mean",
    )

    x_num = mdates.date2num(dates)
    coeffs = np.polyfit(x_num, values, 1)
    y_pred = np.polyval(coeffs, x_num)
    ax.plot(dates, y_pred, color=STYLE["RED"], linestyle="--", linewidth=1.5, label="Linear Trend")

    years = sorted(set(pd.DatetimeIndex(dates).year))
    for year in years:
        start = pd.Timestamp(year=year, month=3, day=1)
        end = pd.Timestamp(year=year, month=8, day=31)
        ax.axvspan(start, end, color=STYLE["BLUE"], alpha=0.08, linewidth=0)

    y_min = float(np.nanmin(values))
    y_max = float(np.nanmax(values))
    y_range = y_max - y_min
    if y_range <= 0:
        y_range = max(abs(y_max), 1.0)
    ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.2 * y_range)

    ax.text(
        0.02,
        0.92,
        "Shaded = Antarctic Fall-Winter (March-August)",
        color=STYLE["BLUE"],
        fontsize=10,
        transform=ax.transAxes,
        bbox=dict(facecolor=STYLE["BG"], alpha=0.85, edgecolor=STYLE["GRID"]),
    )

    ax.set_xlabel("Month and Year", color=STYLE["TEXT"])
    ax.set_ylabel(ylabel, color=STYLE["TEXT"])
    ax.set_title(title, color=STYLE["TEXT"], fontsize=14, fontweight="bold")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_xlim(dates.min() - pd.DateOffset(months=1), dates.max() + pd.DateOffset(months=1))
    ax.legend(loc="lower left", fontsize=9, facecolor=STYLE["BG"], edgecolor=STYLE["GRID"], labelcolor=STYLE["TEXT"])
    ax.grid(axis="y", linestyle="--", alpha=0.35)


# =====================================================================
# 15. FIGURE 09 -- MONTHLY MEAN DELTA_H
# =====================================================================


def fig_09_delta_h_timeseries(pdf_monthly: pd.DataFrame | None):
    """Monthly mean delta_h with linear trend and seasonal shading."""
    print("\n Building fig_09: delta_h Time Series ...")

    if pdf_monthly is None or pdf_monthly.empty:
        print(" [SKIP] No monthly data available.")
        return

    pdf = pdf_monthly.dropna(subset=["mean_delta_h"])
    if pdf.empty:
        print(" [SKIP] No valid delta_h data in monthly aggregates.")
        return

    fig, ax = new_fig(figsize=(14, 6))
    plot_timeseries_with_trend(
        ax=ax,
        dates=pdf["month_year"],
        values=pdf["mean_delta_h"],
        line_colour=STYLE["GREEN"],
        ylabel="Average Ice Height Change (delta_h, m)",
        title="Mean Antarctic Ice Height Change, by Month",
        annotation_y_factor=0.92,
    )
    fig.tight_layout()
    save_fig(fig, OUTPUT_DIR, "fig_09_delta_h_timeseries.png")


# =====================================================================
# 16. FIGURE 10 -- MONTHLY MEAN LWE
# =====================================================================


def fig_10_lwe_timeseries(pdf_monthly: pd.DataFrame | None):
    """Monthly mean LWE-like fused variable with linear trend and shading."""
    print("\n Building fig_10: LWE Fused Time Series ...")

    if pdf_monthly is None or pdf_monthly.empty:
        print(" [SKIP] No monthly data available.")
        return

    if "mean_lwe_fused" not in pdf_monthly.columns:
        print(" [SKIP] No monthly LWE aggregate available.")
        return

    pdf = pdf_monthly.dropna(subset=["mean_lwe_fused"])
    if pdf.empty:
        print(" [SKIP] No valid LWE data in monthly aggregates.")
        return

    fig, ax = new_fig(figsize=(14, 6))
    plot_timeseries_with_trend(
        ax=ax,
        dates=pdf["month_year"],
        values=pdf["mean_lwe_fused"],
        line_colour=STYLE["AMBER"],
        ylabel="Average LWE (cm eq. water thickness)",
        title="Mean Antarctic LWE (Fused), by Month",
        annotation_y_factor=1.05,
    )
    fig.tight_layout()
    save_fig(fig, OUTPUT_DIR, "fig_10_lwe_timeseries.png")


# =====================================================================
# 17. MAIN ENTRY POINT
# =====================================================================


def main() -> int:
    wall_start = _time.perf_counter()

    print(f" Output dir : {os.path.abspath(OUTPUT_DIR)}")
    print(f" Sample dir : {os.path.abspath(SAMPLE_DIR)}")

    try:
        spark = build_spark_session()
    except Exception as exc:
        print(f"[FATAL] Failed to create SparkSession: {exc}")
        return 2

    n_failures = 0

    try:
        print(f"\n{'=' * 60}\n STAGE 1: LOAD SAMPLE PARQUETS\n{'=' * 60}")
        samples = load_samples(spark)

        if not samples:
            print("[FATAL] No samples found. Run data_exploration.py first.")
            return 1

        print(f"\n{'=' * 60}\n STAGE 2: CORE EDA FIGURES\n{'=' * 60}")
        print(" Building fig_01: dataset overview ...")
        fig_dataset_overview(samples)

        print(" Building fig_02: completeness heatmap ...")
        fig_data_completeness(samples)

        print(" Building fig_03: histograms per dataset ...")
        for key, df in samples.items():
            colour = DATASET_COLOURS.get(key, STYLE["BLUE"])
            fig_histograms_for_dataset(df, key, colour)

        print(" Building fig_04: correlation heatmaps ...")
        for key, df in samples.items():
            fig_correlation_for_dataset(df, key)

        print(" Building fig_05: physical ranges ...")
        fig_physical_ranges(samples)

        print(" Building fig_06: null structure ...")
        fig_null_structure(samples)

        print(f"\n{'=' * 60}\n STAGE 3: BONUS SPATIAL FIGURES\n{'=' * 60}")
        try:
            fig_07_ice_mask_ocean_coverage()
        except Exception as exc:
            print(f" [ERROR] fig_07 failed: {exc}")
            n_failures += 1

        try:
            fig_08_delta_h_vs_lwe_spatial()
        except Exception as exc:
            print(f" [ERROR] fig_08 failed: {exc}")
            n_failures += 1

        print(f"\n{'=' * 60}\n STAGE 4: MONTHLY AGGREGATION + TIME SERIES\n{'=' * 60}")
        pdf_monthly = None
        try:
            pdf_monthly = compute_monthly_aggregates(spark)
        except Exception as exc:
            print(f" [ERROR] Monthly aggregation failed: {exc}")
            n_failures += 1

        try:
            fig_09_delta_h_timeseries(pdf_monthly)
        except Exception as exc:
            print(f" [ERROR] fig_09 failed: {exc}")
            n_failures += 1

        try:
            fig_10_lwe_timeseries(pdf_monthly)
        except Exception as exc:
            print(f" [ERROR] fig_10 failed: {exc}")
            n_failures += 1

    finally:
        spark.stop()

    total_wall = _time.perf_counter() - wall_start

    if n_failures > 0:
        print(f"[WARNING] {n_failures} figure block(s) encountered errors.")

    print(f"\n PLOTS-ONLY PIPELINE COMPLETE")
    print(f" Total time : {total_wall:.1f}s ({total_wall / 60:.1f} min)")
    print(f" Plots dir : {os.path.abspath(OUTPUT_DIR)}")

    return 0 if n_failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
