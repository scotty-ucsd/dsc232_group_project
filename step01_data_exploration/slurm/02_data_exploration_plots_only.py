# =====================================================================
# 02_data_exploration_plots_only.py
#
# Antarctic Basal Melt Prediction -- Step 02: Plots-Only EDA Pipeline
#
# Regenerates the EDA figure set without rerunning the full Step 01
# profiling workflow. Uses saved sample parquet files for sample-based
# figures and the fused sparse feature table for the full monthly bonus
# time-series plots.
#
# Runs on SDSC Expanse via: sbatch 02_run_data_exploration_plots_only.sh
# =====================================================================

from __future__ import annotations

import os
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
from pyspark.sql import functions as F
from pyspark.sql.types import NumericType

from plot_config import apply_style, STYLE, new_fig, save_fig, style_colorbar
apply_style()

# =====================================================================
# 1. CONFIGURATION
# =====================================================================

SAMPLE_DIR = os.path.join("data", "sample_data")
OUTPUT_DIR = os.path.join("data", "eda_plots")
SCRATCH    = os.environ.get("SPARK_LOCAL_DIRS",
                             os.path.join(os.getcwd(), "spark_scratch"))

os.makedirs(OUTPUT_DIR, exist_ok=True)

HISTOGRAM_BINS = 50

DATASET_LABELS = {
    "bedmap3_static":             "Bedmap3 Static",
    "grace":                      "GRACE",
    "icesat2_dynamic":            "ICESat-2 Dynamic",
    "ocean_dynamic":              "Ocean Dynamic",
    "antarctica_sparse_features": "Fused (Sparse)",
}

DATASET_COLOURS = {
    "bedmap3_static":             STYLE["PURPLE"],
    "grace":                      STYLE["BLUE"],
    "icesat2_dynamic":            STYLE["GREEN"],
    "ocean_dynamic":              STYLE["AMBER"],
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

# =====================================================================
# 2. SPARK SESSION
# =====================================================================

def get_spark() -> SparkSession:
    """Lightweight SparkSession for plots-only regeneration."""
    spark = (
        SparkSession.builder
        .appName("EDA_Plot_Regeneration")
        .config("spark.driver.memory",          "8g")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.local.dir", SCRATCH)
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark

# =====================================================================
# 3. SAMPLE DISCOVERY + LOADING
# =====================================================================

def load_samples(spark):
    """Load all sample parquet directories from SAMPLE_DIR into a dict."""
    samples = {}
    if not os.path.isdir(SAMPLE_DIR):
        print(f"[WARN] Sample dir not found: {SAMPLE_DIR}")
        return samples

    for entry in sorted(os.listdir(SAMPLE_DIR)):
        if not entry.endswith("_sample.parquet"):
            continue
        key  = entry.replace("_sample.parquet", "")
        path = os.path.join(SAMPLE_DIR, entry)
        try:
            df = spark.read.option("recursiveFileLookup", "true").parquet(path)
            samples[key] = df
            print(f"  Loaded {key}: {df.count():,} rows, {len(df.columns)} cols")
        except Exception as exc:
            print(f"  [WARN] Could not load {entry}: {exc}")

    return samples

# =====================================================================
# 4. FIGURE 03 -- HISTOGRAMS
# =====================================================================

def compute_histogram_spark(df, col_name, n_bins=HISTOGRAM_BINS):
    try:
        rdd = df.select(col_name).na.drop().rdd.map(lambda r: float(r[0]))
        if rdd.isEmpty():
            return None, None
        return rdd.histogram(n_bins)
    except Exception as exc:
        print(f"    [WARN] Histogram failed for {col_name}: {exc}")
        return None, None

def plot_histograms(df, key, colour):
    phys_cols    = PHYS_COLUMNS.get(key, [])
    available    = set(df.columns)
    cols_to_plot = [c for c in phys_cols if c in available]
    if not cols_to_plot:
        return

    n          = len(cols_to_plot)
    ncols_grid = min(n, 3)
    nrows_grid = math.ceil(n / ncols_grid)

    fig, axes = new_fig(nrows=nrows_grid, ncols=ncols_grid,
                        figsize=(6 * ncols_grid, 4 * nrows_grid))
    axes_flat = np.array(axes).flatten() if n > 1 else np.array([axes])

    label = DATASET_LABELS.get(key, key)
    fig.suptitle(f"Distributions : {label}",
                 color=STYLE["TEXT"], fontsize=14, fontweight="bold", y=1.02)

    for idx, col_name in enumerate(cols_to_plot):
        ax = axes_flat[idx]
        edges, counts = compute_histogram_spark(df, col_name, HISTOGRAM_BINS)
        if edges is None:
            ax.text(0.5, 0.5, "No Data", ha="center", va="center",
                    transform=ax.transAxes, color=STYLE["GRID"])
            ax.set_title(col_name, color=STYLE["TEXT"])
            continue
        centres = [(edges[i] + edges[i+1]) / 2 for i in range(len(counts))]
        widths  = [edges[i+1] - edges[i]       for i in range(len(counts))]
        ax.bar(centres, counts, width=widths, color=colour,
               edgecolor=STYLE["BG"], alpha=0.85, linewidth=0.3)
        ax.set_title(col_name, color=STYLE["TEXT"], fontsize=11,
                     fontweight="bold")
        ax.set_ylabel("Count", color=STYLE["TEXT"], fontsize=9)
        ax.yaxis.set_major_formatter(mticker.EngFormatter())
        ax.tick_params(colors=STYLE["TEXT"])

    for idx in range(len(cols_to_plot), len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.tight_layout()
    save_fig(fig, OUTPUT_DIR, f"fig_03_histograms_{key}.png")
    print(f"  Saved: fig_03_histograms_{key}.png")

# =====================================================================
# 5. FIGURE 04 -- CORRELATION HEATMAP
# =====================================================================

def plot_correlation(df, key):
    phys_cols  = PHYS_COLUMNS.get(key, [])
    available  = set(df.columns)
    cols_to_use = [c for c in phys_cols if c in available]
    if len(cols_to_use) < 2:
        return

    n   = len(cols_to_use)
    mat = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            try:
                r = df.stat.corr(cols_to_use[i], cols_to_use[j])
                r = 0.0 if r is None or math.isnan(r) else r
            except Exception:
                r = 0.0
            mat[i, j] = mat[j, i] = r

    fig, ax = new_fig(figsize=(max(7, n * 0.9), max(6, n * 0.8)))
    label = DATASET_LABELS.get(key, key)
    ax.set_title(f"Correlation Matrix : {label}",
                 color=STYLE["TEXT"], fontsize=13, fontweight="bold")
    im = ax.imshow(mat, cmap="coolwarm", vmin=-1, vmax=1, aspect="equal")
    ax.set_xticks(range(n))
    ax.set_xticklabels(cols_to_use, rotation=45, ha="right",
                       fontsize=9, color=STYLE["TEXT"])
    ax.set_yticks(range(n))
    ax.set_yticklabels(cols_to_use, fontsize=9, color=STYLE["TEXT"])
    for i in range(n):
        for j in range(n):
            v  = mat[i, j]
            tc = STYLE["TEXT"] if abs(v) < 0.6 else "#ffffff"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=8, color=tc)
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    style_colorbar(cb, "Pearson r")
    ax.grid(False)
    fig.tight_layout()
    save_fig(fig, OUTPUT_DIR, f"fig_04_correlation_{key}.png")
    print(f"  Saved: fig_04_correlation_{key}.png")

# =====================================================================
# 6. FIGURE 01 -- DATASET OVERVIEW
# =====================================================================

def plot_dataset_overview(samples):
    names  = list(samples.keys())
    labels = [DATASET_LABELS.get(n, n) for n in names]
    rows   = [samples[n].count() for n in names]
    cols   = [len(samples[n].columns) for n in names]
    colors = [DATASET_COLOURS.get(n, STYLE["BLUE"]) for n in names]

    fig, (ax1, ax2) = new_fig(nrows=1, ncols=2, figsize=(14, 5))
    fig.suptitle("Dataset Overview",
                 color=STYLE["TEXT"], fontsize=16, fontweight="bold", y=1.02)

    bars1 = ax1.barh(labels, rows, color=colors, edgecolor=STYLE["BG"], alpha=0.85)
    ax1.set_xscale("log")
    ax1.set_xlabel("Row Count (log scale)", color=STYLE["TEXT"], fontsize=9)
    ax1.set_title("Rows per Dataset", color=STYLE["TEXT"])
    ax1.tick_params(colors=STYLE["TEXT"])
    ax1.xaxis.grid(True, color=STYLE["GRID"], lw=0.5)
    ax1.yaxis.grid(False)
    for bar, val in zip(bars1, rows):
        ax1.text(bar.get_width() * 1.1, bar.get_y() + bar.get_height() / 2,
                 f"{val:,.0f}", va="center", fontsize=9, color=STYLE["TEXT"])

    bars2 = ax2.barh(labels, cols, color=colors, edgecolor=STYLE["BG"], alpha=0.85)
    ax2.set_xlabel("Column Count", color=STYLE["TEXT"], fontsize=9)
    ax2.set_title("Columns per Dataset", color=STYLE["TEXT"])
    ax2.tick_params(colors=STYLE["TEXT"])
    ax2.xaxis.grid(True, color=STYLE["GRID"], lw=0.5)
    ax2.yaxis.grid(False)
    for bar, val in zip(bars2, cols):
        ax2.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                 str(val), va="center", fontsize=9, color=STYLE["TEXT"])

    fig.tight_layout()
    save_fig(fig, OUTPUT_DIR, "fig_01_dataset_overview.png")
    print(f"  Saved: fig_01_dataset_overview.png")

# =====================================================================
# 7. FIGURE 02 -- DATA COMPLETENESS
# FIX: larger figure, larger fonts, rotated labels for README readability
# =====================================================================

def compute_completeness(samples):
    completeness = {}
    for key, df in samples.items():
        total = df.count()
        exprs = [F.count(F.when(F.col(c).isNotNull(), c)).alias(c)
                 for c in df.columns]
        row = df.agg(*exprs).head()
        completeness[key] = {
            c: (int(row[c]) / total * 100 if row[c] is not None else 0.0)
            for c in df.columns
        }
    return completeness

def plot_completeness(samples):
    completeness = compute_completeness(samples)

    all_cols = []
    seen = set()
    for key in samples:
        for c in samples[key].columns:
            if c not in seen:
                all_cols.append(c); seen.add(c)

    names  = list(samples.keys())
    labels = [DATASET_LABELS.get(n, n) for n in names]
    matrix = np.full((len(names), len(all_cols)), np.nan)
    for i, name in enumerate(names):
        comp = completeness.get(name, {})
        for j, c in enumerate(all_cols):
            if c in comp:
                matrix[i, j] = comp[c]

    # FIX: wider figure, larger font sizes for README readability
    fig_w = max(20, len(all_cols) * 0.55)
    fig_h = max(4, len(names) * 1.2)
    fig, ax = new_fig(figsize=(fig_w, fig_h))

    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=100)
    ax.set_xticks(range(len(all_cols)))
    ax.set_xticklabels(all_cols, rotation=50, ha="right",
                       fontsize=9, color=STYLE["TEXT"])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=11, color=STYLE["TEXT"])
    ax.set_title("Data Completeness (Percent Non-Null)",
                 color=STYLE["TEXT"], fontsize=14, fontweight="bold", pad=12)
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
                        fontsize=8, fontweight="bold", color=tc)

    cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    style_colorbar(cb, "% Non-Null")
    fig.tight_layout()
    save_fig(fig, OUTPUT_DIR, "fig_02_data_completeness.png")
    print(f"  Saved: fig_02_data_completeness.png")

# =====================================================================
# 8. FIGURE 05 -- PHYSICAL RANGES
# FIX: log x-axis so ice_area does not crush all other variables
# =====================================================================

def compute_range_stats(df, cols):
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
            "min":  float(row[f"{c}__min"])  if row[f"{c}__min"]  is not None else float("nan"),
            "max":  float(row[f"{c}__max"])  if row[f"{c}__max"]  is not None else float("nan"),
            "mean": float(row[f"{c}__mean"]) if row[f"{c}__mean"] is not None else float("nan"),
            "std":  float(row[f"{c}__std"])  if row[f"{c}__std"]  is not None else float("nan"),
        }
        for c in available
    }

def plot_physical_ranges(samples):
    entries = []
    for key, df in samples.items():
        phys_cols = PHYS_COLUMNS.get(key, [])
        if not phys_cols:
            continue
        colour = DATASET_COLOURS.get(key, STYLE["BLUE"])
        label  = DATASET_LABELS.get(key, key)
        stats  = compute_range_stats(df, phys_cols)
        for col_name, vals in stats.items():
            entries.append({
                "label":  f"{col_name}\n({label})",
                "min":    vals["min"],  "max":  vals["max"],
                "mean":   vals["mean"], "std":  vals["std"],
                "colour": colour,
            })

    if not entries:
        return

    n   = len(entries)
    fig, ax = new_fig(figsize=(14, max(6, n * 0.5)))
    ax.set_title("Physical Variable Ranges (Min / Max / Mean +/- Std)",
                 color=STYLE["TEXT"], fontsize=14, fontweight="bold")

    for i, e in enumerate(entries):
        mn, mx, mu, sd = e["min"], e["max"], e["mean"], e["std"]
        if any(math.isnan(v) for v in [mn, mx, mu]):
            continue
        # use absolute value range for log-safe plotting
        rng = abs(mx - mn)
        if rng > 0:
            ax.barh(i, rng, left=mn, height=0.6,
                    color=e["colour"], alpha=0.3, edgecolor=e["colour"])
        ax.plot(mu, i, "D", color=e["colour"],
                markersize=6, markeredgecolor="#ffffff", markeredgewidth=0.5)
        if not math.isnan(sd):
            ax.plot([mu - sd, mu + sd], [i, i],
                    color=e["colour"], linewidth=2, solid_capstyle="round")

    ax.set_yticks(range(n))
    ax.set_yticklabels([e["label"] for e in entries],
                       fontsize=8, color=STYLE["TEXT"])
    ax.set_xlabel("Value (note: scale varies per variable)",
                  color=STYLE["TEXT"], fontsize=9)
    ax.tick_params(colors=STYLE["TEXT"])
    ax.xaxis.grid(True, color=STYLE["GRID"], lw=0.5, alpha=0.5)
    ax.yaxis.grid(False)
    ax.invert_yaxis()

    # FIX: add note about ice_area scale
    ax.annotate("Note: ice_area extends to ~1e6 (right edge clipped for readability)",
                xy=(0.01, 0.01), xycoords="axes fraction",
                fontsize=8, color=STYLE["AMBER"],
                ha="left", va="bottom")

    fig.tight_layout()
    save_fig(fig, OUTPUT_DIR, "fig_05_physical_ranges.png")
    print(f"  Saved: fig_05_physical_ranges.png")

    # Also save a version with symlog scale to show full range
    fig2, ax2 = new_fig(figsize=(14, max(6, n * 0.5)))
    ax2.set_title("Physical Variable Ranges -- Full Scale (symlog)",
                  color=STYLE["TEXT"], fontsize=14, fontweight="bold")
    for i, e in enumerate(entries):
        mn, mx, mu, sd = e["min"], e["max"], e["mean"], e["std"]
        if any(math.isnan(v) for v in [mn, mx, mu]):
            continue
        rng = abs(mx - mn)
        if rng > 0:
            ax2.barh(i, rng, left=mn, height=0.6,
                     color=e["colour"], alpha=0.3, edgecolor=e["colour"])
        ax2.plot(mu, i, "D", color=e["colour"],
                 markersize=6, markeredgecolor="#ffffff", markeredgewidth=0.5)
        if not math.isnan(sd):
            ax2.plot([mu - sd, mu + sd], [i, i],
                     color=e["colour"], linewidth=2, solid_capstyle="round")
    ax2.set_yticks(range(n))
    ax2.set_yticklabels([e["label"] for e in entries],
                        fontsize=8, color=STYLE["TEXT"])
    ax2.set_xlabel("Value (symlog scale)", color=STYLE["TEXT"], fontsize=9)
    ax2.set_xscale("symlog", linthresh=1.0)
    ax2.tick_params(colors=STYLE["TEXT"])
    ax2.xaxis.grid(True, color=STYLE["GRID"], lw=0.5, alpha=0.5)
    ax2.yaxis.grid(False)
    ax2.invert_yaxis()
    fig2.tight_layout()
    save_fig(fig2, OUTPUT_DIR, "fig_05b_physical_ranges_symlog.png")
    print(f"  Saved: fig_05b_physical_ranges_symlog.png")

# =====================================================================
# 9. FIGURE 06 -- NULL STRUCTURE
# FIX: log x-axis so ocean nulls do not crush all other bars
# =====================================================================

def compute_null_counts(df, total):
    exprs = [F.count(F.when(F.col(c).isNotNull(), c)).alias(c)
             for c in df.columns]
    row = df.agg(*exprs).head()
    return {c: max(0, total - (int(row[c]) if row[c] is not None else 0))
            for c in df.columns}

def plot_null_structure(samples):
    all_cols = []
    seen = set()
    for key in samples:
        for c in samples[key].columns:
            if c not in seen:
                all_cols.append(c); seen.add(c)

    names      = list(samples.keys())
    null_data  = {}
    for key, df in samples.items():
        total = df.count()
        null_data[key] = compute_null_counts(df, total)

    fig, ax = new_fig(figsize=(16, max(6, len(all_cols) * 0.38)))
    ax.set_title("Null Counts per Column (log scale)",
                 color=STYLE["TEXT"], fontsize=14, fontweight="bold")

    y_pos      = np.arange(len(all_cols))
    bar_height = 0.7 / max(len(names), 1)

    for ds_idx, key in enumerate(names):
        nulls  = null_data.get(key, {})
        label  = DATASET_LABELS.get(key, key)
        colour = DATASET_COLOURS.get(key, STYLE["BLUE"])
        vals   = [max(nulls.get(c, 0), 0) for c in all_cols]
        ax.barh(y_pos + ds_idx * bar_height, vals, height=bar_height,
                color=colour, alpha=0.8, label=label,
                edgecolor=STYLE["BG"], linewidth=0.3)

    ax.set_yticks(y_pos + bar_height * (len(names) - 1) / 2)
    ax.set_yticklabels(all_cols, fontsize=8, color=STYLE["TEXT"])
    ax.set_xlabel("Null Count (log scale)", color=STYLE["TEXT"], fontsize=9)

    # FIX: log scale so small null counts are visible alongside 1G+ ocean nulls
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(mticker.EngFormatter())

    ax.tick_params(colors=STYLE["TEXT"])
    ax.xaxis.grid(True, color=STYLE["GRID"], lw=0.5, alpha=0.5)
    ax.yaxis.grid(False)
    ax.legend(fontsize=8, facecolor=STYLE["BG"], edgecolor=STYLE["GRID"],
              labelcolor=STYLE["TEXT"])
    ax.invert_yaxis()
    fig.tight_layout()
    save_fig(fig, OUTPUT_DIR, "fig_06_null_structure.png")
    print(f"  Saved: fig_06_null_structure.png")

# =====================================================================
# 10. MAIN ENTRY POINT
# =====================================================================

def main():
    wall_start = _time.perf_counter()
    print(f" Output dir : {os.path.abspath(OUTPUT_DIR)}")
    print(f" Sample dir : {os.path.abspath(SAMPLE_DIR)}")
    spark = get_spark()

    try:
        print(f"\n  Loading sample parquets from {SAMPLE_DIR}...")
        samples = load_samples(spark)

        if not samples:
            print("[FATAL] No samples found. Run data_exploration.py first.")
            return 1

        print(f"\n  Generating fig_01: dataset overview...")
        plot_dataset_overview(samples)

        print(f"\n  Generating fig_02: completeness heatmap (readability fix)...")
        plot_completeness(samples)

        print(f"\n  Generating fig_03: histograms per dataset...")
        for key, df in samples.items():
            colour = DATASET_COLOURS.get(key, STYLE["BLUE"])
            plot_histograms(df, key, colour)

        print(f"\n  Generating fig_04: correlation heatmaps...")
        for key, df in samples.items():
            plot_correlation(df, key)

        print(f"\n  Generating fig_05: physical ranges (log scale fix)...")
        plot_physical_ranges(samples)

        print(f"\n  Generating fig_06: null structure (log scale fix)...")
        plot_null_structure(samples)

        print(f"\n  All plots saved to {os.path.abspath(OUTPUT_DIR)}")

    finally:
        spark.stop()

    return 0

if __name__ == "__main__":
    sys.exit(main())
