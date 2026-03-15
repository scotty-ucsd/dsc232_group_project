# =====================================================================
# plot_config.py — Centralized plotting style for Antarctic Basal Melt
#
# Style: "Deep Field"
#   White background, deep navy + coral accents, clean gridlines.
#   All pipeline scripts import from here — change once, applies
#   everywhere across step01, step02, step03.
#
# Usage:
#   from plot_config import STYLE, apply_style, new_fig, save_fig
#
# =====================================================================

from __future__ import annotations

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from typing import Optional

# =====================================================================
# 1. COLOUR PALETTE  (Style 2C — Deep Field)
# =====================================================================

STYLE = dict(
    # ── backgrounds ──────────────────────────────────────────────────
    BG      = "#ffffff",   # figure background
    PANEL   = "#f7f8fc",   # axes background
    GRID    = "#e2e6ef",   # gridline / spine colour

    # ── text ─────────────────────────────────────────────────────────
    TEXT    = "#0d1b2a",   # titles, labels, tick labels

    # ── primary accents ───────────────────────────────────────────────
    BLUE    = "#1a3a6b",   # primary bars, histograms, SVD components
    GREEN   = "#1a6b5a",   # secondary / cumulative lines, TP
    RED     = "#c0392b",   # errors, FN, Amundsen highlight
    AMBER   = "#d4780a",   # warning / FP / baseline reference lines
    PURPLE  = "#4a2575",   # tertiary accent / extra cluster colour

    # ── geo error scatter colours ─────────────────────────────────────
    GEO_TP  = "#1a6b5a",   # True Positive  — green
    GEO_TN  = "#c8d0dc",   # True Negative  — light grey
    GEO_FP  = "#d4780a",   # False Positive — amber
    GEO_FN  = "#c0392b",   # False Negative — red

    # ── colormaps ─────────────────────────────────────────────────────
    CMAP_GEO  = "RdBu",    # hexbin geographic maps (delta_h)
    CMAP_CM   = "Blues",   # confusion matrix

    # ── cluster palette (8 colours for KMeans k≤8) ───────────────────
    CLUSTERS  = [
        "#1a3a6b",  # 0 — navy
        "#4a2575",  # 1 — purple
        "#1a6b5a",  # 2 — teal
        "#d4780a",  # 3 — amber
        "#c0392b",  # 4 — red
        "#0d6b6b",  # 5 — dark teal
        "#2d6b1a",  # 6 — forest
        "#6b1a3a",  # 7 — burgundy
    ],
)

# =====================================================================
# 2. MATPLOTLIB RC DEFAULTS
#    Call apply_style() once at the top of each notebook / script.
# =====================================================================

def apply_style():
    """
    Apply the Deep Field rcParams globally.
    Call once at the top of any script or notebook cell.

    Example:
        from plot_config import apply_style
        apply_style()
    """
    plt.rcParams.update({
        # figure
        "figure.facecolor":      STYLE["BG"],
        "figure.dpi":            150,
        "figure.titlesize":      13,
        "figure.titleweight":    "bold",

        # axes
        "axes.facecolor":        STYLE["PANEL"],
        "axes.edgecolor":        STYLE["GRID"],
        "axes.labelcolor":       STYLE["TEXT"],
        "axes.labelsize":        9,
        "axes.titlesize":        11,
        "axes.titleweight":      "bold",
        "axes.titlepad":         8,
        "axes.spines.top":       False,
        "axes.spines.right":     False,
        "axes.grid":             True,
        "axes.grid.axis":        "y",
        "grid.color":            STYLE["GRID"],
        "grid.linewidth":        0.6,
        "grid.alpha":            0.7,

        # ticks
        "xtick.color":           STYLE["TEXT"],
        "ytick.color":           STYLE["TEXT"],
        "xtick.labelsize":       8,
        "ytick.labelsize":       8,
        "xtick.direction":       "out",
        "ytick.direction":       "out",

        # lines / markers
        "lines.linewidth":       2.0,
        "lines.markersize":      5,

        # legend
        "legend.framealpha":     0.92,
        "legend.edgecolor":      STYLE["GRID"],
        "legend.fontsize":       8,
        "legend.title_fontsize": 8,

        # colorbar
        "figure.constrained_layout.use": False,

        # savefig
        "savefig.facecolor":     STYLE["BG"],
        "savefig.bbox":          "tight",
        "savefig.dpi":           150,
    })


# =====================================================================
# 3. CONVENIENCE HELPERS
# =====================================================================

def new_fig(nrows=1, ncols=1, figsize=None, **kwargs):
    """
    Create a pre-styled figure + axes with Deep Field colours applied.

    Returns (fig, axes) — axes is a single Axes if nrows==ncols==1,
    otherwise a numpy array matching (nrows, ncols).
    """
    if figsize is None:
        w = ncols * 6
        h = nrows * 5
        figsize = (w, h)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                             facecolor=STYLE["BG"], **kwargs)

    # Apply panel style to every axes
    ax_flat = np.array(axes).flatten() if nrows * ncols > 1 else [axes]
    for ax in ax_flat:
        ax.set_facecolor(STYLE["PANEL"])
        for sp in ax.spines.values():
            sp.set_color(STYLE["GRID"])
            sp.set_linewidth(1.0)
        ax.tick_params(colors=STYLE["TEXT"])
        ax.yaxis.grid(True, color=STYLE["GRID"], lw=0.6, alpha=0.7)
        ax.set_axisbelow(True)

    return fig, axes


def save_fig(fig, output_dir: str, filename: str):
    """Save figure to output_dir/filename.png"""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename if filename.endswith(".png") else filename + ".png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=STYLE["BG"])
    plt.close(fig)
    print(f"  [plot] -> {path}")
    return path


def style_colorbar(cb, label: Optional[str] = None):
    """Apply Deep Field style to a colorbar."""
    cb.ax.tick_params(labelcolor=STYLE["TEXT"], labelsize=7)
    if label:
        cb.set_label(label, color=STYLE["TEXT"], fontsize=8)


def style_legend(ax, **kwargs):
    """Apply Deep Field style to an axes legend."""
    leg = ax.legend(
        facecolor=STYLE["BG"],
        edgecolor=STYLE["GRID"],
        labelcolor=STYLE["TEXT"],
        framealpha=0.92,
        fontsize=8,
        **kwargs,
    )
    return leg


# =====================================================================
# 4. REUSABLE PLOT FUNCTIONS
#    Drop-in replacements for the ad-hoc versions in each pipeline script.
# =====================================================================

def plot_histogram(counts, edges, col_name, output_dir, title=None, log_scale=True):
    """
    Spark-aggregated histogram (matches plot_spark_dist in data_exploration.ipynb).

    Args:
        counts : numpy array of bin counts (from Spark groupBy agg)
        edges  : numpy array of bin edges (len = len(counts) + 1)
        col_name: feature name for axis label
        output_dir: where to save
        title  : override title (defaults to col_name)
        log_scale: y-axis log scale (default True)
    """
    fig, ax = new_fig(figsize=(10, 4))
    ax.bar(edges[:-1], counts, width=np.diff(edges),
           align="edge", color=STYLE["BLUE"], alpha=0.82,
           edgecolor=STYLE["BG"], linewidth=0.3)
    ax.fill_between(edges[:-1], counts, alpha=0.08,
                    color=STYLE["BLUE"], step="post")
    ax.axvline(0, color=STYLE["RED"], lw=1.5, ls="--", alpha=0.75, label="Zero")
    if log_scale:
        ax.set_yscale("log")
    ax.set_title(title or f"Distribution: {col_name}", color=STYLE["TEXT"])
    ax.set_xlabel(col_name, color=STYLE["TEXT"], fontsize=9)
    ax.set_ylabel("Count (log)" if log_scale else "Count",
                  color=STYLE["TEXT"], fontsize=9)
    ax.grid(True, alpha=0.3, color=STYLE["GRID"], axis="y")
    return save_fig(fig, output_dir, f"hist_{col_name}.png")


def plot_hexbin_geo(x, y, c, output_dir, title="ICESat-2 Δh",
                   cbar_label="Δh (m)", gridsize=150,
                   vmin=-5, vmax=5, filename=None):
    """
    Geographic hexbin map (matches step01 spatial check + winter comparison).
    x, y, c are pandas Series or numpy arrays (sampled from Spark).
    """
    fig, ax = new_fig(figsize=(10, 8))
    hb = ax.hexbin(x, y, C=c, gridsize=gridsize,
                   cmap=STYLE["CMAP_GEO"], reduce_C_function=np.mean,
                   vmin=vmin, vmax=vmax)
    cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.02)
    style_colorbar(cb, cbar_label)
    ax.set_title(title, color=STYLE["TEXT"])
    ax.set_xlabel("EPSG:3031 X (m)", color=STYLE["TEXT"], fontsize=9)
    ax.set_ylabel("EPSG:3031 Y (m)", color=STYLE["TEXT"], fontsize=9)
    ax.set_aspect("equal")
    ax.grid(False)  # turn off grid for geographic plots
    fname = filename or f"hexbin_{title.replace(' ','_').lower()}.png"
    return save_fig(fig, output_dir, fname)


def plot_hexbin_comparison(x1, y1, c1, x2, y2, c2,
                           title1, title2, suptitle,
                           output_dir, filename=None,
                           gridsize=150, vmin=-5, vmax=5):
    """
    Two-panel geographic hexbin comparison (matches step01 winter 2019 vs 2024).
    """
    fig, axes = new_fig(nrows=2, ncols=1, figsize=(10, 16))
    for ax, x, y, c, t in [(axes[0], x1, y1, c1, title1),
                             (axes[1], x2, y2, c2, title2)]:
        hb = ax.hexbin(x, y, C=c, gridsize=gridsize,
                       cmap=STYLE["CMAP_GEO"], reduce_C_function=np.mean,
                       vmin=vmin, vmax=vmax)
        ax.set_title(t, color=STYLE["TEXT"], fontsize=18, fontweight="bold", pad=15)
        ax.set_aspect("equal")
        ax.axis("off")

    # shared colorbar
    cb = fig.colorbar(hb, ax=axes, orientation="vertical",
                      fraction=0.05, pad=0.02)
    style_colorbar(cb, "Surface Height Change (m)")
    cb.ax.tick_params(labelsize=12)

    fig.suptitle(suptitle, color=STYLE["TEXT"], fontsize=22,
                 fontweight="bold")
    fname = filename or "hexbin_comparison.png"
    return save_fig(fig, output_dir, fname)


def plot_geographic_errors(preds_pdf, model_name, output_dir):
    """
    Geographic error scatter: TP/FP/FN/TN coloured by Deep Field palette.
    preds_pdf: pandas DataFrame with columns x, y, prediction, label.

    Replaces plot_geographic_errors in pipeline_config.py.
    """
    from pipeline_config import LABEL_COL, PREDICTION_COL

    if preds_pdf.empty:
        print(f"  [{model_name}] Empty predictions, skipping geographic plot.")
        return

    preds_pdf = preds_pdf.copy()
    preds_pdf["error_type"] = "TN"
    preds_pdf.loc[(preds_pdf[LABEL_COL]==1)&(preds_pdf[PREDICTION_COL]==1), "error_type"] = "TP"
    preds_pdf.loc[(preds_pdf[LABEL_COL]==0)&(preds_pdf[PREDICTION_COL]==1), "error_type"] = "FP"
    preds_pdf.loc[(preds_pdf[LABEL_COL]==1)&(preds_pdf[PREDICTION_COL]==0), "error_type"] = "FN"

    pal = {"TP": STYLE["GEO_TP"], "TN": STYLE["GEO_TN"],
           "FP": STYLE["GEO_FP"], "FN": STYLE["GEO_FN"]}

    # Plot 1: all predictions
    fig, ax = new_fig(figsize=(10, 8))
    for et, c in pal.items():
        sub = preds_pdf[preds_pdf["error_type"] == et]
        if not sub.empty:
            ax.scatter(sub["x"], sub["y"], c=c, label=et,
                       s=4 if et=="TN" else 8,
                       alpha=0.3 if et=="TN" else 0.85,
                       rasterized=True, edgecolors="none")
    ax.set_title(f"{model_name}: Geographic Errors (EPSG:3031)",
                 color=STYLE["TEXT"])
    ax.set_xlabel("x (m)", color=STYLE["TEXT"], fontsize=9)
    ax.set_ylabel("y (m)", color=STYLE["TEXT"], fontsize=9)
    ax.grid(False)
    style_legend(ax, markerscale=2)
    save_fig(fig, output_dir, f"{model_name}_geo_errors.png")

    # Plot 2: regional error rates bar chart
    if "regional_subset_id" in preds_pdf.columns:
        from pipeline_config import LABEL_COL, PREDICTION_COL
        regions = sorted(preds_pdf["regional_subset_id"].unique())
        fnrs, fprs = [], []
        for r in regions:
            g = preds_pdf[preds_pdf["regional_subset_id"] == r]
            fnrs.append((g["error_type"]=="FN").sum() / max(1,(g[LABEL_COL]==1).sum()))
            fprs.append((g["error_type"]=="FP").sum() / max(1,(g[LABEL_COL]==0).sum()))

        fig, ax = new_fig(figsize=(10, 5))
        x_pos = np.arange(len(regions))
        w = 0.35
        ax.bar(x_pos - w/2, fnrs, w, label="FNR", color=STYLE["RED"],   alpha=0.85)
        ax.bar(x_pos + w/2, fprs, w, label="FPR", color=STYLE["AMBER"], alpha=0.85)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(regions, rotation=35, ha="right",
                           color=STYLE["TEXT"], fontsize=8)
        ax.set_title(f"{model_name}: Regional Error Rates", color=STYLE["TEXT"])
        ax.set_ylabel("Rate", color=STYLE["TEXT"], fontsize=9)
        style_legend(ax)
        save_fig(fig, output_dir, f"{model_name}_regional_errors.png")

    # Plot 3: errors only
    errs = preds_pdf[preds_pdf["error_type"].isin(["FP","FN"])]
    if not errs.empty:
        fig, ax = new_fig(figsize=(10, 8))
        for et in ["FN","FP"]:
            sub = errs[errs["error_type"]==et]
            if not sub.empty:
                ax.scatter(sub["x"], sub["y"], c=pal[et], label=et,
                           alpha=0.85, s=5, rasterized=True, edgecolors="none")
        ax.set_title(f"{model_name}: Misclassified Only", color=STYLE["TEXT"])
        ax.set_xlabel("x (m)", color=STYLE["TEXT"], fontsize=9)
        ax.set_ylabel("y (m)", color=STYLE["TEXT"], fontsize=9)
        ax.grid(False)
        style_legend(ax, markerscale=2)
        save_fig(fig, output_dir, f"{model_name}_errors_only.png")


def plot_temporal_residuals(preds_pdf, model_name, output_dir):
    """
    Temporal FNR/FPR line plot.
    preds_pdf: pandas DataFrame aggregated by month_idx.

    Replaces plot_temporal_residuals in pipeline_config.py.
    """
    if preds_pdf.empty or "month_idx" not in preds_pdf.columns:
        return

    from pipeline_config import LABEL_COL, PREDICTION_COL
    grp = (preds_pdf.groupby("month_idx")
           .apply(lambda g: {
               "fnr": ((g[LABEL_COL]==1)&(g[PREDICTION_COL]==0)).sum() /
                       max(1,(g[LABEL_COL]==1).sum()),
               "fpr": ((g[LABEL_COL]==0)&(g[PREDICTION_COL]==1)).sum() /
                       max(1,(g[LABEL_COL]==0).sum()),
           })
           .apply(lambda d: d)
    )

    months = preds_pdf["month_idx"].sort_values().unique()
    fnr = [((preds_pdf[preds_pdf["month_idx"]==m][LABEL_COL]==1) &
             (preds_pdf[preds_pdf["month_idx"]==m][PREDICTION_COL]==0)).sum() /
            max(1,(preds_pdf[preds_pdf["month_idx"]==m][LABEL_COL]==1).sum())
           for m in months]
    fpr = [((preds_pdf[preds_pdf["month_idx"]==m][LABEL_COL]==0) &
             (preds_pdf[preds_pdf["month_idx"]==m][PREDICTION_COL]==1)).sum() /
            max(1,(preds_pdf[preds_pdf["month_idx"]==m][LABEL_COL]==0).sum())
           for m in months]

    fig, ax = new_fig(figsize=(10, 5))
    ax.fill_between(months, fnr, alpha=0.12, color=STYLE["RED"])
    ax.fill_between(months, fpr, alpha=0.08, color=STYLE["AMBER"])
    ax.plot(months, fnr, color=STYLE["RED"],   lw=2, marker="o", ms=3,
            label="FNR (miss rate)")
    ax.plot(months, fpr, color=STYLE["AMBER"], lw=2, marker="o", ms=3,
            label="FPR (false alarm)")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"{model_name}: Temporal Error Rates by Month",
                 color=STYLE["TEXT"])
    ax.set_xlabel("month_idx", color=STYLE["TEXT"], fontsize=9)
    ax.set_ylabel("Error Rate", color=STYLE["TEXT"], fontsize=9)
    style_legend(ax)
    save_fig(fig, output_dir, f"{model_name}_temporal_residuals.png")


def plot_eigenvalues(s_array, explained_ratio, output_dir, model_name="SVD"):
    """
    Three-panel eigenvalue analysis: singular value decay, per-component
    variance, cumulative variance. Matches step03 academic requirement.
    """
    cumulative = np.cumsum(explained_ratio)
    k = len(explained_ratio)
    x = np.arange(1, k + 1)

    fig, axes = new_fig(nrows=1, ncols=3, figsize=(18, 5))

    # Panel 1: singular value decay (raw from RowMatrix)
    axes[0].bar(x[:len(s_array)], s_array, color=STYLE["BLUE"],
                alpha=0.82, width=0.7)
    axes[0].set_title("Singular Value Decay\n(RowMatrix.computeSVD)",
                      color=STYLE["TEXT"])
    axes[0].set_xlabel("Component", color=STYLE["TEXT"], fontsize=9)
    axes[0].set_ylabel("Singular Value (σ)", color=STYLE["TEXT"], fontsize=9)

    # Panel 2: per-component explained variance
    axes[1].bar(x, explained_ratio * 100, color=STYLE["BLUE"],
                alpha=0.82, width=0.7)
    axes[1].set_title("Per-Component Variance\n(from PCA model)",
                      color=STYLE["TEXT"])
    axes[1].set_xlabel("Component", color=STYLE["TEXT"], fontsize=9)
    axes[1].set_ylabel("Explained Variance (%)", color=STYLE["TEXT"], fontsize=9)

    # Panel 3: cumulative
    axes[2].plot(x, cumulative * 100, "o-", color=STYLE["BLUE"],
                 lw=2.5, markersize=5)
    axes[2].fill_between(x, cumulative * 100, alpha=0.08, color=STYLE["BLUE"])
    axes[2].axhline(90, color=STYLE["AMBER"], ls="--", lw=1.5,
                    alpha=0.8, label="90% threshold")
    axes[2].set_ylim(0, 105)
    axes[2].set_title(
        f"Cumulative Explained Variance\n(k={k}: {cumulative[-1]*100:.1f}%)",
        color=STYLE["TEXT"])
    axes[2].set_xlabel("Number of Components", color=STYLE["TEXT"], fontsize=9)
    axes[2].set_ylabel("Cumulative Variance (%)", color=STYLE["TEXT"], fontsize=9)
    style_legend(axes[2])

    fig.suptitle(f"{model_name}: Eigenvalue Analysis",
                 color=STYLE["TEXT"], fontsize=14, fontweight="bold")
    return save_fig(fig, output_dir, f"{model_name}_eigenvalue_analysis.png")


def plot_cluster_scatter(df_pdf, svd_col_0, svd_col_1,
                         cluster_col, label_col,
                         output_dir, model_name="KMeans"):
    """
    Two-panel KMeans cluster scatter: cluster assignments + ground truth.
    df_pdf: sampled pandas DataFrame with svd_0, svd_1, cluster, label columns.
    """
    fig, axes = new_fig(nrows=1, ncols=2, figsize=(16, 7))

    # Panel 1: cluster assignments
    for cid in sorted(df_pdf[cluster_col].unique()):
        sub = df_pdf[df_pdf[cluster_col] == cid]
        axes[0].scatter(sub[svd_col_0], sub[svd_col_1],
                        c=STYLE["CLUSTERS"][int(cid) % len(STYLE["CLUSTERS"])],
                        s=18, alpha=0.75, label=f"C{cid}",
                        edgecolors="white", linewidths=0.2)
    axes[0].set_title(f"{model_name}: Clusters in PC Space",
                      color=STYLE["TEXT"])
    axes[0].set_xlabel("SVD Component 1", color=STYLE["TEXT"], fontsize=9)
    axes[0].set_ylabel("SVD Component 2", color=STYLE["TEXT"], fontsize=9)
    style_legend(axes[0], ncol=4, loc="upper right")
    axes[0].grid(color=STYLE["GRID"], lw=0.4, alpha=0.5)

    # Panel 2: ground truth label overlay
    lc = {0: STYLE["GEO_TN"], 1: STYLE["RED"]}
    for lv in [0, 1]:
        sub = df_pdf[df_pdf[label_col] == lv]
        axes[1].scatter(sub[svd_col_0], sub[svd_col_1],
                        c=lc[lv], label=f"Label={lv}",
                        s=3 if lv == 0 else 8,
                        alpha=0.3 if lv == 0 else 0.85,
                        edgecolors="none")
    axes[1].set_title(f"{model_name}: Ground Truth in PC Space",
                      color=STYLE["TEXT"])
    axes[1].set_xlabel("SVD Component 1", color=STYLE["TEXT"], fontsize=9)
    axes[1].set_ylabel("SVD Component 2", color=STYLE["TEXT"], fontsize=9)
    style_legend(axes[1], markerscale=3)
    axes[1].grid(color=STYLE["GRID"], lw=0.4, alpha=0.5)

    fig.suptitle(f"{model_name}: Cluster Analysis",
                 color=STYLE["TEXT"], fontsize=13, fontweight="bold")
    return save_fig(fig, output_dir, f"{model_name}_cluster_scatter.png")


def plot_regional_prauc(regions, pr_aucs, baseline, output_dir, model_name,
                        highlight_region="amundsen"):
    """
    Horizontal bar chart of per-region PR-AUC with baseline reference line.
    Amundsen (or highlight_region) shown in RED, all others in BLUE.
    """
    fig, ax = new_fig(figsize=(9, 5))

    colors = [STYLE["RED"] if r == highlight_region else STYLE["BLUE"]
              for r in regions]
    bars = ax.bar(regions, pr_aucs, color=colors, alpha=0.85,
                  width=0.6, edgecolor="white", linewidth=0.5)

    # value labels
    for bar, val in zip(bars, pr_aucs):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.003,
                f"{val:.3f}", ha="center", va="bottom",
                color=STYLE["TEXT"], fontsize=7.5, fontweight="bold")

    ax.axhline(baseline, color=STYLE["AMBER"], ls="--",
               lw=1.8, label=f"XGB Baseline ({baseline:.3f})")
    ax.set_title(f"{model_name}: Regional PR-AUC", color=STYLE["TEXT"])
    ax.set_xlabel("Region", color=STYLE["TEXT"], fontsize=9)
    ax.set_ylabel("PR-AUC", color=STYLE["TEXT"], fontsize=9)
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    style_legend(ax)
    return save_fig(fig, output_dir, f"{model_name}_regional_prauc.png")


def plot_confusion_matrix(cm_data, output_dir, model_name,
                          split_name="test"):
    """
    Styled confusion matrix heatmap with counts and percentages.
    cm_data: 2x2 numpy array [[TN, FP], [FN, TP]]
    """
    cm_norm = cm_data / cm_data.sum()
    labels  = [["TN", "FP"], ["FN", "TP"]]

    fig, ax = new_fig(figsize=(6, 5))
    im = ax.imshow(cm_norm, cmap=STYLE["CMAP_CM"], vmin=0, vmax=1)

    for i in range(2):
        for j in range(2):
            v = cm_norm[i, j]
            n = int(cm_data[i, j])
            ax.text(j, i,
                    f"{labels[i][j]}\n{n:,}\n({v:.1%})",
                    ha="center", va="center",
                    fontsize=10, fontweight="bold",
                    color="white" if v > 0.35 else STYLE["TEXT"])

    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"],
                       color=STYLE["TEXT"], fontsize=9)
    ax.set_yticklabels(["True 0", "True 1"],
                       color=STYLE["TEXT"], fontsize=9)
    ax.set_title(f"{model_name}: Confusion Matrix ({split_name})",
                 color=STYLE["TEXT"])
    ax.grid(False)

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    style_colorbar(cb)
    return save_fig(fig, output_dir,
                    f"{model_name}_confusion_matrix_{split_name}.png")