# =====================================================================
# 04_confusion_matrix.py
# Generates XGB confusion matrix PNGs using the Deep Field plot_config
# style. Numbers are back-calculated from the threshold calibration
# tables in the results files -- no model reload needed.
#
#   python 04_confusion_matrix.py
# =====================================================================

import os
import numpy as np

from plot_config import apply_style, STYLE, new_fig, save_fig, style_colorbar

apply_style()

# =====================================================================
# CONSTANTS -- sourced directly from results files
# =====================================================================

# -- Test set totals (from 02_xgb_baseline_results.txt DATA LOADING) --
TEST_TOTAL     = 126_716_756
TEST_POSITIVES =   5_193_607   # rate=0.040986
TEST_NEGATIVES = TEST_TOTAL - TEST_POSITIVES   # 121,523,149

# -- XGB_Baseline @ threshold=0.40 ------------------------------------
# Source: 02_xgb_baseline_results.txt threshold calibration on test
#   Threshold=0.40  Precision=0.4916  Recall=0.8881  F1=0.6329
BASELINE_THRESHOLD = 0.40
BASELINE_PRECISION = 0.4916
BASELINE_RECALL    = 0.8881

# -- XGB_Tuned @ threshold=0.15 ---------------------------------------
# Source: 03_xgb_tuned_results.txt threshold calibration on test
#   Threshold=0.15  Precision=0.4563  Recall=0.9724  F1=0.6211
TUNED_THRESHOLD = 0.15
TUNED_PRECISION = 0.4563
TUNED_RECALL    = 0.9724

# -- Output directory (matches pipeline_config.py OUTPUT_DIR) ---------
OUTPUT_DIR = os.path.join(os.getcwd(), "dataunified_output")

# =====================================================================
# BACK-CALCULATION
# TP = Recall    * positives
# FN = positives - TP
# FP = TP * (1 - Precision) / Precision
# TN = negatives - FP
# Total check must equal TEST_TOTAL exactly.
# =====================================================================

"""
## `back_calculate()` function
* Back-calculates TP, FP, FN, TN from precision, recall, and test set totals using the formula from threshold calibration tables.
"""
def back_calculate(precision, recall, positives, negatives):
    TP = round(recall    * positives)
    FN = positives - TP
    FP = round(TP * (1 - precision) / precision)
    TN = negatives - FP
    total = TP + FP + FN + TN
    assert total == positives + negatives, (
        f"Row count mismatch: got {total}, expected {positives + negatives}"
    )
    return TP, FP, FN, TN

TP_b, FP_b, FN_b, TN_b = back_calculate(
    BASELINE_PRECISION, BASELINE_RECALL, TEST_POSITIVES, TEST_NEGATIVES
)
TP_t, FP_t, FN_t, TN_t = back_calculate(
    TUNED_PRECISION, TUNED_RECALL, TEST_POSITIVES, TEST_NEGATIVES
)

print(f"XGB_Baseline @ t={BASELINE_THRESHOLD}")
print(f"  TP={TP_b:>12,}  FP={FP_b:>12,}")
print(f"  FN={FN_b:>12,}  TN={TN_b:>12,}")
print(f"  Total: {TP_b+FP_b+FN_b+TN_b:,}")

print(f"\nXGB_Tuned @ t={TUNED_THRESHOLD}")
print(f"  TP={TP_t:>12,}  FP={FP_t:>12,}")
print(f"  FN={FN_t:>12,}  TN={TN_t:>12,}")
print(f"  Total: {TP_t+FP_t+FN_t+TN_t:,}")

# =====================================================================
# PLOT FUNCTION
# Uses a sqrt-normalised colormap so the small TP/FP/FN cells are
# visible against the dominant TN cell, while still using CMAP_CM.
# =====================================================================

"""
## `plot_cm()` function
* Renders a sqrt-normalised confusion matrix heatmap using the Deep Field plot style.
"""
def plot_cm(TN, FP, FN, TP, model_name, threshold, output_dir):
    cm    = np.array([[TN, FP], [FN, TP]], dtype=np.float64)
    total = cm.sum()

    # sqrt normalisation improves contrast when counts span 3+ orders
    cm_sqrt = np.sqrt(cm / total)
    cm_sqrt = cm_sqrt / cm_sqrt.max()

    labels = [["TN", "FP"], ["FN", "TP"]]
    counts = [[TN, FP], [FN, TP]]

    fig, ax = new_fig(figsize=(6, 5))
    ax.grid(False)

    im = ax.imshow(cm_sqrt, cmap=STYLE["CMAP_CM"], vmin=0, vmax=1,
                   aspect="auto")

    for i in range(2):
        for j in range(2):
            n   = counts[i][j]
            pct = n / total
            v   = cm_sqrt[i, j]
            txt = f"{labels[i][j]}\n{n:,}\n({pct:.2%})"
            ax.text(j, i, txt,
                    ha="center", va="center",
                    fontsize=9, fontweight="bold",
                    color="white" if v > 0.55 else STYLE["TEXT"])

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred Negative", "Pred Positive"],
                       color=STYLE["TEXT"], fontsize=9)
    ax.set_yticklabels(["True Negative", "True Positive"],
                       color=STYLE["TEXT"], fontsize=9)
    ax.set_title(
        f"{model_name}: Confusion Matrix (test, threshold={threshold})\n"
        f"Color scale: sqrt-normalised for contrast",
        color=STYLE["TEXT"],
    )

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    style_colorbar(cb, label="sqrt( count / total )")

    fname = f"{model_name}_confusion_matrix_test.png"
    return save_fig(fig, output_dir, fname)


# =====================================================================
# GENERATE PLOTS
# =====================================================================

plot_cm(TN_b, FP_b, FN_b, TP_b, "XGB_Baseline", BASELINE_THRESHOLD, OUTPUT_DIR)
plot_cm(TN_t, FP_t, FN_t, TP_t, "XGB_Tuned",    TUNED_THRESHOLD,    OUTPUT_DIR)

print("\nDone.")
