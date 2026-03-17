# =====================================================================
# 04_confusion_matrix.py
# Generates XGB confusion matrix PNGs using the Deep Field plot_config.
#
# All constants are copied verbatim from the results files.
#
# Source files:
#   02_xgb_baseline_results.txt  
#   03_xgb_tuned_results.txt
#
# Method: back-calculation from threshold calibration table.
#   TP = round(Recall    * test_positives)
#   FN = test_positives - TP
#   FP = round(TP * (1 - Precision) / Precision)
#   TN = test_negatives - FP
#   Total is asserted to equal TEST_TOTAL exactly.
#
# =====================================================================

import sys
import os
import numpy as np

from plot_config import apply_style, STYLE, new_fig, save_fig, style_colorbar

apply_style()

# =====================================================================
# CONSTANTS -- copied verbatim from results files, no edits
# =====================================================================

# -- DATA LOADING section (identical in both results files) -----------
# test: 126,716,756 rows, pos=5,193,607, rate=0.040986
TEST_TOTAL     = 126_716_756
TEST_POSITIVES =   5_193_607
TEST_NEGATIVES = TEST_TOTAL - TEST_POSITIVES

# -- XGB_Baseline threshold calibration on TEST -----------------------
# Source: 02_xgb_baseline_results.txt
# Best threshold: 0.40 (F1=0.6329)
# Threshold  Precision  Recall    F1
#     0.40     0.4916    0.8881  0.6329
BASELINE_THRESHOLD = 0.40
BASELINE_PRECISION = 0.4916
BASELINE_RECALL    = 0.8881

# -- XGB_Tuned threshold calibration on TEST --------------------------
# Source: 03_xgb_tuned_results.txt
# Best threshold: 0.15 (F1=0.6211)
# Threshold  Precision  Recall    F1
#     0.15     0.4563    0.9724  0.6211
TUNED_THRESHOLD = 0.15
TUNED_PRECISION = 0.4563
TUNED_RECALL    = 0.9724

# -- Output directory (matches pipeline_config.py OUTPUT_DIR) ---------
OUTPUT_DIR = "../imgs"

# =====================================================================
# BACK-CALCULATION -- Python does all arithmetic
# =====================================================================

"""
## `back_calculate()` function
* Back-calculates TP, FP, FN, TN from precision, recall, and test set counts.
* Asserts the total matches.
"""
def back_calculate(model_name, threshold, precision, recall,
                   positives, negatives):
    total    = positives + negatives
    TP       = round(recall * positives)
    FN       = positives - TP
    FP       = round(TP * (1 - precision) / precision)
    TN       = negatives - FP
    computed = TP + FP + FN + TN
    assert computed == total, (
        f"[{model_name}] Row count mismatch: "
        f"got {computed:,}, expected {total:,}"
    )
    prec_check = TP / (TP + FP)
    rec_check  = TP / (TP + FN)
    print(f"{model_name} @ t={threshold}")
    print(f"  TP={TP:>12,}  FP={FP:>12,}")
    print(f"  FN={FN:>12,}  TN={TN:>12,}")
    print(f"  total      : {computed:,}")
    print(f"  prec check : {prec_check:.4f}  (source={precision})")
    print(f"  rec  check : {rec_check:.4f}  (source={recall})")
    return TP, FP, FN, TN


print("=" * 60)
print("  CONFUSION MATRIX BACK-CALCULATION")
print("=" * 60)

TP_b, FP_b, FN_b, TN_b = back_calculate(
    "XGB_Baseline", BASELINE_THRESHOLD,
    BASELINE_PRECISION, BASELINE_RECALL,
    TEST_POSITIVES, TEST_NEGATIVES,
)

print()

TP_t, FP_t, FN_t, TN_t = back_calculate(
    "XGB_Tuned", TUNED_THRESHOLD,
    TUNED_PRECISION, TUNED_RECALL,
    TEST_POSITIVES, TEST_NEGATIVES,
)

# =====================================================================
# PLOT -- sqrt-normalised colormap for contrast
# The TN cell is ~25x larger than TP/FP/FN. Linear normalisation makes
# the minority cells invisible. sqrt compression brings them into range
# while preserving the relative ordering of all four cells.
# =====================================================================

"""
## `plot_cm()` function
* Renders a sqrt-normalised confusion matrix heatmap with counts and percentages.
"""
def plot_cm(TN, FP, FN, TP, model_name, threshold, output_dir):
    cm    = np.array([[TN, FP], [FN, TP]], dtype=np.float64)
    total = cm.sum()

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
        f"{model_name} Confusion Matrix (test, threshold={threshold})",
        color=STYLE["TEXT"],
    )

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    style_colorbar(cb, label="sqrt( count / total )")

    fname = f"{model_name}_confusion_matrix_test.png"
    return save_fig(fig, output_dir, fname)


# =====================================================================
# GENERATE PLOTS
# =====================================================================

print()
plot_cm(TN_b, FP_b, FN_b, TP_b, "XGB_Baseline", BASELINE_THRESHOLD, OUTPUT_DIR)
plot_cm(TN_t, FP_t, FN_t, TP_t, "XGB_Tuned",    TUNED_THRESHOLD,    OUTPUT_DIR)

print("\nDone.")
