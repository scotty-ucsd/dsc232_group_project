# =====================================================================
# 01_svd_kmeans.py: SVD + KMeans + XGBoost pipeline (Model 2).
#
# Loads unified features from data/ml_ready_unified/, trims to 20
# clean features, applies regional residualization and lag engineering,
# reduces dimensionality via RowMatrix.computeSVD + PCA, clusters with
# KMeans, and trains a SparkXGBClassifier on the SVD components plus
# cluster ID.  Runs on SDSC Expanse via: sbatch run_svd_kmeans.sh
# =====================================================================

from __future__ import annotations

import os
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_config import apply_style, STYLE, new_fig, save_fig
apply_style()

from pyspark.ml import Pipeline
from pyspark.ml.clustering import KMeans as SparkKMeans
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    ClusteringEvaluator,
)
from pyspark.ml.feature import Imputer, PCA, StandardScaler, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.mllib.linalg import Vectors as MLLibVectors
from pyspark.mllib.linalg.distributed import RowMatrix
from pyspark.sql import SparkSession, functions as F
from pyspark import StorageLevel
from xgboost.spark import SparkXGBClassifier

from pipeline_config import (
    MODE, LOCAL, scratch_dir, OUTPUT_DIR,
    LABEL_COL, WEIGHT_COL, PREDICTION_COL,
    get_spark, load_and_split, undersample,
    evaluate, regional_summary, save_predictions,
    print_results_table, fitting_analysis, print_conclusion,
    plot_geographic_errors, plot_temporal_residuals,
)

# =====================================================================
# 1. CONSTANTS
# =====================================================================

FEATURES_CLEAN = [
    "thetao_mo", "t_star_mo", "so_mo", "t_f_mo",
    "thetao_quarterly_avg", "thetao_quarterly_std",
    "t_star_quarterly_avg", "t_star_quarterly_std",
    "regional_t_star_climatology",
    "dist_to_grounding_line", "grounding_line_vulnerability",
    "retrograde_flag", "bed_slope", "bed", "clamped_depth",
    "ice_draft", "thickness", "ice_area",
    "lwe_trend",
    "month_idx",
]

OCEAN_RESID_COLS = [
    "thetao_mo", "t_star_mo", "so_mo", "t_f_mo",
    "thetao_quarterly_avg", "thetao_quarterly_std",
    "t_star_quarterly_avg", "t_star_quarterly_std",
    "regional_t_star_climatology",
]

META_COLS  = ["regional_subset_id", "month_idx", "x", "y"]
REGION_COL = "regional_subset_id"

SVD_K        = 15
KMEANS_SEED  = 42
KMEANS_ITERS = 50

# FIX 6 -- expanded weight map (ross/ronne raised; totten added)
REGION_WEIGHT_MAP = {
    "amundsen_sea":        10.0,
    "totten_and_aurora":    5.0,
    "bellingshausen":       5.0,
    "antarctic_peninsula":  2.0,
    "ross":                 2.0,
    "ronne":                2.0,
}

# FIX 5 -- hard cap on scale_pos_weight
SPW_CAP = 15.0


# =====================================================================
# 2. FEATURE TRIMMING
# =====================================================================

def trim_to_clean_features(df, label_col=LABEL_COL, weight_col=WEIGHT_COL):
    """Select ONLY the 20 clean features + label + weight + metadata."""
    keep_meta   = [c for c in META_COLS      if c in df.columns]
    keep_feat   = [c for c in FEATURES_CLEAN if c in df.columns]
    keep_target = [c for c in [label_col, weight_col] if c in df.columns]
    missing = [c for c in FEATURES_CLEAN if c not in df.columns]
    if missing:
        print(f"  [trim] WARNING -- {len(missing)} clean features absent: {missing}")
    selected = list(dict.fromkeys(keep_meta + keep_feat + keep_target))
    df = df.select(*selected)
    print(f"  [trim] Kept {len(keep_feat)}/20 clean features + "
          f"{len(keep_meta)} meta + {len(keep_target)} target cols "
          f"= {len(selected)} total columns.")
    return df


# =====================================================================
# 3. REGIONAL WEIGHTS
# =====================================================================

def add_region_weights(df):
    """FIX 6: Expanded map; ross/ronne no longer downweighted."""
    region_weight = F.coalesce(
        *[F.when(F.col(REGION_COL) == F.lit(r), F.lit(w))
          for r, w in REGION_WEIGHT_MAP.items()],
        F.lit(1.0),
    )
    return df.withColumn(WEIGHT_COL, F.col(WEIGHT_COL) * region_weight)


# =====================================================================
# 4. REGIONAL RESIDUALIZATION
# =====================================================================

def compute_region_means(train_df, cols):
    cols = [c for c in cols if c in train_df.columns]
    return train_df.groupBy(REGION_COL).agg(
        *[F.avg(c).alias(f"{c}_region_mean") for c in cols])


def apply_region_residuals(df, region_means_df, cols):
    cols = [c for c in cols if c in df.columns]
    df   = df.join(region_means_df, on=REGION_COL, how="left")
    resid_cols = []
    for c in cols:
        resid_col = f"{c}_resid"
        df = df.withColumn(resid_col,
            F.col(c) - F.coalesce(F.col(f"{c}_region_mean"), F.lit(0.0)))
        resid_cols.append(resid_col)
    df = df.drop(*[f"{c}_region_mean" for c in cols])
    return df, resid_cols


# =====================================================================
# 5. LAG FEATURES
# =====================================================================

def add_lag_features(df):
    from pyspark.sql.window import Window
    if "x" not in df.columns or "y" not in df.columns:
        print("  [lag] No x/y columns -- skipping.")
        return df, []
    cell_w = Window.partitionBy("x", "y").orderBy("month_idx")
    lag_cols = []
    for col in [c for c in ["thetao_mo", "t_star_mo"] if c in df.columns]:
        for lag in [1, 3]:
            nc = f"{col}_lag{lag}"
            df = df.withColumn(nc, F.lag(col, lag).over(cell_w))
            lag_cols.append(nc)
    print(f"  [lag] Added {len(lag_cols)} lag features: {lag_cols}")
    return df, lag_cols


# =====================================================================
# 6. ENGINEERED FEATURES
# =====================================================================

def add_engineered_features(df):
    feat_cols = []
    if "thetao_mo" in df.columns and "t_f_mo" in df.columns:
        df = df.withColumn("thermal_driving", F.col("thetao_mo") - F.col("t_f_mo"))
        feat_cols.append("thermal_driving")
    if "t_star_mo" in df.columns and "regional_t_star_climatology" in df.columns:
        df = df.withColumn("t_star_anomaly",
            F.col("t_star_mo") - F.col("regional_t_star_climatology"))
        feat_cols.append("t_star_anomaly")
    if "month_idx" in df.columns:
        df = (df
              .withColumn("sin_month",
                  F.sin(2.0 * 3.14159265 * F.col("month_idx") / 12.0))
              .withColumn("cos_month",
                  F.cos(2.0 * 3.14159265 * F.col("month_idx") / 12.0)))
        feat_cols += ["sin_month", "cos_month"]
    print(f"  [engineer] Added {len(feat_cols)} engineered features: {feat_cols}")
    return df, feat_cols


# =====================================================================
# 7. PREPROCESSING PIPELINE
# =====================================================================

def build_preprocessing_pipeline(feature_cols):
    imputed = [f"{c}_imp" for c in feature_cols]
    stages  = [
        Imputer(strategy="median", inputCols=feature_cols, outputCols=imputed),
        VectorAssembler(inputCols=imputed, outputCol="raw_features",
                        handleInvalid="skip"),
        StandardScaler(inputCol="raw_features", outputCol="scaled_features",
                       withStd=True, withMean=False),
    ]
    print(f"  [preprocess] {len(feature_cols)} features -> "
          "Imputer -> Assembler -> StandardScaler(withMean=False)")
    return Pipeline(stages=stages)


# =====================================================================
# 8. SVD  (FIX 1)
# =====================================================================

def compute_rowmatrix_svd(scaled_df, k=SVD_K, feature_col="scaled_features"):
    """
    FIX 1: Returns only s_array (singular values for the decay plot).
    Explained variance ratios come from pca_model.explainedVariance instead,
    which is already a proper [0,1] per-component ratio.
    """
    rdd = (scaled_df
           .select(vector_to_array(F.col(feature_col)).alias("arr"))
           .rdd.map(lambda row: MLLibVectors.dense(row["arr"])))
    mat = RowMatrix(rdd)
    print(f"  [SVD] Running RowMatrix.computeSVD(k={k}, computeU=False)...")
    svd_result = mat.computeSVD(k, computeU=False)
    s_array = (np.array(svd_result.s.toArray())
               if hasattr(svd_result.s, "toArray") else np.array(svd_result.s))
    V       = svd_result.V
    print(f"  [SVD] Top-5 singular values : {np.round(s_array[:5], 4)}")
    print(f"  [SVD] NOTE: Explained variance ratios come from PCA model "
          f"(correct [0,1] scale -- same Lanczos algorithm).")
    return V, s_array


def project_with_pca(train_scaled, val_scaled, test_scaled, k=SVD_K):
    """
    FIX 1: Now returns explained_ratio from pca_model.explainedVariance.
    """
    pca       = PCA(k=k, inputCol="scaled_features", outputCol="svd_features")
    print(f"  [PCA] Fitting PCA(k={k}) on training data...")
    pca_model = pca.fit(train_scaled)

    explained = (np.array(pca_model.explainedVariance.toArray())
                 if hasattr(pca_model.explainedVariance, "toArray")
                 else np.array(pca_model.explainedVariance))
    cumul = np.cumsum(explained)
    print(f"  [PCA] Explained variance (top 5): {np.round(explained[:5], 4)}")
    print(f"  [PCA] Cumulative at k={k}: {cumul[-1]:.4f}")   # now prints e.g. 0.8762

    def _expand_pc_cols(df, k_eff):
        arr = vector_to_array(F.col("svd_features"))
        for i in range(k_eff):
            df = df.withColumn(f"svd_{i}", arr.getItem(i))
        return df

    train_svd = _expand_pc_cols(pca_model.transform(train_scaled), k)
    val_svd   = _expand_pc_cols(pca_model.transform(val_scaled),   k)
    test_svd  = _expand_pc_cols(pca_model.transform(test_scaled),  k)

    return pca_model, explained, train_svd, val_svd, test_svd


# =====================================================================
# 9. EIGENVALUE PLOTS  (FIX 9)
# =====================================================================

def plot_eigenvalues(s_array, explained_ratio, model_name="SVD"):
    """
    FIX 9: All panels use PCA-derived explained_ratio.
    Cumulative panel y-axis capped at 105 -- correct % values.
    Style 2C Deep Field applied via plot_config.
    """
    cumulative = np.cumsum(explained_ratio)
    k = len(explained_ratio)
    x = np.arange(1, k + 1)

    fig, axes = new_fig(nrows=1, ncols=3, figsize=(18, 5))

    # Panel 1: singular value decay (raw from RowMatrix)
    axes[0].bar(x[:len(s_array)], s_array, color=STYLE["BLUE"], alpha=0.82, width=0.7)
    axes[0].set_xlabel("Component", color=STYLE["TEXT"], fontsize=9)
    axes[0].set_ylabel("Singular Value (sigma)", color=STYLE["TEXT"], fontsize=9)
    axes[0].set_title("Singular Value Decay\n(RowMatrix.computeSVD)",
                      color=STYLE["TEXT"])

    # Panel 2: per-component explained variance
    axes[1].bar(x, explained_ratio * 100, color=STYLE["BLUE"], alpha=0.82, width=0.7)
    axes[1].set_xlabel("Component", color=STYLE["TEXT"], fontsize=9)
    axes[1].set_ylabel("Explained Variance (%)", color=STYLE["TEXT"], fontsize=9)
    axes[1].set_title("Per-Component Variance\n(from PCA model)",
                      color=STYLE["TEXT"])

    # Panel 3: cumulative explained variance
    axes[2].plot(x, cumulative * 100, "o-", color=STYLE["BLUE"],
                 linewidth=2.5, markersize=5)
    axes[2].fill_between(x, cumulative * 100, alpha=0.08, color=STYLE["BLUE"])
    axes[2].axhline(y=90, color=STYLE["AMBER"], linestyle="--",
                    alpha=0.8, label="90% threshold")
    axes[2].set_ylim(0, 105)
    axes[2].set_xlabel("Number of Components", color=STYLE["TEXT"], fontsize=9)
    axes[2].set_ylabel("Cumulative Variance (%)", color=STYLE["TEXT"], fontsize=9)
    axes[2].set_title(
        f"Cumulative Explained Variance\n(k={k}: {cumulative[-1]*100:.1f}%)",
        color=STYLE["TEXT"])
    axes[2].legend(facecolor=STYLE["BG"], edgecolor=STYLE["GRID"],
                   labelcolor=STYLE["TEXT"])

    fig.suptitle(f"{model_name}: Eigenvalue Analysis",
                 color=STYLE["TEXT"], fontsize=14, fontweight="bold")
    save_fig(fig, OUTPUT_DIR, f"{model_name}_eigenvalue_analysis.png")
    print(f"  [{model_name}] Eigenvalue plot saved")


# =====================================================================
# 10. KMEANS
# =====================================================================

def run_kmeans(df_svd, model_name="SVD_KMeans"):
    print(f"\n  [KMeans] Silhouette sweep...")
    best_k, best_sil = 6, -1.0
    for k_test in [4, 6, 8, 10]:
        km_m = SparkKMeans(featuresCol="svd_features", predictionCol="cluster",
                           k=k_test, seed=KMEANS_SEED, maxIter=KMEANS_ITERS).fit(df_svd)
        sil = ClusteringEvaluator(featuresCol="svd_features", predictionCol="cluster",
                                  metricName="silhouette",
                                  distanceMeasure="squaredEuclidean"
                                  ).evaluate(km_m.transform(df_svd))
        print(f"    k={k_test}: silhouette={sil:.4f}")
        if sil > best_sil:
            best_sil, best_k = sil, k_test

    print(f"  [KMeans] Best k={best_k} (sil={best_sil:.4f}).")
    km_model     = SparkKMeans(featuresCol="svd_features", predictionCol="cluster",
                               k=best_k, seed=KMEANS_SEED, maxIter=KMEANS_ITERS).fit(df_svd)
    df_clustered = km_model.transform(df_svd)
    sil_final    = ClusteringEvaluator(featuresCol="svd_features", predictionCol="cluster",
                                       metricName="silhouette",
                                       distanceMeasure="squaredEuclidean"
                                       ).evaluate(df_clustered)

    cluster_counts = (df_clustered.groupBy("cluster")
                      .agg(F.count("*").alias("n"),
                           F.avg(F.col(LABEL_COL).cast("float")).alias("pos_rate"))
                      .orderBy("cluster").collect())
    print(f"\n  [{model_name}] Cluster breakdown (k={best_k}):")
    for r in cluster_counts:
        print(f"    cluster={r['cluster']}: n={r['n']:>12,}  pos_rate={r['pos_rate']:.6f}")

    region_cluster = (df_clustered.groupBy("cluster", REGION_COL)
                      .count().orderBy("cluster", REGION_COL).collect())
    print(f"\n  [{model_name}] Cluster x Region:")
    for r in region_cluster:
        print(f"    cluster={r['cluster']}  {r[REGION_COL]:25s}  n={r['count']:>10,}")

    return km_model, df_clustered, sil_final, best_k


def plot_clusters(df_clustered, k_eff, model_name="SVD_KMeans"):
    total  = df_clustered.count()
    frac   = min(1.0, 15_000 / max(total, 1))
    avail  = [c for c in ["svd_0", "svd_1", "cluster", LABEL_COL, REGION_COL]
              if c in df_clustered.columns]
    pdf    = df_clustered.select(*avail).sample(fraction=frac, seed=42).toPandas()
    if pdf.empty:
        return

    # Panel 1: cluster assignments
    fig, ax = new_fig(figsize=(10, 8))
    for cid in sorted(pdf["cluster"].unique()):
        sub = pdf[pdf["cluster"] == cid]
        ax.scatter(sub["svd_0"], sub["svd_1"],
                   c=STYLE["CLUSTERS"][int(cid) % len(STYLE["CLUSTERS"])],
                   label=f"Cluster {cid}", alpha=0.5, s=3)
    ax.set_xlabel("SVD Component 1", color=STYLE["TEXT"], fontsize=9)
    ax.set_ylabel("SVD Component 2", color=STYLE["TEXT"], fontsize=9)
    ax.set_title(f"{model_name}: Clusters in PC Space", color=STYLE["TEXT"])
    ax.legend(facecolor=STYLE["BG"], edgecolor=STYLE["GRID"],
              labelcolor=STYLE["TEXT"], markerscale=3)
    ax.grid(color=STYLE["GRID"], lw=0.4, alpha=0.5)
    save_fig(fig, OUTPUT_DIR, f"{model_name}_cluster_scatter.png")
    print(f"  [{model_name}] Cluster scatter saved")

    if LABEL_COL in pdf.columns:
        fig2, ax2 = new_fig(figsize=(10, 8))
        lc = {0: STYLE["GEO_TN"], 1: STYLE["RED"]}
        for lv in [0, 1]:
            sub = pdf[pdf[LABEL_COL] == lv]
            ax2.scatter(sub["svd_0"], sub["svd_1"], c=lc[lv],
                        label=f"Label={lv}",
                        alpha=0.3 if lv == 0 else 0.85,
                        s=2 if lv == 0 else 6,
                        edgecolors="none")
        ax2.set_xlabel("SVD Component 1", color=STYLE["TEXT"], fontsize=9)
        ax2.set_ylabel("SVD Component 2", color=STYLE["TEXT"], fontsize=9)
        ax2.set_title(f"{model_name}: Ground Truth in PC Space",
                      color=STYLE["TEXT"])
        ax2.legend(facecolor=STYLE["BG"], edgecolor=STYLE["GRID"],
                   labelcolor=STYLE["TEXT"], markerscale=3)
        ax2.grid(color=STYLE["GRID"], lw=0.4, alpha=0.5)
        save_fig(fig2, OUTPUT_DIR, f"{model_name}_label_scatter.png")
        print(f"  [{model_name}] Label scatter saved")


# =====================================================================
# 11. APPEND CLUSTER AS FEATURE  (FIX 8)
# =====================================================================

def append_cluster_feature(df_with_cluster, svd_k):
    """
    FIX 8: Appends KMeans cluster_id to the SVD feature vector.
    Cluster pos_rates ranged 0.046-0.132 in run 1 -> meaningful signal.
    """
    pc_cols_avail = [f"svd_{i}" for i in range(svd_k)
                     if f"svd_{i}" in df_with_cluster.columns]

    if "cluster" not in df_with_cluster.columns:
        print("  [cluster_feat] No 'cluster' column -- using svd_features unchanged.")
        return df_with_cluster

    df  = df_with_cluster.withColumn("cluster_float", F.col("cluster").cast("float"))
    asm = VectorAssembler(
        inputCols=pc_cols_avail + ["cluster_float"],
        outputCol="xgb_features",
        handleInvalid="skip",
    )
    df = asm.transform(df)
    print(f"  [cluster_feat] xgb_features = {len(pc_cols_avail)} SVD PCs + cluster_id "
          f"= {len(pc_cols_avail) + 1} dims")
    return df


# =====================================================================
# 12. XGBOOST SUPERVISED MODEL  (FIX 4, 5, 8)
# =====================================================================

def train_xgb_on_pcs(train_svd, val_svd, test_svd, svd_k, model_name="SVD_XGB"):
    """FIX 4+5: Tighter regularisation; SPW capped. FIX 8: xgb_features input."""
    n_neg = train_svd.filter(F.col(LABEL_COL) == 0).count()
    n_pos = train_svd.filter(F.col(LABEL_COL) == 1).count()
    spw   = min(float(n_neg) / float(max(n_pos, 1)), SPW_CAP)
    print(f"  [XGB] N_neg={n_neg:,}  N_pos={n_pos:,}  "
          f"scale_pos_weight={spw:.2f} (cap={SPW_CAP})")

    xgb_cfg = dict(
        max_depth        = 4,          # FIX 4 (was 5)
        eta              = 0.1,        # FIX 4 (was 0.05)
        num_round        = 300 if not LOCAL else 30,  # FIX 4 (was 500)
        subsample        = 0.7,        # FIX 4 (was 0.8)
        colsample_bytree = 0.7,        # FIX 4 (was 0.8)
        min_child_weight = 10.0,       # FIX 4 (was 3.0)
        reg_lambda       = 5.0,        # FIX 4 (was 1.0)
        reg_alpha        = 1.0,        # FIX 4 (was 0.1)
        tree_method      = "hist",
        max_bin          = 256,
        scale_pos_weight = spw,        # FIX 5: capped
        eval_metric      = "aucpr",
        num_workers      = 6,
        use_gpu          = False,
    )

    xgb_clf = SparkXGBClassifier(
        features_col = "xgb_features",   # FIX 8: SVD PCs + cluster_id
        label_col    = LABEL_COL,
        weight_col   = WEIGHT_COL,
        **xgb_cfg,
    )

    print(f"\n{'='*60}\n  TRAINING: {model_name}\n{'='*60}")
    fitted = xgb_clf.fit(train_svd)

    best_threshold = find_optimal_threshold(fitted, val_svd, beta=2.0)
    print(f"  [threshold] Using t={best_threshold:.2f} (Amundsen override=0.35)")

    results = []
    for sname, sdf in [("train", train_svd), ("val", val_svd), ("test", test_svd)]:
        preds = apply_threshold(fitted.transform(sdf), best_threshold)
        preds = preds.persist(StorageLevel.DISK_ONLY)
        preds.count()
        results.append(evaluate(model_name, preds, sname))
        save_predictions(preds, model_name, sname)
        if sname == "test":
            regional_summary(preds, model_name)
            regional_prauc_breakdown(preds, model_name)
            try:
                plot_geographic_errors(preds, model_name)
                plot_temporal_residuals(preds, model_name)
            except Exception as exc:
                print(f"  [{model_name}] Plot failed: {exc}")
        preds.unpersist()

    return results, fitted, best_threshold


# =====================================================================
# 13. DYNAMIC THRESHOLD  (FIX 2, 3)
# =====================================================================

def find_optimal_threshold(fitted_model, val_df, beta=2.0):
    """FIX 2: sweep to 0.95. FIX 3: fallback raised to 0.50."""
    val_preds = fitted_model.transform(val_df)
    val_preds = val_preds.withColumn(
        "pos_prob", vector_to_array(F.col("probability")).getItem(1))
    pdf = val_preds.select("pos_prob", LABEL_COL).toPandas()

    best_t, best_fb = 0.50, 0.0   # FIX 3
    for t in np.arange(0.10, 0.96, 0.02):   # FIX 2
        preds = (pdf["pos_prob"] >= t).astype(int)
        tp = int(((preds == 1) & (pdf[LABEL_COL] == 1)).sum())
        fp = int(((preds == 1) & (pdf[LABEL_COL] == 0)).sum())
        fn = int(((preds == 0) & (pdf[LABEL_COL] == 1)).sum())
        if tp + fp == 0 or tp + fn == 0:
            continue
        prec = tp / (tp + fp)
        rec  = tp / (tp + fn)
        fb   = (1 + beta**2) * prec * rec / (beta**2 * prec + rec + 1e-9)
        if fb > best_fb:
            best_fb, best_t = fb, float(t)

    print(f"  [threshold] Optimal t={best_t:.2f}, F{beta:.0f}={best_fb:.4f}")
    return best_t


def apply_threshold(preds_df, threshold, amundsen_threshold=0.35):
    """FIX 7: Amundsen override raised 0.25->0.35 (TPR was 0.93 in run 1)."""
    preds_df = preds_df.withColumn(
        "pos_prob", vector_to_array(F.col("probability")).getItem(1))
    final_pred = (
        F.when(
            (F.col(REGION_COL) == F.lit("amundsen_sea")) &
            (F.col("pos_prob") >= F.lit(amundsen_threshold)), F.lit(1))
        .when(F.col("pos_prob") >= F.lit(threshold), F.lit(1))
        .otherwise(F.lit(0))
    )
    return preds_df.withColumn(PREDICTION_COL, final_pred.cast("double"))


# =====================================================================
# 14. REGIONAL PR-AUC BREAKDOWN
# =====================================================================

def regional_prauc_breakdown(preds_df, model_name):
    evaluator = BinaryClassificationEvaluator(
        labelCol=LABEL_COL, rawPredictionCol="rawPrediction",
        metricName="areaUnderPR")
    regions = [r[REGION_COL] for r in
               preds_df.select(REGION_COL).distinct().collect()]

    print(f"\n{'='*60}")
    print(f"  REGIONAL BREAKDOWN: {model_name}")
    print(f"{'='*60}")
    print(f"  {'Region':<30} {'PR-AUC':>8} {'TPR':>8} {'FPR':>8} {'N+':>8}")

    amundsen_tpr = None
    for region in sorted(regions):
        rdf   = preds_df.filter(F.col(REGION_COL) == region)
        n_pos = rdf.filter(F.col(LABEL_COL) == 1).count()
        if n_pos < 10:
            continue
        pr_auc = evaluator.evaluate(rdf)
        cm = rdf.agg(
            F.sum(F.when((F.col(LABEL_COL)==1)&(F.col(PREDICTION_COL)==1),1).otherwise(0)).alias("tp"),
            F.sum(F.when((F.col(LABEL_COL)==1)&(F.col(PREDICTION_COL)==0),1).otherwise(0)).alias("fn"),
            F.sum(F.when((F.col(LABEL_COL)==0)&(F.col(PREDICTION_COL)==1),1).otherwise(0)).alias("fp"),
            F.sum(F.when((F.col(LABEL_COL)==0)&(F.col(PREDICTION_COL)==0),1).otherwise(0)).alias("tn"),
        ).collect()[0]
        tp, fn, fp, tn = cm["tp"], cm["fn"], cm["fp"], cm["tn"]
        tpr = tp / (tp + fn + 1e-9)
        fpr = fp / (fp + tn + 1e-9)
        flag = " <- AMUNDSEN" if "amundsen" in region.lower() else ""
        print(f"  {region:<30} {pr_auc:>8.4f} {tpr:>8.4f} "
              f"{fpr:>8.4f} {n_pos:>8}{flag}")
        if "amundsen" in region.lower():
            amundsen_tpr = tpr

    if amundsen_tpr is not None:
        status = "PASS PASS" if amundsen_tpr >= 0.50 else "FAIL FAIL (target >= 50%)"
        print(f"\n  Amundsen Sea TPR: {amundsen_tpr:.4f} -> {status}")


# =====================================================================
# 15. PREDICTIONS ANALYSIS
# =====================================================================

def predictions_analysis(preds, model_name):
    cols = [c for c in [REGION_COL, "month_idx", LABEL_COL, PREDICTION_COL]
            if c in preds.columns]
    print(f"\n{'='*60}\n  PREDICTIONS ANALYSIS: {model_name}\n{'='*60}")
    for label, cond in [
        ("True Positives  (correct melt detection)",
         (F.col(LABEL_COL)==1) & (F.col(PREDICTION_COL)==1)),
        ("False Positives (false alarm)",
         (F.col(LABEL_COL)==0) & (F.col(PREDICTION_COL)==1)),
        ("False Negatives (missed melt event)",
         (F.col(LABEL_COL)==1) & (F.col(PREDICTION_COL)==0)),
    ]:
        print(f"\n  {label}:")
        for r in preds.filter(cond).select(*cols).limit(5).collect():
            print(f"    region={r[REGION_COL]}, month={r['month_idx']}, "
                  f"truth={r[LABEL_COL]}, pred={r[PREDICTION_COL]}")

    cm = preds.agg(
        F.sum(F.when((F.col(LABEL_COL)==1)&(F.col(PREDICTION_COL)==1),1).otherwise(0)).alias("TP"),
        F.sum(F.when((F.col(LABEL_COL)==0)&(F.col(PREDICTION_COL)==1),1).otherwise(0)).alias("FP"),
        F.sum(F.when((F.col(LABEL_COL)==1)&(F.col(PREDICTION_COL)==0),1).otherwise(0)).alias("FN"),
        F.sum(F.when((F.col(LABEL_COL)==0)&(F.col(PREDICTION_COL)==0),1).otherwise(0)).alias("TN"),
    ).collect()[0]
    total    = cm["TP"] + cm["FP"] + cm["FN"] + cm["TN"]
    prec_man = cm["TP"] / max(cm["TP"] + cm["FP"], 1)
    rec_man  = cm["TP"] / max(cm["TP"] + cm["FN"], 1)
    print(f"\n  Confusion Matrix (total={total:,}):")
    print(f"    TP={cm['TP']:>10,}  FP={cm['FP']:>10,}")
    print(f"    FN={cm['FN']:>10,}  TN={cm['TN']:>10,}")
    print(f"  Precision (manual): {prec_man:.4f}")
    print(f"  Recall    (manual): {rec_man:.4f}")


# =====================================================================
# 16. MAIN ENTRY POINT
# =====================================================================

def run_svd_kmeans():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    spark = get_spark()

    try:
        print(f"\n{'='*60}\n  STAGE 1: DATA LOADING\n{'='*60}")
        train_raw, val_raw, test_raw = load_and_split(spark)

        print(f"\n{'='*60}\n  STAGE 2: TRIM TO 20 CLEAN FEATURES\n{'='*60}")
        train_raw = trim_to_clean_features(train_raw)
        val_raw   = trim_to_clean_features(val_raw)
        test_raw  = trim_to_clean_features(test_raw)

        print(f"\n{'='*60}\n  STAGE 3: CHECKPOINT RAW TRAIN\n{'='*60}")
        raw_path = os.path.join(scratch_dir, "_svd_train_raw")
        train_raw.write.mode("overwrite").parquet(raw_path)
        train_raw = spark.read.parquet(raw_path)
        print(f"  Checkpointed -> {raw_path}")

        print(f"\n{'='*60}\n  STAGE 4: REGIONAL WEIGHTS\n{'='*60}")
        train_raw = add_region_weights(train_raw)
        val_raw   = add_region_weights(val_raw)
        test_raw  = add_region_weights(test_raw)

        print(f"\n{'='*60}\n  STAGE 5-6: REGIONAL RESIDUALIZATION\n{'='*60}")
        ocean_avail = [c for c in OCEAN_RESID_COLS if c in train_raw.columns]
        rm_path     = os.path.join(scratch_dir, "_region_means")
        compute_region_means(train_raw, ocean_avail).write.mode("overwrite").parquet(rm_path)
        region_means = spark.read.parquet(rm_path)

        train_resid, resid_cols = apply_region_residuals(train_raw, region_means, ocean_avail)
        val_resid,   _          = apply_region_residuals(val_raw,   region_means, ocean_avail)
        test_resid,  _          = apply_region_residuals(test_raw,  region_means, ocean_avail)

        print(f"\n{'='*60}\n  STAGE 7: LAG FEATURES\n{'='*60}")
        train_resid, lag_cols = add_lag_features(train_resid)
        val_resid,   _        = add_lag_features(val_resid)
        test_resid,  _        = add_lag_features(test_resid)

        print(f"\n{'='*60}\n  STAGE 8: ENGINEERED FEATURES\n{'='*60}")
        train_resid, eng_cols = add_engineered_features(train_resid)
        val_resid,   _        = add_engineered_features(val_resid)
        test_resid,  _        = add_engineered_features(test_resid)

        print(f"\n{'='*60}\n  STAGE 9: CHECKPOINT RESIDUALIZED SPLITS\n{'='*60}")
        for sname, sdf in [("train", train_resid),
                            ("val",   val_resid),
                            ("test",  test_resid)]:
            sdf.write.mode("overwrite").parquet(
                os.path.join(scratch_dir, f"_svd_{sname}_resid"))
        train_resid = spark.read.parquet(os.path.join(scratch_dir, "_svd_train_resid"))
        val_resid   = spark.read.parquet(os.path.join(scratch_dir, "_svd_val_resid"))
        test_resid  = spark.read.parquet(os.path.join(scratch_dir, "_svd_test_resid"))

        print(f"\n{'='*60}\n  STAGE 10: UNDERSAMPLE MAJORITY CLASS\n{'='*60}")
        balanced = undersample(train_resid,
                               checkpoint_dir=os.path.join(scratch_dir, "_svd_us_ckpt"))

        print(f"\n{'='*60}\n  STAGE 11: FIT PREPROCESSING PIPELINE\n{'='*60}")
        static_geo    = [c for c in [
            "dist_to_grounding_line", "grounding_line_vulnerability",
            "retrograde_flag", "bed_slope", "bed", "clamped_depth",
            "ice_draft", "thickness", "ice_area", "lwe_trend",
        ] if c in balanced.columns]
        all_feat_cols = list(dict.fromkeys(
            c for c in (resid_cols + lag_cols + eng_cols + static_geo)
            if c in balanced.columns))

        pp_model = build_preprocessing_pipeline(all_feat_cols).fit(balanced)

        def _tp(df):
            out = pp_model.transform(df).persist(StorageLevel.DISK_ONLY)
            out.count()
            return out

        train_scaled = _tp(balanced)
        val_scaled   = _tp(val_resid)
        test_scaled  = _tp(test_resid)

        print(f"\n{'='*60}\n  STAGE 12: ROWMATRIX.COMPUTESVD (k={SVD_K})\n{'='*60}")
        V, s_array = compute_rowmatrix_svd(train_scaled, k=SVD_K)  # FIX 1

        print(f"\n{'='*60}\n  STAGE 13: PCA PROJECTION\n{'='*60}")
        pca_model, explained_ratio, train_svd, val_svd, test_svd = project_with_pca(
            train_scaled, val_scaled, test_scaled, k=SVD_K)         # FIX 1

        train_scaled.unpersist(); val_scaled.unpersist(); test_scaled.unpersist()

        print(f"\n{'='*60}\n  STAGE 14: CHECKPOINT PROJECTED SPLITS\n{'='*60}")
        pc_cols = [f"svd_{i}" for i in range(SVD_K)]
        keep    = pc_cols + ["svd_features", LABEL_COL, WEIGHT_COL,
                              REGION_COL, "month_idx", "x", "y"]
        for sname, sdf in [("train", train_svd), ("val", val_svd), ("test", test_svd)]:
            p = os.path.join(scratch_dir, f"_svd_{sname}_projected")
            sdf.select(*[c for c in keep if c in sdf.columns]).write.mode("overwrite").parquet(p)
            print(f"  Saved {sname} -> {p}")

        def _reload_svd(sname):
            p  = os.path.join(scratch_dir, f"_svd_{sname}_projected")
            df = spark.read.parquet(p)
            if "svd_features" not in df.columns:
                avail = [c for c in pc_cols if c in df.columns]
                df = VectorAssembler(inputCols=avail, outputCol="svd_features",
                                     handleInvalid="skip").transform(df)
            return df

        train_svd = _reload_svd("train")
        val_svd   = _reload_svd("val")
        test_svd  = _reload_svd("test")

        print(f"\n{'='*60}\n  EIGENVALUE ANALYSIS\n{'='*60}")
        plot_eigenvalues(s_array, explained_ratio, model_name="SVD")  # FIX 9

        print(f"\n{'='*60}\n  STAGE 15: KMEANS CLUSTERING\n{'='*60}")
        km_model, train_clustered, silhouette, best_k = run_kmeans(
            train_svd, model_name="SVD_KMeans")
        val_clustered  = km_model.transform(val_svd)
        test_clustered = km_model.transform(test_svd)
        try:
            plot_clusters(train_clustered, best_k, model_name="SVD_KMeans")
        except Exception as exc:
            print(f"  [KMeans] Plot failed: {exc}, continuing.")

        print(f"\n{'='*60}\n  STAGE 15b: APPEND CLUSTER FEATURE\n{'='*60}")
        train_xgb = append_cluster_feature(train_clustered, SVD_K)  # FIX 8
        val_xgb   = append_cluster_feature(val_clustered,   SVD_K)
        test_xgb  = append_cluster_feature(test_clustered,  SVD_K)

        for sname, sdf in [("train", train_xgb), ("val", val_xgb), ("test", test_xgb)]:
            sdf.write.mode("overwrite").parquet(
                os.path.join(scratch_dir, f"_svd_{sname}_xgb"))
        train_xgb = spark.read.parquet(os.path.join(scratch_dir, "_svd_train_xgb"))
        val_xgb   = spark.read.parquet(os.path.join(scratch_dir, "_svd_val_xgb"))
        test_xgb  = spark.read.parquet(os.path.join(scratch_dir, "_svd_test_xgb"))

        # Reassemble xgb_features if lost in parquet round-trip
        for ref, setter in [(train_xgb, "train"), (val_xgb, "val"), (test_xgb, "test")]:
            if "xgb_features" not in ref.columns:
                avail = [c for c in pc_cols + ["cluster_float"] if c in ref.columns]
                reassembled = VectorAssembler(inputCols=avail, outputCol="xgb_features",
                                             handleInvalid="skip").transform(ref)
                if setter == "train":
                    train_xgb = reassembled
                elif setter == "val":
                    val_xgb   = reassembled
                else:
                    test_xgb  = reassembled

        print(f"\n{'='*60}\n  STAGE 16-18: SUPERVISED MODEL (SVD_XGB)\n{'='*60}")
        results, xgb_fitted, threshold = train_xgb_on_pcs(
            train_xgb, val_xgb, test_xgb, SVD_K, model_name="SVD_XGB")

        print(f"\n{'='*60}\n  STAGE 19: PREDICTIONS ANALYSIS\n{'='*60}")
        test_preds = apply_threshold(xgb_fitted.transform(test_xgb), threshold)
        predictions_analysis(test_preds, "SVD_XGB")

        print_results_table(results)
        fitting_analysis(results)

        print(f"\n{'='*60}")
        print(f"  SVD + KMEANS + XGB PIPELINE COMPLETE")
        print(f"{'='*60}")
        print(f"  SVD components retained : {SVD_K}")
        print(f"  PCA cumulative variance : {np.cumsum(explained_ratio)[-1]:.4f}")
        print(f"  KMeans k (best)         : {best_k}")
        print(f"  KMeans silhouette       : {silhouette:.4f}")
        print(f"  Optimal threshold       : {threshold:.2f}")

        return results

    finally:
        spark.stop()


if __name__ == "__main__":
    run_svd_kmeans()