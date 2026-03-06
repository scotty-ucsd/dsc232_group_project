<div align="center">
  <h1>Antarctic Ice Sheet Instability Prediction</h1>
  <h3><i>Distributed Machine Learning on Multi-Sensor Satellite Data</i></h3>
  <h4>DSC 232R: Big Data Analytics — Final Report</h4>

  <p>
    <strong>Scotty Rogers</strong> (Pipeline Architect & Data Engineer) &nbsp;&bull;&nbsp;
    <strong>Hans Hanson</strong> (Analysis & Writeup)
  </p>

  <div>
    <img src="https://img.shields.io/badge/Apache_Spark-E25A1C?style=for-the-badge&logo=apachespark&logoColor=white" />
    <img src="https://img.shields.io/badge/XGBoost-189FDD?style=for-the-badge&logo=xgboost&logoColor=white" />
    <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" />
    <img src="https://img.shields.io/badge/SDSC_Expanse-005F73?style=for-the-badge" />
    <img src="https://img.shields.io/badge/Status-Complete-success?style=for-the-badge" />
  </div>
</div>

---

<p align="right">
  <a href="#1-introduction">Introduction</a> |
  <a href="#2-figures">Figures</a> |
  <a href="#3-methods">Methods</a> |
  <a href="#4-results">Results</a> |
  <a href="#5-discussion">Discussion</a> |
  <a href="#6-conclusion">Conclusion</a> |
  <a href="#7-statement-of-collaboration">Collaboration</a>
</p>

---

## 1. Introduction

### Why This Project?

The Antarctic Ice Sheet is the primary regulator of the global hydrological cycle and contains enough frozen water to raise global mean sea levels by approximately 58 meters. However, current ice sheet projections are plagued by uncertainty. The West Antarctic Ice Sheet (WAIS), in particular, is losing mass at an accelerating rate primarily due to the intrusion of warm Circumpolar Deep Water (CDW) that drives vigorous sub-ice-shelf basal melting. This basal melt thins the ice shelves, reducing their buttressing effect, and exposes marine-terminating glaciers to Marine Ice Sheet Instability (MISI). MISI acts as a catastrophic positive feedback loop: as the grounding line retreats into deeper waters along the inland sloping bedrock, ice discharge accelerates, and leads to  runaway retreat. Predicting where and when the ice sheet will cross these critical thresholds is essential for accurate climate modelling, coastal planning, and future policy development. Traditional approaches rely on coupled numerical models (such as NEMO or PISM). While rich in detail, these physics-based simulations are computationally expensive, and can be restricted in a temporal or spatial scale. As a result, calculating comprehensive uncertainty estimates or continent-wide probabilistic projections remains a massive computational bottleneck. Our project bypasses these limitations by taking a novel, data-driven "Digital Twin" approach. 

We propose fusing five heterogeneous datasets: ICESat-2(high-resolution laser altimetry), GRACE-FO (gravimetry), Bedmap3 (subglacial topography), and GLORYS12V1 4D ocean thermodynamics, into a unified, ~1.3 billion row feature space. An ML-ready dataset of this scale covering all major Antarctic drainage basins does not currently exist in the published literature and (**IF RESULTS GOOD**) this project represents a shift in computational glaciology. The use of distributed machine learning frameworks via Apache Spark (including SVD/PCA dimensionality reduction and XGBoost classifiers), this project aims to bypass expensive numerical simulations to directly classify basal loss events and identify high-resolution, early-warning signatures of ice sheet instability.

### Why Big Data and Distributed Computing?

This problem requires big data and distributed computing for the following reasons:

1. **Data Volume**: Our fused spatiotemporal dataset comprises 1,386,866,499 rows × 28+ columns, representing over 38.8 billion unique observations spanning 2019 to 2025. The compressed Parquet footprint is ~40 GB. When the data is uncompressed and loaded into memory for matrix operations, this data volume easily tripling the RAM usage on a local environment.

2. **Feature Engineering at Scale**: Computing per-pixel; temporal features , requires window operations partitioned by spatial coordinates. Roughly 2 million unique pixel spaning  just a few years, generates billions of evaluations that must be distributed across executors.

3. **Model Training**: XGBoost's distributed histogram construction and Spark's ML pipeline evaluation on the full dataset would be impractical on a single machine. SDSC Expanse provides 32 cores and 128 GB RAM per node, enabling parallel training across 6 executors.

**What would be impossible without Spark?** The lag feature engineering pipeline alone generates around 14 GB of shuffle data per phase. Without Spark's shuffle architecture and disk spill capability, the processing would require manual batch processing.

### Project Overview

| Aspect | Detail |
|---|---|
| **Target Variable** | `basal_loss_agreement` : dual-sensor binary label (GRACE mass anomaly AND ICESat-2 elevation thinning) |
| **Model 1** | SparkXGBClassifier (distributed gradient boosting) |
| **Model 2** | SVD dimensionality reduction → KMeans clustering → GBTClassifier on principal components |
| **Evaluation** | PR-AUC (primary), ROC-AUC, F1, Precision, Recall |
| **Infrastructure** | SDSC Expanse, Singularity container, Apache Spark 3.x |

---

## 2. Figures

### Data Exploration

> **Figure 2.1:** [INSERT COMMENT HERE]
 
> **Figure 2.2:** [INSERT COMMENT HERE]

> **Figure 2.3:** [INSERT COMMENT HERE]

### SVD / PCA Results (or corrected_stack)

> **Figure 2.4:

### Model Performance

> **Figure 2.5: [INSERT COMMENT HERE]

> **Figure 2.6: [INSERT COMMENT HERE]

### Predictions

> **Figure 2.7: [INSERT COMMENT HERE]

---

## 3. Methods

### 3.1 Data Exploration

The Antarctic fused dataset was constructed by spatially joining five satellite products onto a common 500m Antarctic Polar Stereographic (EPSG:3031) grid:

| Dataset | Measures | Resolution |
|---|---|---|
| ICESat-2 ATL15 | Ice surface elevation change | 1 km |
| GRACE/GRACE-FO | Gravitational mass anomaly | ~27 km |
| Bedmap3 | Sub-surface topography & ice thickness | 500 m |
| GLORYS12V1 | Ocean temperature & salinity (4D) | ~8 km |
| Master Grid | Coordinate reference template | 500 m |

Key EDA findings:
- 28 raw columns, ~1.3 billion rows
- Extreme class imbalance: <1% positive rate for `basal_loss_agreement`
- Ocean features are structurally null for inland pixels (expected)
- Strong multicollinearity within ocean feature group and GRACE feature group

### 3.2 Preprocessing (using Spark)

**Phase 1: Feature Engineering** (`feature_engineering_pipeline.py`)

[INSERT COMMENT HERE]
```python
# [INSERT COMMENT/CODE HERE]
```

**Phase 2: ML Preprocessing**

For XGBoost: [INSERT COMMENT HERE(WHY?!?!?)]

For SVD/KMeans: [INSERT COMMENT HERE]

**Class Imbalance Handling:**
- Region-stratified undersampling (Amundsen: 1:5, Ross/Ronne: 1:12)
- Sample weights combining regional importance and class balance

### 3.3 Model 1: SparkXGBClassifier

[INSERT COMMENT HERE]
```python
# [INSERT COMMENT HERE]
```

Two configurations compared: **Baseline** (above) and **Tuned** (lower LR, explicit L1/L2 regularisation, higher `min_child_weight`).

Temporal train/val/test split: Apr 2020 – Dec 2022 (train), Jan – Oct 2023 (val), Nov 2023+ (test).

### 3.4 Model 2: SVD + KMeans + GBTClassifier

[INSERT COMMENT HERE]
```python
# [INSERT COMMENT HERE]RowMatrix
```

---

## 4. Results

### 4.1 Model 1: XGBoost

| Model | Split | ROC-AUC | PR-AUC | F1 | Precision | Recall |
|---|---|---|---|---|---|---|
| XGB_Baseline | train | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` |
| XGB_Baseline | test | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` |
| XGB_Tuned | train | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` |
| XGB_Tuned | test | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` |
| Physics_Threshold | test | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` |

### 4.2 Model 2: SVD + KMeans + GBT

**SVD Explained Variance:**

| Components | Cumulative Variance |
|---|---|
| Top 5 | `[PLACEHOLDER]`% |
| Top 10 | `[PLACEHOLDER]`% |
| Top 20 | `[PLACEHOLDER]`% |

**KMeans Clustering:**

| Metric | Value |
|---|---|
| Silhouette Score | `[PLACEHOLDER]` |
| Clusters | 6 |

**Supervised on PCs:**

| Model | Split | ROC-AUC | PR-AUC | F1 |
|---|---|---|---|---|
| SVD_GBT | train | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` |
| SVD_GBT | test | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` |

### 4.3 Model Comparison

| Metric | XGB (full) | SVD_GBT (20 PCs) | Physics Baseline |
|---|---|---|---|
| Test ROC-AUC | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` |
| Test PR-AUC | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` |
| Test F1 | `[PLACEHOLDER]` | `[PLACEHOLDER]` | `[PLACEHOLDER]` |

### 4.4 Predictions Analysis

**Confusion Matrix (SVD_GBT on Test Set):**

|  | Predicted Positive | Predicted Negative |
|---|---|---|
| **Actual Positive** | `[PLACEHOLDER]` | `[PLACEHOLDER]` |
| **Actual Negative** | `[PLACEHOLDER]` | `[PLACEHOLDER]` |

---

## 5. Discussion

### 5.1 Model 1 Interpretation

[INSERT COMMENT HERE]

**Fitting diagnosis**: [INSERT COMMENT HERE]

> [!TIP]
> **Key Insight:** [INSERT COMMENT HERE]

### 5.2 Model 2 Interpretation

[INSERT COMMENT HERE]

### 5.3 Shortcomings

We acknowledge several limitations:

1. **Temporal stationarity assumption**: The temporal split assumes that the relationship between features and basal loss is stationary. If MISI acceleration changes the feature-target relationship, the model may degrade on future data.

2. **Label quality**: The `basal_loss_agreement` label requires both GRACE mass anomaly AND ICESat-2 elevation thinning. This is conservative. iIt misses events visible to only one sensor. The AND-logic reduces the positive rate to <1%, creating extreme class imbalance.

3. **Spatial autocorrelation**: Adjacent pixels share features (ice geometry, ocean conditions). Our evaluation does not account for spatial dependence, which may inflate performance estimates.

> [!CAUTION]
> **If results seem too good:** High ROC-AUC (>0.95) with <1% positive rate is common but misleading. The PR-AUC metric is the true test. It penalises models that achieve high accuracy by simply predicting the majority class.

### 5.4 Impact of Distributed Computing

| Operation | Serial Estimate | Distributed (Spark) | Enabled By |
|---|---|---|---|
| Feature engineering (windows) | ~8 hours | ~45 min | Partitioned parallelism |
| XGBoost training | Impossible (OOM) | ~30 min | Barrier-mode histogram construction |
| SVD computation | ~2 hours | ~15 min | `RowMatrix.computeSVD()` distributed Gramian |
| Full pipeline | >12 hours | ~2 hours | DAG lineage truncation + disk spill |

---

## 6. Conclusion

### What We Learned

1. **Big data processing fundamentals**: Managing Spark's DAG lineage is critical at scale. Without Parquet-flush lineage truncation, the feature engineering pipeline exhausted executor memory within the first stage. Understanding Spark's shuffle architecture (partition count, spill strategy) was as important as the ML modelling itself.

2. **Distributed computing changed our approach**: We could iterate on the full 1.3 billion row dataset rather than sampling. This is essential for detecting ~1% positive-rate events that would be lost via random sampling.

3. **Domain knowledge + ML**: The physics-inspired features (ocean heat content proxy, grounding line vulnerability, thermal driving x ice draft) were designed from glaciological understanding. Note that pure data-driven features alone would miss the MISI physics.

### What We Would Do Differently

1. **Start with simpler models**: We should have established a logistic regression baseline on the SVD components first, before jumping to XGBoost. This would have quantified the marginal value of non-linear modelling.

2. **Spatial cross-validation**: Splitting by region instead of by time for validation would better test geographic generalisation.

3. **More aggressive feature selection**: The 55-feature space could likely be pruned to ~20 features without performance loss, reducing training time and overfitting risk.

### What We Would Explore with More Time/Resources

1. **Multi-node training**: Our current setup uses a single 32-core node. Multi-node Spark clusters would enable faster training and larger hyperparameter searches.
2. **Deep learning**: A 1D-CNN or LSTM on the per-pixel time series could capture temporal patterns that tree-based models miss.
3. **Uncertainty quantification**: Bootstrap prediction intervals for risk-critical predictions (e.g., "this pixel has a 95% CI of [0.60, 0.85] for basal loss probability").

---

## 7. Statement of Collaboration

`[PLACEHOLDER — fill in with actual team member names, roles, and contributions]`

**Format:** `Name: Title: Contribution`

- **Scotty Rogers**: Pipeline Architect & Data Engineer: Designed and implemented the end-to-end Spark pipeline including raw data fusion (5 satellite datasets), feature engineering (55+ features), label construction (dual-sensor), XGBoost + SVD/KMeans model training, and HPC deployment on SDSC Expanse. Managed the Singularity container environment and Slurm job scheduling.

- **Hans Hanson**: `[PLACEHOLDER — fill in contribution]`

---

<details>
  <summary><b>Appendix A: How to Reproduce</b></summary>
  <br>

  **Prerequisites:** Access to SDSC Expanse with Singularity, the `spark_py_latest_jupyter_dsc232r.sif` container, and the fused Parquet dataset.

  ```bash
  # Step 1: Feature engineering
  sbatch run_fe.sh

  # Step 2: XGBoost training (Model 1)
  sbatch run_xgb.sh

  # Step 3: SVD + KMeans + GBT (Model 2)
  sbatch run_svd_kmeans.sh
  ```

  Each script runs inside the Singularity container, binds the Lustre project directory, and outputs logs to `*_pipeline_<jobid>.out`.
</details>

<details>
  <summary><b>Appendix B: Mathematical Specification</b></summary>
  <br>

  **Label Construction (Dual-Sensor Agreement):**

  $$\text{label} = \mathbb{1}\left[\text{LWE}_\text{quarterly} < \mu_\text{mascon} - 0.5\sigma_\text{mascon}\right] \wedge \mathbb{1}\left[\Delta h < \mu_\text{regional} - 0.5\sigma_\text{regional}\right]$$

  **SVD Projection:**

  $$Z = X V_k, \quad V_k \in \mathbb{R}^{d \times k}, \quad k = 20$$

  **XGBoost Objective:**

  $$\mathcal{L}(\theta) = \sum_{i=1}^{n} w_i \cdot \ell(y_i, \hat{y}_i) + \Omega(f_t)$$

  where $w_i$ encodes both regional importance and class balance.
</details>

---

<p align="center">
  <i>Repository: <a href="https://github.com/scotty-ucsd/dsc232_group_project">github.com/scotty-ucsd/dsc232_group_project</a></i>
</p>
