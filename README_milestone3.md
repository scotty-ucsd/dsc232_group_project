

# Milestone 3
We composed this readme file specifically for Milestone 3 and for ease of evaluation, not as a complete one for the whole project and final submission.
However, we have updated the complete readme file with our new work and all changes, per Milestone 3 instructions, and have linked to it [here](https://github.com/scotty-ucsd/dsc232_group_project/tree/main/README.md) and below.  Thus, if desired, one may use the complete readme file instead of this one to evaluate our Milestone 3 submission.

## Predicting Antarctic Melting

* This project explores a high-resolution, multimodal dataset fusing laser altimetry, gravity fields, ocean thermodynamics, and sub-glacial topography across the Antarctic continent (2020–2025)
* The fused dataset is intended to be a "physics-ready" feature space for machine learning models predicting ice sheet instability.
* We are employing classification models to predict whether the datasets agree upon major mass anomalies ("basal_loss_agreement").

## Previous (Milestone 2 ) Links
- [Dataset](README_milestone2.md#dataset)
- [GitHub Repository Setup](README_milestone2.md#github-repository-setup)
- [SDSC Expanse Environment Setup](README_milestone2.md#sdsc-expanse-environment-setup)
    - [Jupyter Session Details](README_milestone2.md#jupyter-session-details)
    - [SparkSession Configuration](README_milestone2.md#sparksession-configuration)
- [Columns Overview](README_milestone2.md#columns-overview)

## Milestone 3 Contents
- [Completion of Major Preprocessing](#completion-of-major-preprocessing)
    - [Link to Code and Slurm Script for Preprocessing](#link-to-code-and-slurm-script-for-preprocessing)
- [Training of First Model](#training-of-first-model)
- [Evaluation of First Model Comparing Training vs. Test Error](#evaluation-of-first-model-comparing-training-vs-test-error)
- [Where First Model Fits in Fitting Graph](#where-first-model-fits-in-fitting-graph)
- [Next Models Under Consideration and Explanations](#next-models-under-consideration-and-explanations)
- [All Updates, Including to Final Submission README.md](#all-updates-including-to-final-submission-readmemd)
- [Conclusion of First Model and Possible Improvements](#conclusion-of-first-model-and-possible-improvements)


## Completion of Major Preprocessing
A number of preprocessing steps had already been performed for the creation of this dataset (i.e. before Milestone 2), since it was fused together from five sources (see [here](https://github.com/scotty-ucsd/dsc232_group_project/tree/Milestone2/pre_pre_processing_pipeline/docs/COMPREHENSIVE_EDA_AND_PREPROCESSING.md) for details on this "pre-pre-processing").  So the following section will focus on the feature engineering, scaling, imputing, and encoding completed for Milestone 3.

## Features Engineering

### 1.1 What Features Were Engineered and Why

The pipeline engineers **64+ features** across five categories, all motivated by glaciological physics:

| Category | Example Features | Rationale |
|---|---|---|
| Static geometry | `draft_x_thermal_access`, `grounding_line_vulnerability`, `retrograde_flag` | Encode ice-ocean exposure and marine instability geometry |
| Dynamic pixel-level | `pixel_mean_delta_h`, `delta_h_deviation`, `surface_slope_change` | Capture per-pixel thinning trends via expanding windows |
| Ocean interactions | `thermal_driving_x_draft`, `thermal_anomaly`, `salinity_stratification_proxy`, `lwe_trend` | Encode warm-water intrusion and gravitational mass signals |
| Context / cyclical | `sin_month`, `cos_month`, `mascon_mean_delta_h`, `regional_delta_h_percentile` | Seasonal cycles + neighborhood context without leakage |
| Temporal memory | `t_star_6mo_avg`, `delta_h_rate`, `delta_h_momentum`, `delta_h_acceleration` | 6-month rolling averages, momentum, and acceleration for trend detection |
| Physics interactions | `ocean_heat_content_proxy`, `draft_ratio`, `thermal_x_gl_proximity`, `freezing_departure`, `bed_geometry_risk`, `mass_flux_proxy` | Hand-crafted domain features combining ice draft, ocean temperature, and bed topography |

### 1.2 Scaling / Transforming

Three distinct scalers are used across the three preprocessing pipelines:

```python
# XGBoost pipeline: MinMaxScaler (L1146)
MinMaxScaler(inputCol="raw_features", outputCol="features")

# Classic pipeline: StandardScaler with mean centering (L1292-1293)
StandardScaler(inputCol="raw_features", outputCol="features",
               withMean=True, withStd=True)

# Stack pipeline: L2 Normalizer (L1228)
Normalizer(inputCol="raw_features", outputCol="features", p=2.0)
```

### 1.3 Imputing

Every pipeline uses `Imputer` with median strategy as its first stage:

```python
# L1136  (identical pattern in all three pipelines)
Imputer(strategy="median", inputCols=numeric, outputCols=imputed)
```

This is critical because the Antarctic dataset has significant sparsity(i.e., ocean variables are `null` for inland pixels, and temporal features have nulls at series boundaries).

### 1.4 Encoding

```python
# StringIndexer: region name -> integer index (L1140-1141)
StringIndexer(inputCol="regional_subset_id",
              outputCol="region_index", handleInvalid="keep")

# OneHotEncoder: grounding-line distance buckets + region (L1142-1143)
OneHotEncoder(inputCol="gl_bucket_idx", outputCol="gl_bucket_ohe")
OneHotEncoder(inputCol="region_index", outputCol="region_ohe")

# Bucketizer: continuous distance -> categorical bins (L1137-1139)
Bucketizer(splits=[-inf, 5000, 20000, 50000, 100000, inf],
           inputCol="dist_to_grounding_line",
           outputCol="gl_bucket_idx", handleInvalid="keep")

# VectorAssembler: combine all features into a single vector (L1144-1145)
VectorAssembler(inputCols=imputed + ["gl_bucket_ohe", "region_ohe"],
                outputCol="raw_features", handleInvalid="skip")
```

### 1.5 Feature Engineering (Spark SQL + Advanced Transformers)

```python
# Window functions: pixel-level expanding mean (L570-584)
pixel_time_w = Window.partitionBy("x", "y").orderBy("month_idx")
    .rowsBetween(Window.unboundedPreceding, 0)
df = df.withColumn("pixel_mean_delta_h", F.avg("delta_h").over(pixel_time_w))

# PolynomialExpansion on physics triple (L1276-1279, classic pipeline)
PolynomialExpansion(degree=2, inputCol="poly_input",
                    outputCol="poly_features")
# Inputs: t_star_mo x ice_draft x dist_to_grounding_line

# PCA on correlated ocean variables (L1194-1198)
PCA(k=4, inputCol="ocean_vec", outputCol="ocean_pca")
# Groups: thetao_mo, t_star_mo, so_mo, t_f_mo + quarterly aggregates

# Cyclic encoding via Spark SQL (L640-649)
df = df.withColumn("sin_month", F.sin(F.col("month_of_year") * (2*pi/12)))
       .withColumn("cos_month", F.cos(F.col("month_of_year") * (2*pi/12)))
```

### 1.6 Feature Table

| Feature | Description | Why | Spark MLlib Transformers | Spark DataFrame Ops |
|---|---|---|---|---|
| `gl_bucket_ohe` | Grounding-line distance bins | Captures proximity thresholds for warm-water access | `Bucketizer` $\rightarrow$ `OneHotEncoder` | n/a |
| `region_ohe` | One-hot region encoding | Regional ice dynamics vary fundamentally | `StringIndexer` $\rightarrow$ `OneHotEncoder` | n/a |
| features (scaled) | Min-max scaled feature vector | Normalises heterogeneous scales for XGBoost | `Imputer` $\rightarrow$ `VectorAssembler` $\rightarrow$ `MinMaxScaler` | n/a |
| `ocean_pca` | PCA(k=4) of 8 ocean variables | Decorrelates collinear ocean measurements | `VectorAssembler` $\rightarrow$ `PCA` | n/a |
| `poly_features` | Degree-2 expansion of physics triple | Captures nonlinear interactions | `VectorAssembler` $\rightarrow$ `PolynomialExpansion` | n/a |
| `pixel_mean_delta_h` | Expanding-window mean of ice-height change | Tracks cumulative thinning per pixel | n/a | `F.avg().over(Window)` |
| `thermal_driving_x_draft` | `t_star_mo x ice_draft` | Ocean thermal forcing x ice exposure | n/a | `F.col() * F.col()` |
| `sin_month` / `cos_month` | Cyclic month encoding | Preserves seasonal continuity | n/a | `F.sin()`, `F.cos()` |
| `delta_h_momentum` | 1-month lag difference | Rate of thinning acceleration | n/a | `F.lag().over(Window)` |
| `regional_delta_h_percentile` | Z-score within region-month | Where this pixel ranks relative to regional peers | n/a | `groupBy` + `broadcast join` |

---


### Link to Code and Slurm Script for Preprocessing
[Here](ml_prep/feature_engineering_pipeline.py) is a link to our full code for feature engineering and other preprocessing steps, which is kept in a separate python file and can be run independently via [slurm script](ml_prep/run_fe.sh).

### [Back to Milestone 3 Contents](#milestone-3-contents)
## Training of First Model

For Milestone 3, we will treat a tuned XGB model as our "first model", comparing it to a baseline version.  (In reality this model is more like an intermediate model, since we tried some simple ones beforehand and are already moving to other ones.)

**`SparkXGBClassifier`** was selected because:

1. **Native Spark integration:** `xgboost.spark.SparkXGBClassifier` runs inside the Spark execution engine, distributing histogram construction across workers.
2. **Threshold-like physics:** XGBoost's gradient-boosted decision trees excel at learning threshold-based decision boundaries (e.g., "if thermal forcing > X AND grounding-line distance < Y $\rightarrow$ basal melt"). This matches the physics of ice-sheet basal melt.
3. **Handles imbalanced data:** Via `weightCol` support and `logloss` evaluation metric.
4. **Scale:** 400M rows, 72 columns, 40+ GB: only a distributed framework is feasible.

#### Spark Configuration (HPC Mode)

```python
# L354-373 : HPC config
builder = (
    builder
    .master(master)                               # local[24] or YARN
    .config("spark.executor.instances", "6")       # 6 executor JVMs
    .config("spark.executor.cores", "4")           # 4 cores each = 24 total
    .config("spark.executor.memory", "20g")         # 20 GB per executor
    .config("spark.yarn.executor.memoryOverhead", "10g")
    .config("spark.driver.memory", "80g")
    .config("spark.sql.shuffle.partitions", "800")  # high parallelism
    .config("spark.memory.fraction", "0.5")
    .config("spark.memory.storageFraction", "0.1")  # favour execution over caching
)
```

Since the Milestone 3 instructions for this section are simply to train the model, in the remainder of this section we will simply provide references and links to the relevant code sections.  Our evaluation, discussion, etc. appear further below in the relevant sections.

All of the functions referenced below appear in [ml_pipeline.py](https://github.com/scotty-ucsd/dsc232_group_project/blob/Milestone3/ml_prep/ml_pipeline.py).
* There is a global dictionary [TRAIN_MAP](https://github.com/scotty-ucsd/dsc232_group_project/blob/Milestone3/ml_prep/ml_pipeline.py#L1976) that stores the functions for training various models.
* A wrapper function [run_all](https://github.com/scotty-ucsd/dsc232_group_project/blob/Milestone3/ml_prep/ml_pipeline.py#L2255) refers to this dictionary and trains the relevant model.

* The specific function for training the XGB models is the [train_xgb](https://github.com/scotty-ucsd/dsc232_group_project/blob/Milestone3/ml_prep/ml_pipeline.py#L1678) model.
  * This function calls a [build_xgb_preprocessing](https://github.com/scotty-ucsd/dsc232_group_project/blob/Milestone3/ml_prep/ml_pipeline.py#L1129) function before training.

* After the training is completed and results are returned, the [run_all](https://github.com/scotty-ucsd/dsc232_group_project/blob/Milestone3/ml_prep/ml_pipeline.py#L2255) function calls two evaluative functions, namely [print_results_table](https://github.com/scotty-ucsd/dsc232_group_project/blob/Milestone3/ml_prep/ml_pipeline.py#L1996) and [fitting_analysis](https://github.com/scotty-ucsd/dsc232_group_project/blob/Milestone3/ml_prep/ml_pipeline.py#L2019).


### [Back to Milestone 3 Contents](#milestone-3-contents)
## Evaluation of First Model Comparing Training vs. Test Error


| Model | Split | ROC-AUC | PR-AUC | F1 | Precision | Recall |
|---|---|---|---|---|---|---|
| **XGB_Baseline** | train | 0.9304 | 0.5103 | 0.7512 | 0.9279 | 0.6841 |
| **XGB_Baseline** | val | 0.6969 | 0.0475 | 0.9700 | 0.9636 | 0.9776 |
| **XGB_Baseline** | test | 0.7403 | 0.0914 | 0.9440 | 0.9346 | 0.9558 |
| **XGB_Tuned** | train | 0.9422 | 0.5655 | 0.7667 | 0.9292 | 0.7037 |
| **XGB_Tuned** | val | 0.6957 | 0.0430 | 0.9705 | 0.9630 | 0.9794 |
| **XGB_Tuned** | test | 0.7490 | 0.0900 | 0.9451 | 0.9343 | 0.9608 |


### Metric Selection Justification

- **PR-AUC (primary):** With a ~3% positive rate (`rate=0.030064` in train), ROC-AUC is **inflated by the massive true-negative count**. PR-AUC directly measures the precision-recall trade-off for the rare positive class. A PR-AUC of 0.09 is more honest than a ROC-AUC of 0.74: it says the model finds very few true positives without false alarms.
- **ROC-AUC:** Used for overfitting diagnosis (train-test gap) but misleading as a standalone quality metric for imbalanced problems.
- **F1, Precision, Recall:** Complement threshold-dependent analysis. The high F1/Precision/Recall on val/test are artifacts of the majority-class dominance at the default 0.5 threshold.

### Ground Truth vs. Predictions (Regional Breakdown)

**XGB_Baseline: Test Set:**

| Region | Predicted Rate | True Rate | n |
|---|---|---|---|
| amundsen_sea | 0.0029 | 0.1155 | 10,869,309 |
| antarctic_peninsula | 0.0870 | 0.0396 | 14,578,163 |
| lambert_amery | 0.0003 | 0.0099 | 27,227,872 |
| ronne | 0.0003 | 0.0224 | 23,682,149 |
| ross | 0.0011 | 0.0440 | 34,371,219 |
| totten_and_aurora | 0.0010 | 0.0312 | 15,988,044 |

> **Critical failure in Amundsen Sea:** The model predicts a 0.29% positive rate against a true rate of 11.55%: a **40x underestimation** of the most rapidly melting region in Antarctica. The model appears to have learned that "most pixels are negative" and fails to flag the high-loss region. This is the most scientifically important region and must be addressed in Milestone 4.

---



### [Back to Milestone 3 Contents](#milestone-3-contents)
## Where First Model Fits in Fitting Graph

**Both models are OVERFITTING:**

```
XGB_Baseline:
  Train ROC-AUC: 0.9304
  Test  ROC-AUC: 0.7403  (gap: 0.1902)
  Test  PR-AUC:  0.0914
  $\rightarrow$ OVERFITTING

XGB_Tuned:
  Train ROC-AUC: 0.9422
  Test  ROC-AUC: 0.7490  (gap: 0.1932)
  Test  PR-AUC:  0.0900
  $\rightarrow$ OVERFITTING
```

The ~19% ROC-AUC gap between train and test is a clear overfitting signal. The model memorises training patterns (especially after undersampling alters the class distribution) that do not generalise to the temporal hold-out.

**Why overfitting occurs:**
1. **Temporal distribution shift:** Training data (Apr 2020 – Dec 2022) does not contain the climatic patterns present in the test period (post-Oct 2023). Ice-sheet dynamics are non-stationary.
2. **Undersampling artefact:** The 1:10 ratio amplifies positive examples, which the model can memorise since many are spatially clustered in a few glaciers.
3. **High model capacity:** `max_depth=4-8`, 100–400 trees is sufficient capacity to overfit spatial patterns.


| Hyperparameter | XGB_Baseline | XGB_Tuned |
|---|---|---|
| `max_depth` | 4 | 8 |
| `n_estimators` | 100 | 400 |
| `learning_rate` | 0.1 | 0.02 |
| `subsample` | 0.8 | 0.75 |
| `colsample_bytree` | 0.8 | 0.7 |
| `min_child_weight` | 10 | 20 |
| `reg_alpha` | n/a | 0.1 |
| `reg_lambda` | n/a | 1.0 |

**Key differences in tuning rationale:**
- **Lower learning rate (0.02) + more trees (400):** Slower, more incremental gradient steps for finer convergence.
- **Higher `min_child_weight` (20):** Requires more samples per leaf $\rightarrow$ reduces overfitting to small clusters.
- **L1/L2 regularisation added:** `reg_alpha=0.1` (L1) and `reg_lambda=1.0` (L2) explicitly penalise complexity.
- **Lower `subsample` and `colsample_bytree`:** More aggressive bagging $\rightarrow$ reduces variance.

**Result:** XGB_Tuned achieves marginally better test ROC-AUC (0.749 vs 0.740) but **worse test PR-AUC** (0.090 vs 0.091). The overfitting gap actually *increased* slightly (0.193 vs 0.190). The tuned hyperparameters successfully increased train performance but the regularisation was insufficient to close the generalisation gap.

* **Note on the current [XGB_CONFIGS](https://github.com/scotty-ucsd/dsc232_group_project/blob/Milestone3/ml_prep/ml_pipeline.py#L212):** Both produced similar overfitting patterns.

#### Which Model Performs Best?

**XGB_Baseline is marginally better** on the metric that matters (PR-AUC: 0.091 vs 0.090), despite worse ROC-AUC. The tuned model's deeper trees and more iterations increased memorisation without improving rare-event detection. For a geoscience application where false negatives in Amundsen Sea cost real predictive value, neither model is adequate.


### [Back to Milestone 3 Contents](#milestone-3-contents)
## Next Models Under Consideration and Explanations

The next models under consideration are stacked meta-learner ones that use Random Forest (RF) and Gradient Boosted Trees (GBT) as base learners.

We have done some preliminary training and testing on a XGB meta-learner model, but it seems to overfit as well.

So the current model under consideration and our plan for the next steps is as follows:

1. **Corrected Stacking (Model 4):** Uses **out-of-fold (OOF) training** with pruned meta-features and `LogisticRegression` as the meta-learner. This directly attacks the overfitting problem by:
   - Training base learners on fold A, predicting on fold B
   - Using only 8 meta-features (3 base predictions + 5 context cols) instead of all 64+
   - Using a low-capacity linear meta-learner (LR) instead of XGBoost

2. **Region-aware resampling:** The regional breakdown shows the model completely ignores Amundsen Sea. A **region-stratified undersampling** strategy that ensures proportional representation of each region's positive class would help.

3. **Temporal cross-validation:** Instead of a single temporal split, use **expanding-window CV** (train on months 1–N, validate on N+1–N+3) to capture seasonal and multi-year patterns.

4. **Threshold tuning:** The default 0.5 classification threshold is suboptimal for 3% positive rate. A **PR-curve-optimised threshold** (e.g., maximising F1 on the PR curve) would significantly improve operational predictions.


### [Back to Milestone 3 Contents](#milestone-3-contents)
## All Updates, Including to Final Submission README.md

- Fixed problems identified in Milestone 2 checkpoint [here](README_milestone2.md#preprocessing-plan), including 2019 spike in GRACE-FO `lwe_fused` variable.  In the case of the latter problem, we've decided the problem was filling null values with zero.  Since we have plenty of observations, the easiest solution is to simply drop 2019.
- Organized various pipeline files for the different models (see ml_prep folder [here](https://github.com/scotty-ucsd/dsc232_group_project/blob/Milestone3/ml_prep)).
- Created numerous slurm scripts to streamline training (see .sh files in ml_prep folder above)
- Updating [README.md](https://github.com/scotty-ucsd/dsc232_group_project/tree/main/README.md) for overall project, which will be used for final submission 

### [Back to Milestone 3 Contents](#milestone-3-contents)
## Conclusion of First Model and Possible Improvements

To summarize, our first model(s) are overfitting the data, we think largely because of spareness and class imbalance in terms of regions (e.g. Amundsen sea is melting rapidly while other regions are not, have more ocean data, etc.)  So we are hoping that the stacked meta-learner model, re-sampling, temporal cross-validation, and threshold tuning described above will generalize better.


### [Back to Milestone 3 Contents](#milestone-3-contents)