## Second Model Requirements

### 1. Train Your Second Model using Dimensionality Reduction
* Your second model must include unsupervised learning for dimensionality reduction, followed by additional analysis( **Technique:** Implementation ):
    * **PCA(Principal Component Analysis):** `pyspark.ml.feature.PCA` or manual implementation with Spark
    * **SVD (Singular Value Decomposition):**`pyspark.mllib.linalg.distributed.RowMatrix.computeSVD`
* Follow dimensionality reduction with one of:
    * **Clustering:** K-Means, GMM, or other clustering on reduced features
    * **Visualization & Interpretation:** Eigenvalue analysis, explained variance plots, component interpretation
    * **Supervised Model:** Train a model on the reduced-dimension features

### 2. Evaluate Your Model 
* Compare training vs. test performance
* Analyze explained variance (for PCA/SVD)
* Evaluate clustering quality (silhouette score, etc.) if applicable

### 3. Fitting Analysis
* Answer the following:
    * Where does your model fit in the fitting graph?
    * What are potential future improvements or next models?
    * How does dimensionality reduction affect your results compared to the full feature set?

### 4. Conclusion Section
* What is the conclusion of your 2nd model? 
* What can be done to improve it?
* Note: The conclusion should be its own independent section. 
* Methods will have models 1 and 2, Conclusion will have results and discussion for both.

### 5. Predictions Analysis
* Provide predictions showing correct classifications, false positives (FP), and false negatives (FN) from your test dataset.
