## Pre-Processing and First Model Requirements

### 1. Complete Preprocessing using Spark 
* Finish major preprocessing using Spark DataFrame operations or Spark MLlib transformers: 
    * **Scaling/Transforming:** Use `StandardScaler`, `MinMaxScaler`, or `Normalizer` from `pyspark.ml.feature`
    * **Imputing:** Use `Imputer` for handling missing values
    * **Encoding:** Use `StringIndexer`, `OneHotEncoder`, or `VectorAssembler`
    * **Feature Engineering:** Create new features using Spark SQL functions, `PolynomialExpansion`, or custom transformations


### 2. Train First Distributed Model 
* Train your first model using one of the following distributed implementations( **Model:** Implementation ):
    * **Decision Trees:**`pyspark.ml.classification.DecisionTreeClassifier`
    * **Random Forests:** `pyspark.ml.classification.RandomForestClassifier`
    * **Gradient Boosted Trees:** `pyspark.ml.classification.GBTClassifier`
    * **XGBoost:** `xgboost.spark.SparkXGBClassifier`
* Model must run on SDSC Expanse (not locally)
* Training must use multiple executors/workers
* Evaluate your model: compare training vs. test error
* For supervised learning: include example ground truth and predictions for train, validation, and test sets

### 3. Fitting Analysis
* Answer the following:
    * Where does your model fit in the fitting graph (underfitting vs. overfitting)?
    * Build at least one model with different hyperparameters and compare results
    * Which model performs best and why?
    * What are the next models you are thinking of for Milestone 4 and why?
    
    
### 4. Important Notes
* What is the conclusion of your 1st model? 
* What can be done to improve it?
* How did distributed computing help with this task?

### 5. Predictions Analysis
* You must use a distributed implementation (Spark MLlib, Spark XGBoost, etc.)
* Simple linear regression is not acceptable unless combined with substantial feature engineering using Spark
* Models that only run on the driver (e.g., scikit-learn on collected data) are not acceptable
* Your training should demonstrate meaningful parallelization across multiple executors
