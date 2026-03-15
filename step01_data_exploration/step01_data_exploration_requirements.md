## Data Exploration

### 1. SDSC Expanse Environment Setup
* Document your SDSC Expanse setup in your README.md
* Include your SparkSession configuration with justification for your memory/executor settings
* Use the formula: Executor instances = Total Cores - 1 and Executor memory = (Total Memory - Driver Memory) / Executor Instances
    * Example:
        ```python
        # Example: 8 cores, 128GB total memory
        spark = SparkSession.builder \
            .config("spark.driver.memory", "2g") \
            .config("spark.executor.memory", "18g") \
            .config("spark.executor.instances", 7) \
            .getOrCreate()
        ```
* Include a screenshot of your Spark UI showing multiple executors active during data loading

### 2. Data Exploration using Spark
* All data exploration must be done using Spark DataFrames, not Pandas. Use operations like:
    * `df.count()`,`df.describe()`,`df.show()`,`df.printSchema()`
    * `df.groupBy().agg()` for aggregations
    * `df.select().distinct.count()` for unique values
    * **XGBoost:** `xgboost.spark.SparkXGBClassifier`
* Answer the following:
    * How many observations does your dataset have?
    * Describe all columns in your dataset: their scales and data distributions. 
    * Describe categorical and continuous variables. 
    * Describe your target column.
    * Do you have missing and duplicate values in your dataset?
    * For image data: describe number of classes, image sizes, uniformity, cropping/normalization needs.

### 3. Data Plots
* Do the following:
    * Create visualizations using Spark aggregations + matplotlib/plotly (sample data for plotting if needed)
    * Plot your data with various chart types: bar charts, histograms, scatter plots, etc.
    * Clearly explain each plot and what insights it provides
    * For image data: plot example classes

### 4. Preprocessing Plan
* Describe how you will preprocess your data. Only explain—do not perform preprocessing
    * How will you handle missing values?
    * How will you handle data imbalance (if applicable)?
    * What transformations will you apply (scaling, encoding, feature engineering)?
    * What Spark operations will you use for preprocessing?

### 5. Predictions Analysis
* You must use a distributed implementation (Spark MLlib, Spark XGBoost, etc.)
* Simple linear regression is not acceptable unless combined with substantial feature engineering using Spark
* Models that only run on the driver (e.g., scikit-learn on collected data) are not acceptable
* Your training should demonstrate meaningful parallelization across multiple executors
