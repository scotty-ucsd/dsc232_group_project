## Milestone 2

### Data
* [Brief data description here]
* While sample data can be accessed directly in the repo, the full merged dataset can be accessed [here](https://drive.google.com/file/d/1SCAh3grsFHkzpx7UXMOyG7_7V2c_xyG0/view?usp=sharing)

### SDSC Expanse Environment Setup

#### Jupyter Session Details
* Account:
    ```
    TG-SEE260003
    ```
* Partition:
    ```
    shared
    ```
* Time limit (min): 45
    * To get statistics, create plots, generate sample subset for `antarctica_sparse_features.parquet` took ~ 26 minutes 
* Number of cores: 32
    * *note: see next section for more details*
* Memory required per node (GB): 128
    * *note: see next section for more details*
* Singularity Image File Location:
    ```
    ~/esolares/spark_py_latest_jupyter_dsc232r.sif
    ```
* Environment Modules to be loaded:
    ```
    singularitypro
    ```
* Working Directory:
    ```
    home
    ```
* Type:
    ```
    JupyterLab
    ```

#### SparkSession Configuration
* Configuration Details:
    ```python
    spark = (
        SparkSession.builder
        .appName("HPC_Antarctic_Unified_EDA_Pipeline")
        .config("spark.driver.memory",            driver_mem)
        .config("spark.executor.instances",        str(EXECUTOR_INSTANCES))
        .config("spark.executor.cores",            str(EXECUTOR_CORES))
        .config("spark.executor.memory",           exec_mem)
        .config("spark.sql.shuffle.partitions",    str(SHUFFLE_PARTITIONS))
        # --- Driver result budget ---
        .config("spark.driver.maxResultSize",      "4g")
        # --- Network stability for large scans ---
        .config("spark.network.timeout",           "1200s")
        # --- Parallel partition discovery for deep dirs ---
        .config("spark.sql.sources.parallelPartitionDiscovery.threshold", "32")
        .config("spark.sql.sources.parallelPartitionDiscovery.parallelism", "64")
        # --- Adaptive Query Execution ---
        .config("spark.sql.adaptive.enabled",                      "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled",   "true")
        .config("spark.sql.adaptive.advisoryPartitionSizeInBytes", "128m")
        # --- Parquet pushdown & vectorisation ---
        .config("spark.sql.parquet.filterPushdown",                "true")
        .config("spark.sql.parquet.mergeSchema",                   "false")
        # --- Disk spill safety : use Lustre scratch, NOT /tmp ---
        .config("spark.local.dir",
                os.environ.get("TMPDIR",
                               os.path.join(os.getcwd(), "spark_scratch")))
        .getOrCreate()
    )
    ```
* Spark SDSC Resource Reasoning:
    * 1. Cores
        * Recall we selected 32 for the *Number of cores* in our SDSC setup
        * `spark.executor.instances` is `6` and `spark.executor.cores` is `5`, so this results in `30` cores being used for our executor and the remaining `2` cores left for our driver
    * 2. Memory
        * Recall we selected 128 GB for the *Memory required per node* in our SDSC setup
        * `spark.executor.memory` is `19` GB and we have a total of `6` `spark.executor.instances`, this results in using `114` GB
        * `spark.driver.memory` is set to `10` GB, so that brings the total up to `124` GB and leaves a safety buffer of `4`GB for the VM operating system.

### Data Exploration
#### Spark methods
* `df.schema` was used to find column names and data types
* `df.count()` was used to find the number of rows
* `len(df.columns)` was used to find the number of columns
* `df.agg()` with SQL functions was used to find min,max,mean,standard deviation

#### EDA Results
* `antarctica_sparse_features.parquet` is a massive ~40GB compressed parquet file
    * Total Number of Columns: 28
    * Total Number of Rows: 1,386,866,499
    * Total Observations: 38,832,261,972

#### Data Polts

### Preprocessing Plan


