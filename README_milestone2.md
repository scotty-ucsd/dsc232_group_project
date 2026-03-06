

# Milestone 2


## Antarctica's Digital Twin: Exploratory Data Analysis (EDA)

* This project explores a high-resolution, multimodal dataset fusing laser altimetry, gravity fields, ocean thermodynamics, and sub-glacial topography across the Antarctic continent (2019–2025)
* The goal is to provide a "physics-ready" feature space for machine learning models predicting ice sheet instability.

## Dataset
The dataset is the combination of 5 heterogeneous Antartic datasets:

| Dataset | What It Measures | Native Format | Resolution |
|---|---|---|---|
| **ICESat-2 ATL15** | Ice surface elevation change | NetCDF (HDF5 groups, 4 tiles) | 1 km |
| **GRACE/GRACE-FO** | Gravitational mass anomaly | NetCDF (lat/lon, 0.25°) | ~27 km |
| **Bedmap3** | Subsurface topography & ice thickness | NetCDF | 500 m |
| **GLORYS12V1** | Ocean temperature & salinity (4D) | NetCDF (lat/lon/depth/time) | 1/12° (~8 km) |
| **Master Grid** | Coordinate reference template | Created by pipeline | 500 m |

The unification of these datasets is reviewed below in "Preprocessing Plan" and described in detail [here](https://github.com/scotty-ucsd/dsc232_group_project/tree/Milestone2/pre_pre_processing_pipeline/docs/COMPREHENSIVE_EDA_AND_PREPROCESSING.md) (also linked below).



## GitHub Repository Setup
- GitHub IDs for Scotty Rogers and Hans Hanson: *scotty-ucsd* and *hanspeder*
- Public GitHub Repository for this project: https://github.com/scotty-ucsd/dsc232_group_project
    - Scotty Rogers and Hans Hanson are collaborators
- Links to data
    - Full dataset on SDSC and also available [here](https://drive.google.com/file/d/1SCAh3grsFHkzpx7UXMOyG7_7V2c_xyG0/view?usp=sharing)
    - Sample data can be accessed directly in this repo




## SDSC Expanse Environment Setup

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

Screenshot of Spark UI showing multiple executors active during data loading:

!['sparkUI_screenshot'](screenshot.png)

## Data Exploration using Spark
#### Spark methods
* `df.schema` was used to find column names and data types
* `df.count()` was used to find the number of rows
* `len(df.columns)` was used to find the number of columns
* `df.agg()` with SQL functions was used to find min,max,mean,standard deviation

#### EDA Results
How many observations does this dataset have?
* `antarctica_sparse_features.parquet` is a massive ~40GB compressed parquet file
    * Total Number of Columns: 28
    * Total Number of Rows: 1,386,866,499
    * Total Observations: 38,832,261,972

#### Columns Overview

| # | Column | Type | Dims | Description | Source |
|---|---|---|---|---|---|
| 1 | `y` | float64 | — | EPSG:3031 northing [m] | Coordinates |
| 2 | `x` | float64 | — | EPSG:3031 easting [m] | Coordinates |
| 3 | `exact_time` | timestamp | — | ICESat-2 observation timestamp | ICESat-2 |
| 4 | `month_idx` | int32 | — | Year×12+Month (partition key) | Derived |
| 5 | `mascon_id` | int32 | — | GRACE mascon identifier | GRACE/Master map |
| 6 | `surface` | float32 | — | Ice surface elevation [m] | Bedmap3 |
| 7 | `bed` | float32 | — | Bedrock elevation [m] | Bedmap3 |
| 8 | `thickness` | float32 | — | Ice thickness [m] | Bedmap3 |
| 9 | `bed_slope` | float32 | — | \|∇(bed)\| [m/m] | Spatial features |
| 10 | `dist_to_grounding_line` | float32 | — | Distance to grounding line [m] | Spatial features |
| 11 | `clamped_depth` | float32 | — | Draft depth clamped to ocean floor [m] | Ocean |
| 12 | `dist_to_ocean` | float32 | — | Distance to nearest ocean pixel [m] | Ocean |
| 13 | `ice_draft` | float32 | — | Ice base depth below sea level [m] | Ocean |
| 14 | `delta_h` | float32 | t | Elevation anomaly [m] | ICESat-2 |
| 15 | `ice_area` | float32 | t | Fractional ice coverage | ICESat-2 |
| 16 | `surface_slope` | float32 | t | \|∇(h_dynamic)\| [m/m] | Spatial features |
| 17 | `h_surface_dynamic` | float32 | t | surface + delta_h [m] | Spatial features |
| 18 | `thetao_mo` | float32 | t | Monthly avg ocean temp [°C] | Ocean |
| 19 | `t_star_mo` | float32 | t | Monthly avg thermal driving [°C] | Ocean |
| 20 | `so_mo` | float32 | t | Monthly avg salinity [PSU] | Ocean |
| 21 | `t_f_mo` | float32 | t | Monthly avg freezing point [°C] | Ocean |
| 22 | `t_star_quarterly_avg` | float32 | t | 3-month rolling avg T* [°C] | Ocean |
| 23 | `t_star_quarterly_std` | float32 | t | 3-month rolling stddev T* | Ocean |
| 24 | `thetao_quarterly_avg` | float32 | t | 3-month rolling avg θ [°C] | Ocean |
| 25 | `thetao_quarterly_std` | float32 | t | 3-month rolling stddev θ | Ocean |
| 26 | `lwe_mo` | float32 | t | Monthly GRACE LWE [m] | GRACE |
| 27 | `lwe_quarterly_avg` | float32 | t | 3-month rolling avg LWE [m] | GRACE |
| 28 | `lwe_quarterly_std` | float32 | t | 3-month rolling stddev LWE | GRACE |
| 29 | `lwe_fused` | float32 | t | ABS-weighted pixel-level mass [m] | Fusion |
| 30 | `month_idx` | int32 | — | Partition key (Year×12+Month) | Derived |

> **Key**: "t" in the Dims column indicates the column varies with time (per ICESat-2 observation epoch).



## Data Plots

*Plots/visualizations appear first, explanations or comments below each*

!['Histograms'](eda_sdsc/data/eda_plots/fig_03_histograms_antarctica_sparse_features.png)

* These histograms show the distribution of values for each column.  The first two are as one would expect: lots of surface slope values near zero (i.e. near ocean) and a normal distribution for bedrock.  The ice thickness distribution may be due to underlying bedrock and/or water.  The elevation change distribution (delta_h) shows that the changes are relatively small.

!['Correlations'](eda_sdsc/data/eda_plots/fig_04_correlation_antarctica_sparse_features.png)

* Correlation between features in the fused dataset.  These are inuitive: the ice surface elevation is very strongly correlated with ice thickness and, to a less extend, bedrock elevation. 

!['Box Plots/Distributions'](eda_sdsc/data/eda_plots/fig_05_physical_ranges.png)

* These again show distribution of values for these columns.  Note the wide ranges of values for surface and bedrock elevation and ice thickness vs. the much smaller ranges for ocean temperature (thetao_mo), salinity (so_mo), monthly freezing point (t_f_mo), and thermal driving (t_star).

!['Missing Values'](eda_sdsc/data/eda_plots/fig_06_null_structure.png)

* Missing data in for ocean is expected, there is a limited area where it is defined (essentially where ocean touches ice)
* ~8% missing values from ICESat-2 data (dhdt_lag1) is due to missing dates in netCDF files

!['Ice Mask & Ocean Data Coverage'](eda_sdsc/data/eda_plots/fig_07_ice_mask_ocean_coverage.png)

* This is an exploratory test of whether the ice_mask (i.e. whether the ice is grounded or floating) and the (monthly) ocean temperature (thetao_mo) values seem plausible.  Both do: the floating ice appears on coastlines and bays/inlets, as opposed to in the middle of the continent, and this corresponds with where we have valid ocean temperature values.

!['Height Change & Mass Anomaly'](eda_sdsc/data/eda_plots/fig_08_delta_h_vs_lwe_spatial.png)

* This is another exploratory test of whether the height change and liquid water values seem plausible.  Again, both do: the greatest height changes appear near the coasts, as opposed to inland, as well as the liquid water values.

!['Mean Antarctic Ice Height Change, by Month'](eda_sdsc/data/eda_plots/fig_09_delta_h_timeseries.png)

* We're going to investigate this upward trend a bit more, since it may seem counter-intuitive at first glance.  It may be that increased melting in the Antarctic is resulting in greater changes in ice thickness, even if mean ice thickness is decreasing (see graph below).

!['Mean Antarctic Ice Height (LWE), by Month'](eda_sdsc/data/eda_plots/fig_10_lwe_timeseries.png)

* Whereas the previous visualization showed ice height change, this just shows the average ice height.  Intuitively, it has been decreasing over time.



## Preprocessing Plan

- We realize that we are supposed to describe our plan for, not perform, preprocessing here, but a significant amount of "pre-pre-processing" was required to fuse the above datasets.  As mentioned in our project abstract, one of our group members, Scotty Rogers, is familiar and works professionally with satellite data at Los Alamos National Laboratory.  Motivated by his own interest in the Antarctic-related satellite data, he fused the above datasets to create a unified one suitable for this project.  This was a major effort in itself and required numerous pre-processessing steps, such as reconciling the differing spatial resolutions between the datasets (500m, 1km, 8km, and 27km) as well as two different coordinate systems, EPSG: 4326 (geographic) and EPSG:3031 (Antarctic Polar Stereographic).

* A detailed explanation and summary of this "pre-pre-processing pipeline" can be found [here](https://github.com/scotty-ucsd/dsc232_group_project/tree/Milestone2/pre_pre_processing_pipeline/docs/COMPREHENSIVE_EDA_AND_PREPROCESSING.md).


* Fix ice area conservation
    * Comment: Currently the *pre-preprocessing* uses a bilinear interpolation from `xarray` to fill ice area when downscaling.
        * Inplace of the bilinear interpolation we will divide the ice area and distribute equally to each 500m cell
* Investigate 2019 lwe_fused point
    * Comment: As seen in the EDA step there is a large spike in 2019 for the GRACE-FO `lwe_fused` variable and we will need to ensure there is not a mistake in the pre-preprocessing pipeline.
        * We will first investigate the raw netCDF file then work our way down the pre-preprocessing pipeline should an issue be identified. 
* Temporal Feature Engineering
    * Comment: The `exact_time` column is a timestamp and we must ensure that future models do not learn a false linear ordering. To fix this we will extract the month of the year from `exact-time` then project it onto a unit circle.
    * $\text{month}_{sin} = sin(\frac{2\pi \cdot month_{i}}{12})$
    * $\text{month}_{cos} = cos(\frac{2\pi \cdot month_{i}}{12})$
    * Method: The transformation requires extracting the month from the timestamp using `F.month(F.col("exact_time"))`, then applying `F.sin()` and `F.cos()` with the
appropriate scaling factor $\frac{2\pi}{12}$. This will produce two new continuous columns (`month_sin`, `month_cos`) that replace `exact_time` in the feature vector.
* Ocean Null Values
    * Comment: To handle the valid null values in the ocean features, we will take a two step approach.
        * 1. Create a binary classifier where we have valid ocean data.
            * Method: We will use something like `F.when(F.col("thetao_mo").isNotNull(), 1).otherwise(0)` to create the binary classifier. 
        * 2. Using the biniary classifier previously created, we will fill all non-valid ocean data features with `0.0` or other appropriate fill value.
            * Method: We will use df.fillna({ocean_feature_1: 0, ocean_feature_2: 0,..., ocean_feature_N: 0}) to fill non-valid data.
* Scale *Long Tail* Features
    * Comment: As seen in the EDA step, some of the histograms are skewed. To adjust these features we will apply a log transformation. 
    * Method: We will use `F.log1p(F.col(skewed_feature))` to scale highly skewed features.




### Jupyter Notebook Links

[Notebook for SDSC](EDA_SDSC.ipynb)

[Additional EDA on SDSC](/eda_sdsc/bonus_eda_plots.ipynb)

[Notebook for sample data](EDA_local_sample_data.ipynb)

