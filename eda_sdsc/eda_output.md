Output directories:
    Plots:   /expanse/lustre/projects/uci157/rrogers/data/eda_plots
    Samples: /expanse/lustre/projects/uci157/rrogers/data/sample_data
========================================================================
  HPC SparkSession Configuration
========================================================================
  TOTAL_CORES ............. 32
  TOTAL_MEMORY_GB ......... 128
  DRIVER_MEMORY ........... 10g
  EXECUTOR_CORES .......... 5
  EXECUTOR_INSTANCES ...... 6
  EXECUTOR_MEMORY ......... 19g
  SHUFFLE_PARTITIONS ...... 64
========================================================================

  Discovered 5 dataset(s):
    1. bedmap3_static.parquet
    2. grace.parquet
    3. icesat2_dynamic.parquet
    4. ocean_dynamic.parquet
    5. antarctica_sparse_features.parquet

════════════════════════════════════════════════════════════════════════
  PHASE 1 : Per-Dataset EDA Pipeline
════════════════════════════════════════════════════════════════════════

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  DATASET:  bedmap3_static.parquet
  PATH:     /expanse/lustre/projects/uci157/rrogers/data/indiv_data/bedmap3_static.parquet
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  [Phase 1] Ingesting + reading schema ...
  Schema (13 columns):
  Column Name                     Data Type
  ──────────────────────────────  ────────────────────
  x                               double
  y                               double
  spatial_ref                     bigint
  surface                         float
  bed                             float
  thickness                       float
  mask                            tinyint
  mascon_id                       double
  bed_slope                       float
  dist_to_grounding_line          float
  clamped_depth                   float
  dist_to_ocean                   float
  ice_draft                       float
  [1.5s]

  [Phase 2] Counting rows ...
  Total rows:        54,236,727
  Total columns:             13
  [1.2s]

  [Phase 3] Computing summary statistics ...
  Numeric columns (13): ['x', 'y', 'spatial_ref', 'surface', 'bed', 'thickness', 'mask', 'mascon_id', 'bed_slope', 'dist_to_grounding_line', 'clamped_depth', 'dist_to_ocean', 'ice_draft']

  Summary Statistics for: bedmap3_static.parquet
-RECORD 0-------------------------------------
 summary                | count
 x                      | 54236727
 y                      | 54236727
 spatial_ref            | 54236727
 surface                | 54188350
 bed                    | 54236727
 thickness              | 54236727
 mask                   | 54236727
 mascon_id              | 54236727
 bed_slope              | 54236727
 dist_to_grounding_line | 54236727
 clamped_depth          | 6128170
 dist_to_ocean          | 6153122
 ice_draft              | 6128170
-RECORD 1-------------------------------------
 summary                | min
 x                      | -2658250.0
 y                      | -2493750.0
 spatial_ref            | 0
 surface                | 0.9999926
 bed                    | -3003.0
 thickness              | 0.0
 mask                   | 1
 mascon_id              | 201.0
 bed_slope              | 0.0
 dist_to_grounding_line | 0.0
 clamped_depth          | 0.0
 dist_to_ocean          | 5.9053884
 ice_draft              | -2483.0
-RECORD 2-------------------------------------
 summary                | max
 x                      | 2750250.0
 y                      | 2321250.0
 spatial_ref            | 0
 surface                | 4734.0
 bed                    | 4571.0
 thickness              | 4757.0
 mask                   | 3
 mascon_id              | 170457.0
 bed_slope              | 2.0755377
 dist_to_grounding_line | 307358.1
 clamped_depth          | 902.3393
 dist_to_ocean          | 943906.25
 ice_draft              | 150.0
-RECORD 3-------------------------------------
 summary                | mean
 x                      | 457964.8308009073
 y                      | 82705.50544559961
 spatial_ref            | 0.0
 surface                | 1968.1946818273107
 bed                    | -19.090939834897412
 thickness              | 1954.0557107363716
 mask                   | 1.2268987212299887
 mascon_id              | 78412.23599349939
 bed_slope              | 0.04428633084574028
 dist_to_grounding_line | 6026.085969776766
 clamped_depth          | 218.05310435404678
 dist_to_ocean          | 249389.3763039005
 ice_draft              | -413.99859109652107
-RECORD 4-------------------------------------
 summary                | stddev
 x                      | 1154291.1263901023
 y                      | 1010598.3603795386
 spatial_ref            | 0.0
 surface                | 1173.2229926176399
 bed                    | 701.4950507231852
 thickness              | 1055.855700409052
 mask                   | 0.6342826027741315
 mascon_id              | 31465.299134711986
 bed_slope              | 0.06088434571312098
 dist_to_grounding_line | 27611.899182768768
 clamped_depth          | 167.7036173823966
 dist_to_ocean          | 216471.43298194322
 ice_draft              | 238.63534189047147

  [7.5s]

  [Phase 4] Collecting metadata ...
      [Step 1/2] Global row count ...
      → 54,236,727 rows
      [Step 2/2] Column completeness ...
  [1.2s]

  [Phase 5] Building histograms ...
      Computing histogram: surface ...
      Computing histogram: bed ...
      Computing histogram: thickness ...
      Computing histogram: bed_slope ...
      Computing histogram: dist_to_grounding_line ...
      Computing histogram: clamped_depth ...
      Computing histogram: ice_draft ...
    → Saved: data/eda_plots/fig_03_histograms_bedmap3_static.png
  [71.6s]

  [Phase 6] Building correlation heatmap ...
      Computing 7x7 correlation matrix ...
    → Saved: data/eda_plots/fig_04_correlation_bedmap3_static.png
  [4.1s]

  [Phase 7] Computing range stats ...
  [1.0s]

  [Phase 8] Generating representative sample ...
      Sampling fraction: 0.005531 (target ~250,000 rows from 54,236,727)
      → Sample written: data/sample_data/bedmap3_static_sample.parquet
  [3.5s]

  ✓ bedmap3_static.parquet complete : 91.63s

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  DATASET:  grace.parquet
  PATH:     /expanse/lustre/projects/uci157/rrogers/data/indiv_data/grace.parquet
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  [Phase 1] Ingesting + reading schema ...
  Schema (9 columns):
  Column Name                     Data Type
  ──────────────────────────────  ────────────────────
  y                               double
  x                               double
  time                            timestamp_ntz
  WGS84                           bigint
  land_mask                       float
  lwe_length                      float
  mascon_id                       double
  ocean_mask                      float
  spatial_ref                     bigint
  [0.1s]

  [Phase 2] Counting rows ...
  Total rows:        19,839,750
  Total columns:              9
  [0.1s]

  [Phase 3] Computing summary statistics ...
  Numeric columns (8): ['y', 'x', 'WGS84', 'land_mask', 'lwe_length', 'mascon_id', 'ocean_mask', 'spatial_ref']

  Summary Statistics for: grace.parquet
-RECORD 0-------------------------
 summary     | count
 y           | 19839750
 x           | 19839750
 WGS84       | 19839750
 land_mask   | 4065000
 lwe_length  | 19839750
 mascon_id   | 19839750
 ocean_mask  | 15774750
 spatial_ref | 19839750
-RECORD 1-------------------------
 summary     | min
 y           | -4508087.29346864
 x           | -4510662.70653136
 WGS84       | 0
 land_mask   | 1.0
 lwe_length  | -1473.2893
 mascon_id   | 201.0
 ocean_mask  | 1.0
 spatial_ref | 0
-RECORD 2-------------------------
 summary     | max
 y           | 4510662.70653136
 x           | 4508087.29346864
 WGS84       | 0
 land_mask   | 1.0
 lwe_length  | 318.55743
 mascon_id   | 230399.0
 ocean_mask  | 1.0
 spatial_ref | 0
-RECORD 3-------------------------
 summary     | mean
 y           | 6623.249768401322
 x           | 8191.156925136567
 WGS84       | 0.0
 land_mask   | 1.0
 lwe_length  | 0.348350804220669
 mascon_id   | 158163.0278355322
 ocean_mask  | 1.0
 spatial_ref | 0.0
-RECORD 4-------------------------
 summary     | stddev
 y           | 2296064.2412417764
 x           | 2285242.1966471337
 WGS84       | 0.0
 land_mask   | 0.0
 lwe_length  | 28.806726180723775
 mascon_id   | 53916.543098978196
 ocean_mask  | 0.0
 spatial_ref | 0.0

  [0.8s]

  [Phase 4] Collecting metadata ...
      [Step 1/2] Global row count ...
      → 19,839,750 rows
      [Step 2/2] Column completeness ...
  [2.0s]

  [Phase 5] Building histograms ...
      Computing histogram: lwe_length ...
    → Saved: data/eda_plots/fig_03_histograms_grace.png
  [6.3s]

  [Phase 6] Need >= 2 PHYS_COLUMNS : skipping correlation.

  [Phase 7] Computing range stats ...
  [0.4s]

  [Phase 8] Generating representative sample ...
      Sampling fraction: 0.015121 (target ~250,000 rows from 19,839,750)
      → Sample written: data/sample_data/grace_sample.parquet
  [1.4s]

  ✓ grace.parquet complete : 11.06s

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  DATASET:  icesat2_dynamic.parquet
  PATH:     /expanse/lustre/projects/uci157/rrogers/data/indiv_data/icesat2_dynamic.parquet
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  [Phase 1] Ingesting + reading schema ...
  Schema (8 columns):
  Column Name                     Data Type
  ──────────────────────────────  ────────────────────
  y                               double
  x                               double
  delta_h                         float
  ice_area                        float
  h_surface_dynamic               float
  surface_slope                   float
  spatial_ref                     bigint
  time                            timestamp_ntz
  [0.2s]

  [Phase 2] Counting rows ...
  Total rows:     1,386,866,499
  Total columns:              8
  [0.2s]

  [Phase 3] Computing summary statistics ...
  Numeric columns (7): ['y', 'x', 'delta_h', 'ice_area', 'h_surface_dynamic', 'surface_slope', 'spatial_ref']

  Summary Statistics for: icesat2_dynamic.parquet
-RECORD 0---------------------------------
 summary           | count
 y                 | 1386866499
 x                 | 1386866499
 delta_h           | 1386866499
 ice_area          | 1386866499
 h_surface_dynamic | 1386586808
 surface_slope     | 1382181928
 spatial_ref       | 1386866499
-RECORD 1---------------------------------
 summary           | min
 y                 | -2142750.0
 x                 | -2647750.0
 delta_h           | -63.175518
 ice_area          | 28734.4
 h_surface_dynamic | -40.98206
 surface_slope     | 0.0
 spatial_ref       | 0
-RECORD 2---------------------------------
 summary           | max
 y                 | 2251250.0
 x                 | 2749750.0
 delta_h           | 72.98236
 ice_area          | 1056117.8
 h_surface_dynamic | 4083.0994
 surface_slope     | 1.5100615
 spatial_ref       | 0
-RECORD 3---------------------------------
 summary           | mean
 y                 | 100702.80517299451
 x                 | 484456.40172915446
 delta_h           | 0.04365065201126981
 ice_area          | 1021863.5000643947
 h_surface_dynamic | 1980.9560510717363
 surface_slope     | 0.00742234555780814
 spatial_ref       | 0.0
-RECORD 4---------------------------------
 summary           | stddev
 y                 | 1010022.8715430193
 x                 | 1149477.3871542753
 delta_h           | 0.722332850079902
 ice_area          | 28568.54396990228
 h_surface_dynamic | 1175.8764835352185
 surface_slope     | 0.016757591142793443
 spatial_ref       | 0.0

  [40.0s]

  [Phase 4] Collecting metadata ...
      [Step 1/2] Global row count ...
      → 1,386,866,499 rows
      [Step 2/2] Column completeness ...
  [7.1s]

  [Phase 5] Building histograms ...
      Computing histogram: delta_h ...
      Computing histogram: ice_area ...
      Computing histogram: h_surface_dynamic ...
      Computing histogram: surface_slope ...
    → Saved: data/eda_plots/fig_03_histograms_icesat2_dynamic.png
  [1239.4s]

  [Phase 6] Building correlation heatmap ...
      Computing 4x4 correlation matrix ...
    → Saved: data/eda_plots/fig_04_correlation_icesat2_dynamic.png
  [19.5s]

  [Phase 7] Computing range stats ...
  [6.4s]

  [Phase 8] Generating representative sample ...
      Sampling fraction: 0.000216 (target ~250,000 rows from 1,386,866,499)
      → Sample written: data/sample_data/icesat2_dynamic_sample.parquet
  [62.2s]

  ✓ icesat2_dynamic.parquet complete : 65.17s

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  DATASET:  ocean_dynamic.parquet
  PATH:     /expanse/lustre/projects/uci157/rrogers/data/indiv_data/ocean_dynamic.parquet
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  [Phase 1] Ingesting + reading schema ...
  Schema (7 columns):
  Column Name                     Data Type           
  ──────────────────────────────  ────────────────────
  x                               double              
  y                               double              
  time                            timestamp_ntz       
  thetao                          float               
  so                              float               
  T_f                             float               
  T_star                          float               
  [0.3s]

  [Phase 2] Counting rows ...
  Total rows:       191,213,120
  Total columns:              7
  [0.2s]

  [Phase 3] Computing summary statistics ...
  Numeric columns (6): ['x', 'y', 'thetao', 'so', 'T_f', 'T_star']

  Summary Statistics for: ocean_dynamic.parquet
-RECORD 0----------------------
 summary | count               
 x       | 191213120           
 y       | 191213120           
 thetao  | 191213120           
 so      | 191213120           
 T_f     | 191213120           
 T_star  | 191213120           
-RECORD 1----------------------
 summary | min                 
 x       | -2451750.0          
 y       | -2139250.0          
 thetao  | -2.9249792          
 so      | 25.325829           
 T_f     | -2.5926855          
 T_star  | -0.9222529          
-RECORD 2----------------------
 summary | max                 
 x       | 2750250.0           
 y       | 2251250.0           
 thetao  | 6.812678            
 so      | 40.399906           
 T_f     | -1.3668962          
 T_star  | 8.582407            
-RECORD 3----------------------
 summary | mean                
 x       | -260112.21552161273 
 y       | -500864.482939246   
 thetao  | -1.1800908253806466 
 so      | 34.4841175377162    
 T_f     | -2.1386755310422734 
 T_star  | 0.9585847056768133  
-RECORD 4----------------------
 summary | stddev              
 x       | 903303.355351509    
 y       | 880340.3199723623   
 thetao  | 0.8418548940107553  
 so      | 0.24166073314866707 
 T_f     | 0.1022522699341961  
 T_star  | 0.8520680842103103  

  [3.2s]

  [Phase 4] Collecting metadata ...
      [Step 1/2] Global row count ...
      → 191,213,120 rows
      [Step 2/2] Column completeness ...
  [2.2s]

  [Phase 5] Building histograms ...
      Computing histogram: thetao ...
      Computing histogram: so ...
      Computing histogram: T_f ...
      Computing histogram: T_star ...
    → Saved: data/eda_plots/fig_03_histograms_ocean_dynamic.png
  [197.5s]

  [Phase 6] Building correlation heatmap ...
      Computing 4x4 correlation matrix ...
    → Saved: data/eda_plots/fig_04_correlation_ocean_dynamic.png
  [3.2s]

  [Phase 7] Computing range stats ...
  [1.2s]

  [Phase 8] Generating representative sample ...
      Sampling fraction: 0.001569 (target ~250,000 rows from 191,213,120)
      → Sample written: data/sample_data/ocean_dynamic_sample.parquet
  [8.0s]

  ✓ ocean_dynamic.parquet complete : 215.84s

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  DATASET:  antarctica_sparse_features.parquet
  PATH:     /expanse/lustre/projects/uci157/rrogers/data/fused_data/antarctica_sparse_features.parquet
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  [Phase 1] Ingesting + reading schema ...
  Schema (28 columns):
  Column Name                     Data Type           
  ──────────────────────────────  ────────────────────
  y                               double              
  x                               double              
  exact_time                      timestamp_ntz       
  mascon_id                       int                 
  surface                         float               
  bed                             float               
  thickness                       float               
  bed_slope                       float               
  dist_to_grounding_line          float               
  clamped_depth                   float               
  dist_to_ocean                   float               
  ice_draft                       float               
  delta_h                         float               
  ice_area                        float               
  surface_slope                   float               
  h_surface_dynamic               float               
  thetao_mo                       double              
  t_star_mo                       double              
  so_mo                           double              
  t_f_mo                          double              
  t_star_quarterly_avg            double              
  t_star_quarterly_std            double              
  thetao_quarterly_avg            double              
  thetao_quarterly_std            double              
  lwe_mo                          float               
  lwe_quarterly_avg               double              
  lwe_quarterly_std               double              
  lwe_fused                       double              
  [1.5s]

  [Phase 2] Counting rows ...
  Total rows:     1,386,866,499
  Total columns:             28
  [0.9s]

  [Phase 3] Computing summary statistics ...
  Numeric columns (27): ['y', 'x', 'mascon_id', 'surface', 'bed', 'thickness', 'bed_slope', 'dist_to_grounding_line', 'clamped_depth', 'dist_to_ocean', 'ice_draft', 'delta_h', 'ice_area', 'surface_slope', 'h_surface_dynamic', 'thetao_mo', 't_star_mo', 'so_mo', 't_f_mo', 't_star_quarterly_avg', 't_star_quarterly_std', 'thetao_quarterly_avg', 'thetao_quarterly_std', 'lwe_mo', 'lwe_quarterly_avg', 'lwe_quarterly_std', 'lwe_fused']

  Summary Statistics for: antarctica_sparse_features.parquet
-RECORD 0--------------------------------------
 summary                | count                
 y                      | 1386866499           
 x                      | 1386866499           
 mascon_id              | 1386866499           
 surface                | 1386586808           
 bed                    | 1386866499           
 thickness              | 1386866499           
 bed_slope              | 1386866499           
 dist_to_grounding_line | 1386866499           
 clamped_depth          | 157449872            
 dist_to_ocean          | 157677917            
 ice_draft              | 157449872            
 delta_h                | 1386866499           
 ice_area               | 1386866499           
 surface_slope          | 1382181928           
 h_surface_dynamic      | 1386586808           
 thetao_mo              | 60704788             
 t_star_mo              | 60704788             
 so_mo                  | 60704788             
 t_f_mo                 | 60704788             
 t_star_quarterly_avg   | 60704788             
 t_star_quarterly_std   | 60704788             
 thetao_quarterly_avg   | 60704788             
 thetao_quarterly_std   | 60704788             
 lwe_mo                 | 1267665751           
 lwe_quarterly_avg      | 1267665751           
 lwe_quarterly_std      | 1267665751           
 lwe_fused              | 1267665751           
-RECORD 1--------------------------------------
 summary                | min                  
 y                      | -2142750.0           
 x                      | -2647750.0           
 mascon_id              | 10096                
 surface                | 0.9999926            
 bed                    | -2996.0              
 thickness              | 0.0                  
 bed_slope              | 0.0                  
 dist_to_grounding_line | 0.0                  
 clamped_depth          | 0.0                  
 dist_to_ocean          | 5.9053884            
 ice_draft              | -2483.0              
 delta_h                | -63.175518           
 ice_area               | 28734.4              
 surface_slope          | 0.0                  
 h_surface_dynamic      | -40.98206            
 thetao_mo              | -2.90527081489563    
 t_star_mo              | -0.8453882932662964  
 so_mo                  | 27.196569442749023   
 t_f_mo                 | -2.5923352241516113  
 t_star_quarterly_avg   | -0.790689746538798   
 t_star_quarterly_std   | 0.0                  
 thetao_quarterly_avg   | -2.8347113132476807  
 thetao_quarterly_std   | 0.0                  
 lwe_mo                 | -1391.5425           
 lwe_quarterly_avg      | -1382.8002522786458  
 lwe_quarterly_std      | 0.003207170794120565 
 lwe_fused              | -15009.435468693218  
-RECORD 2--------------------------------------
 summary                | max                  
 y                      | 2251250.0            
 x                      | 2749750.0            
 mascon_id              | 161767               
 surface                | 4083.0               
 bed                    | 3854.0               
 thickness              | 4757.0               
 bed_slope              | 1.7332709            
 dist_to_grounding_line | 307358.1             
 clamped_depth          | 902.3393             
 dist_to_ocean          | 943906.25            
 ice_draft              | 95.0                 
 delta_h                | 72.98236             
 ice_area               | 1056117.8            
 surface_slope          | 1.5100615            
 h_surface_dynamic      | 4083.0994            
 thetao_mo              | 6.81267786026001     
 t_star_mo              | 8.582406997680664    
 so_mo                  | 40.399906158447266   
 t_f_mo                 | -1.4737026691436768  
 t_star_quarterly_avg   | 7.131954669952393    
 t_star_quarterly_std   | 4.441052343275747    
 thetao_quarterly_avg   | 5.263535022735596    
 thetao_quarterly_std   | 4.46253709870591     
 lwe_mo                 | 308.6047             
 lwe_quarterly_avg      | 282.78052775065106   
 lwe_quarterly_std      | 124.76495029644343   
 lwe_fused              | 3254.341256274911    
-RECORD 3--------------------------------------
 summary                | mean                 
 y                      | 100702.80517299451   
 x                      | 484456.40172915446   
 mascon_id              | 79068.71370015622    
 surface                | 1980.9125133245632   
 bed                    | -29.703858194457545  
 thickness              | 1978.8053274477413   
 bed_slope              | 0.04199619501934072  
 dist_to_grounding_line | 6033.176128956389    
 clamped_depth          | 218.89688398352388   
 dist_to_ocean          | 252765.55095886035   
 ice_draft              | -417.46685030013265  
 delta_h                | 0.04365065201126981  
 ice_area               | 1021863.5000643943   
 surface_slope          | 0.007422345557808631 
 h_surface_dynamic      | 1980.9560510717333   
 thetao_mo              | -1.186002383281592   
 t_star_mo              | 0.9560781991522632   
 so_mo                  | 34.495694358871276   
 t_f_mo                 | -2.1420805823900877  
 t_star_quarterly_avg   | 0.9616828792730597   
 t_star_quarterly_std   | 0.09339955520210133  
 thetao_quarterly_avg   | -1.1802814501994396  
 thetao_quarterly_std   | 0.09327051291777833  
 lwe_mo                 | -7.492825030805613   
 lwe_quarterly_avg      | -7.590358743259416   
 lwe_quarterly_std      | 4.263521248850509    
 lwe_fused              | -7.492825030805532   
-RECORD 4--------------------------------------
 summary                | stddev               
 y                      | 1010022.8715430268   
 x                      | 1149477.3871542797   
 mascon_id              | 30533.05365748639    
 surface                | 1175.8559615397523   
 bed                    | 690.8748539849273    
 thickness              | 1038.7665388841108   
 bed_slope              | 0.054999620648932976 
 dist_to_grounding_line | 27514.29665569326    
 clamped_depth          | 168.2542422407395    
 dist_to_ocean          | 216233.02748440698   
 ice_draft              | 238.56565281745264   
 delta_h                | 0.7223328500798966   
 ice_area               | 28568.5439699012     
 surface_slope          | 0.01675759114279349  
 h_surface_dynamic      | 1175.8764835352088   
 thetao_mo              | 0.8493957727098758   
 t_star_mo              | 0.8592620640601307   
 so_mo                  | 0.23000426980847521  
 t_f_mo                 | 0.10041485335728434  
 t_star_quarterly_avg   | 0.8450923088855552   
 t_star_quarterly_std   | 0.12575299473930432  
 thetao_quarterly_avg   | 0.8355628637887981   
 thetao_quarterly_std   | 0.1258449708936023   
 lwe_mo                 | 99.04139054477693    
 lwe_quarterly_avg      | 98.36222609132852    
 lwe_quarterly_std      | 4.996658065119011    
 lwe_fused              | 111.478941830573     

  [189.2s]

  [Phase 4] Collecting metadata ...
      [Step 1/2] Global row count ...
      → 1,386,866,499 rows
      [Step 2/2] Column completeness ...
  [18.6s]

  [Phase 5] Building histograms ...
      Computing histogram: surface ...
      Computing histogram: bed ...
      Computing histogram: thickness ...
      Computing histogram: delta_h ...
    → Saved: data/eda_plots/fig_03_histograms_antarctica_sparse_features.png
  [1186.7s]

  [Phase 6] Building correlation heatmap ...
      Computing 4x4 correlation matrix ...
    → Saved: data/eda_plots/fig_04_correlation_antarctica_sparse_features.png
  [13.6s]

  [Phase 7] Computing range stats ...
  [5.0s]

  [Phase 8] Generating representative sample ...
      Sampling fraction: 0.000216 (target ~250,000 rows from 1,386,866,499)
  ✓ antarctica_sparse_features.parquet complete : 1565.08s

════════════════════════════════════════════════════════════════════════
  PHASE 2 : Cross-Dataset Figures
════════════════════════════════════════════════════════════════════════

  Building dataset overview ...
    → Saved: data/eda_plots/fig_01_dataset_overview.png
  Building completeness heatmap ...
    → Saved: data/eda_plots/fig_02_data_completeness.png
  Building null structure chart ...
    → Saved: data/eda_plots/fig_06_null_structure.png
  Building physical ranges chart ...
    → Saved: data/eda_plots/fig_05_physical_ranges.png

========================================================================
  PERFORMANCE REPORT
========================================================================

  Dataset                                     Time (s)
  ────────────────────────────────────────  ──────────
  icesat2_dynamic.parquet                        65.17
  ocean_dynamic.parquet                         215.84
  antarctica_sparse_features.parquet           1565.08
  ────────────────────────────────────────  ──────────
  TOTAL                                        1851.71

  Executor configuration:
    Executor instances  = 6
    Executor cores      = 5
    Executor memory     = 19g
    Driver memory       = 10g
    Tₙ (this run)       = 1851.71s

  ┌────────────────────────────────────────────────────────────────────┐
  │  SPEEDUP & EFFICIENCY (requires a baseline run)                   │
  │                                                                   │
  │  1. Run with TOTAL_CORES=6, EXECUTOR_CORES=5 (→ 1 executor)      │
  │     Record the total wall-clock time as T₁.                       │
  │                                                                   │
  │  2. Run with TOTAL_CORES=32  (→ 6  executors)              │
  │     Tₙ = 1851.71 s (this run).                           │
  │                                                                   │
  │  3. Compute:                                                      │
  │       Speedup    = T₁ / Tₙ                                       │
  │       Efficiency = Speedup / n                                    │
  │                  = T₁ / (Tₙ x n)                                  │
  │                                                                   │
  │  Perfect scaling → Speedup = n, Efficiency = 1.0                  │
  └────────────────────────────────────────────────────────────────────┘


  ┌─────────────────────────────────────────────────────┐
  │  ✓ UNIFIED EDA PIPELINE COMPLETE                    │
  │    Total time: 1851.7s  (30.9 min)              │
  │    Plots dir:  /expanse/lustre/projects/uci157/rrogers/data/eda_plots│
  │    Sample dir: /expanse/lustre/projects/uci157/rrogers/data/sample_data│
  └─────────────────────────────────────────────────────┘
