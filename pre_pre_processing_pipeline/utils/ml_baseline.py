import duckdb
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
SUBSET_PATH = "data/flattened/ml_subset_amundsen_sea.parquet"


# The month index representing approx Jan 2024 (2024 * 12 + 1 = 24289)
# Everything before this is TRAIN, everything after is TEST.
TEMPORAL_SPLIT_MONTH = 24289 


def train_baseline_model():
    print("=" * 72)
    print(" TRAINING GLACIOLOGICAL XGBOOST BASELINE (AMUNDSEN SEA)")
    print("=" * 72)


    # 1. Lazy Data Extraction & Leakage Prevention via DuckDB
    print("\n[1/4] Querying Data and Enforcing Temporal Split...")
    con = duckdb.connect()
    
    # We explicitly select ONLY the physical drivers, dropping ICESat-2 kinematics.
    # We drop NULLs in the target or essential ocean data.
    base_query = f"""
        SELECT 
            month_idx,
            surface, bed, thickness, bed_slope, dist_to_grounding_line, clamped_depth, dist_to_ocean, ice_draft,
            thetao_mo, t_star_mo, so_mo, t_f_mo,
            t_star_quarterly_avg, t_star_quarterly_std, thetao_quarterly_avg, thetao_quarterly_std,
            lwe_mo, lwe_quarterly_avg, lwe_quarterly_std,
            lwe_fused
        FROM read_parquet('{SUBSET_PATH}')
        WHERE lwe_fused IS NOT NULL 
          AND t_star_mo IS NOT NULL
    """


    # Extract Train/Test as Pandas DataFrames directly
    print("  -> Fetching Training Set (< 2024)...")
    df_train = con.execute(f"{base_query} AND month_idx < {TEMPORAL_SPLIT_MONTH}").df()
    
    print("  -> Fetching Testing Set (>= 2024)...")
    df_test = con.execute(f"{base_query} AND month_idx >= {TEMPORAL_SPLIT_MONTH}").df()
    con.close()


    print(f"  -> Train shape: {df_train.shape[0]:,} rows")
    print(f"  -> Test shape:  {df_test.shape[0]:,} rows")


    # 2. Feature Matrix Preparation
    features = [c for c in df_train.columns if c not in ['lwe_fused', 'month_idx']]
    
    X_train, y_train = df_train[features], df_train['lwe_fused']
    X_test, y_test = df_test[features], df_test['lwe_fused']


    # Free up Pandas memory
    del df_train
    del df_test


    # 3. XGBoost Training (Using 'hist' for massive datasets)
    print("\n[2/4] Initializing and Training XGBoost Regressor...")
    
    model = xgb.XGBRegressor(
        n_estimators=200,          # Number of boosting rounds
        learning_rate=0.1,         # Step size shrinkage
        max_depth=8,               # Maximum tree depth
        tree_method='hist',        # CRITICAL: Uses histogram binning for fast big-data training
        subsample=0.8,             # Prevent overfitting
        colsample_bytree=0.8,
        n_jobs=-1,                 # Use all CPU cores
        random_state=42
    )


    # Train the model
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=10
    )


    # 4. Evaluation
    print("\n[3/4] Evaluating Model Physics on Unseen Future Data (2024-2025)...")
    y_pred = model.predict(X_test)


    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)


    print(f"  -> RMSE: {rmse:.4f} mm LWE")
    print(f"  -> MAE:  {mae:.4f} mm LWE")
    print(f"  -> R^2:  {r2:.4f}")


    if r2 < 0.1:
        print("  [!] WARNING: Model failed to capture the variance. Physical drivers might be decoupled from the target.")


    # 5. Feature Importance Extraction
    print("\n[4/4] Extracting Scientific Feature Importance...")
    importance = model.feature_importances_
    feat_imp_df = pd.DataFrame({'Feature': features, 'Importance': importance})
    feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False)
    
    print("\nTop 5 Drivers of Mass Change:")
    for idx, row in feat_imp_df.head(5).iterrows():
        print(f"  1. {row['Feature']:25s} ({row['Importance']*100:.1f}%)")


    print("\n=======================================================")
    print(" BASELINE COMPLETE.")
    print("=======================================================")


if __name__ == "__main__":
    train_baseline_model()

