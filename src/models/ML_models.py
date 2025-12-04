"""
Machine Learning Models for predicting top 25% finishers in Major Championships.
This module contains:
1. Random Forest: ensemble tree-based model
2. XGBoost: gradient boosting model
3. SHAP Analysis: model interpretability

Models use the temporal validation approach: using training data: 2020-2024 and testing on 2025.
"""

import sys
import logging
from pathlib import Path

# Add parent directory (src) to path so we can import feature_engineering
SRC_DIR = Path(__file__).parent.parent
if str(SRC_DIR) not in sys.path: sys.path.insert(0, str(SRC_DIR))

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import shap

# Import feature_engineering module
import feature_engineering

# Get functions and variables from the module
FEATURES = feature_engineering.FEATURES
prepare_features = feature_engineering.prepare_features
create_top25_target = feature_engineering.create_top25_target

logger = logging.getLogger(__name__)

# =============================================================================
# Data Preparation
# =============================================================================

def prepare_ml_data(df):
    """Prepare data with temporal split: train on 2020-2024, test on 2025."""
    logger.info("Preparing data with temporal split")
    
    # Create the desired top_25 target using the feature_engineering function
    df = create_top25_target(df)
    
    # Make the temporal split, so 2020-2024 data is used to train and 2025 data to test
    train_df = df[df['year'] < 2025].copy()
    test_df = df[df['year'] == 2025].copy()
    
    # Prepare the features by standardarizing them (already done in feature_engineering)
    X_train, y_train, scaler = prepare_features(train_df, target='top_25')
    
    logger.info("X_train shape: %s, columns: %d", X_train.shape, len(X_train.columns))
    logger.info("X_train columns: %s", X_train.columns.tolist())
    
    # Apply the same scaler to the test data (in order that it fits on trained data only and therefore avoid data leakage)
    # Only select the featires columns 
    test_features = test_df[FEATURES].copy()
    X_test_scaled = scaler.transform(test_features)
    
    # Create the corresponding data frame with column names
    X_test = pd.DataFrame(X_test_scaled, columns=FEATURES, index=test_features.index)
    
    logger.info("X_test shape: %s, columns: %d", X_test.shape, len(X_test.columns))
    logger.info("X_test columns: %s", X_test.columns.tolist())
    
    y_test = test_df.loc[X_test.index, 'top_25']
    
    logger.info("Train: %d records (2020-2024), Test: %d records (2025)", len(X_train), len(X_test))
    logger.info("Class balance - Train: %.1f%% top 25%%, Test: %.1f%% top 25%%", y_train.mean() * 100, y_test.mean() * 100)
    
    return X_train, X_test, y_train, y_test

# =============================================================================
# Model Fitting Functions
# =============================================================================

def fit_random_forest(X_train, X_test, y_train, y_test, n_estimators=100, max_depth=10, random_state=42):
    """
    Fit Random Forest classifier. 
    Goal: to capture non-linear relationships and provide robust feature importance.
    """

    logger.info("Fitting Random Forest")
    
    # Debug fix: Check what we're training on, to make sure we're inputing the desired variables
    logger.info("RF Training on shape: %s, columns: %d", X_train.shape, len(X_train.columns))
    logger.info("RF Training columns: %s", X_train.columns.tolist())
    
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, n_jobs=-1)  # Using all CPU cores so it performs better
    
    model.fit(X_train, y_train)
    
    # Debug fix: check the model's number of features
    logger.info("RF Model trained on %d features", model.n_features_in_)
    logger.info("RF Model feature_importances_ shape: %s", model.feature_importances_.shape)

    # Make the predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Asses the feature importance
    importance_df = pd.DataFrame({'Feature': FEATURES, 'Importance': model.feature_importances_}).sort_values('Importance', ascending=False)
    
    logger.info("Random Forest fitted successfully")
    
    return {'model': model, 'y_test': y_test, 'y_pred': y_pred, 'y_pred_proba': y_pred_proba, 'importance': importance_df}

def fit_xgboost(X_train, X_test, y_train, y_test, n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42):
    """Fit the XGBoost classifier (which often outperforms RF bc it corrects its errors and doesnt overfit very easily."""
    
    logger.info("Fitting XGBoost")
    
    model = xgb.XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=random_state, eval_metric='logloss')
    model.fit(X_train, y_train)
    
    # Add predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Include feature importance (as part of the model's output)
    importance_df = pd.DataFrame({'Feature': FEATURES, 'Importance': model.feature_importances_}).sort_values('Importance', ascending=False)
    
    logger.info("XGBoost fitted successfully")
    
    return {'model': model,'y_test': y_test, 'y_pred': y_pred, 'y_pred_proba': y_pred_proba, 'importance': importance_df}

# =============================================================================
# SHAP Analysis (Stretch Goal)
# =============================================================================

def compute_shap_values(model, X_test):
    """Compute the SHAP values in order to interpret the model correctly and to show how each feature contributes to predictions.
    """
    logger.info("Computing SHAP values")
    
    # Debug: check X_test shape
    logger.info("SHAP input X_test shape: %s", X_test.shape)
    logger.info("SHAP input columns: %s", X_test.columns.tolist() if hasattr(X_test, 'columns') else 'N/A')

    # TreeExplainer for tree-based models
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Since SHAP can return different shapes depending on the model, check whether its:
    # - a list: therefore we have SHAP values for both classes: positive and negative (however, we only need the positive class)
    # - a 3D array: with samples, features and classes (again we only need the positive class)
    if isinstance(shap_values, list):
        # List of arrays with both negative and positive classes
        shap_values = shap_values[1]
        logger.info("SHAP format: list, selected positive class")
    elif len(shap_values.shape) == 3:
        # 3D array (n_samples, n_features, n_classes)
        # Take positive class (index 1 in last dimension)
        shap_values = shap_values[:, :, 1]
        logger.info("SHAP format: 3D array, selected positive class from last dimension")
    
    logger.info("SHAP values shape after class selection: %s", shap_values.shape)

    # shap_values shape should now be: (n_samples, n_features)
    # Compute mean absolute SHAP value for each feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    logger.info("mean_abs_shap shape: %s, length: %d", mean_abs_shap.shape, len(mean_abs_shap))
    
    # Make sure that the shape matches
    if len(mean_abs_shap) != len(FEATURES):
        logger.error("SHAP shape mismatch: got %d values for %d features", len(mean_abs_shap), len(FEATURES))
        raise ValueError(f"SHAP shape mismatch: {len(mean_abs_shap)} vs {len(FEATURES)}")
    
    # Create the required data frame
    shap_importance = pd.DataFrame({'Feature': list(FEATURES), 'SHAP_Importance': list(mean_abs_shap)}).sort_values('SHAP_Importance', ascending=False)
    
    logger.info("SHAP values computed successfully")
    
    return {'shap_values': shap_values, 'shap_importance': shap_importance, 'explainer': explainer}

# =============================================================================
# Main Analysis Function
# =============================================================================

def run_ml_analysis(df, results_dir=None):
    """
    Run machine learning models and save the feature importance to csv.
    Metrics and comparisons are computed separately in evaluation.py.
    """
    logger.info("Starting ML analysis")
    
    # Setup results directory (golf_project/results/3_ML_models)
    if results_dir is None:
        results_dir = Path(__file__).parent.parent.parent / "results" / "3_ML_models"
    else:
        results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # Prepare data with temporal split
    X_train, X_test, y_train, y_test = prepare_ml_data(df)
    
    # =========================================================================
    # Random Forest
    # =========================================================================
    logger.info("Running Random Forest model")
    results['random_forest'] = fit_random_forest(X_train, X_test, y_train, y_test)
    
    # Save random forest feature importance to its corresponding results file
    results['random_forest']['importance'].to_csv(results_dir / "1_rf_feature_importance.csv", index=False)
    logger.info("Saved RF feature importance")
    
    # =========================================================================
    # XGBoost
    # =========================================================================
    logger.info("Running XGBoost model")
    results['xgboost'] = fit_xgboost(X_train, X_test, y_train, y_test)
    
    # Save XGBoost feature importance to its corresponding results file
    results['xgboost']['importance'].to_csv(results_dir / "2_xgb_feature_importance.csv", index=False)
    logger.info("Saved XGBoost feature importance")
    
    # =========================================================================
    # SHAP Analysis (Stretch Goal)
    # =========================================================================
    logger.info("Running SHAP analysis on Random Forest")
    results['shap'] = compute_shap_values(results['random_forest']['model'], X_test)
    
    # Save SHAP importance to its corresponding results file
    results['shap']['shap_importance'].to_csv(results_dir / "3_shap_importance.csv", index=False)
    logger.info("Saved SHAP importance")
    
    logger.info("ML analysis complete. Results saved to %s", results_dir)
    
    return results

# =============================================================================
# Script Entry Point
# =============================================================================

if __name__ == "__main__":
    import sys
    
    # Add src directory to path for imports
    SRC_DIR = Path(__file__).parent
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    
    logging.basicConfig(level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s')
    
    from data_loader import load_combined_data
    from evaluation import evaluate_ml_models
    
    # Load data
    logger.info("Loading combined dataset")
    data = load_combined_data()
    logger.info("Loaded %d player-tournament records", len(data))
    
    # Run analysis (only fits models and saves feature importance)
    results = run_ml_analysis(data)
    
    logger.info("Machine learning models have successfully completed!")

    # Evaluate all the models and save its corresponding metrics
    results_dir = Path(__file__).parent.parent.parent / "results" / "3_ML_models"
    evaluate_ml_models(results, results_dir)
    
    logger.info("The machine learning models' evaluation has been completed!")