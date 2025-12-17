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
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
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
# ML-Specific Feature Engineering
# =============================================================================


def add_major_dummies(df):
    """Add Major dummies as a categorial feature for ML models in order to be able to analyze the feature importance per major."""
    major_dummies = pd.get_dummies(df["major"], prefix="major", drop_first=False)
    return pd.concat([df, major_dummies], axis=1)


MAJOR_COLS = [
    "major_The Masters",
    "major_PGA Championship",
    "major_The Open Championship",
    "major_US Open",
]
FEATURES_WITH_MAJOR = FEATURES + MAJOR_COLS

# =============================================================================
# Data Preparation
# =============================================================================


def prepare_ml_data(df):
    """Prepare data with temporal split: train on 2020-2024, test on 2025."""

    # Create the desired top_25 target using the feature_engineering function
    df = create_top25_target(df)

    # Add major dummy variables so models learn major-specific patterns
    df = add_major_dummies(df)

    # Make the temporal split, so 2020-2024 data is used to train and 2025 data to test
    train_df = df[df["year"] < 2025].copy()
    test_df = df[df["year"] == 2025].copy()

    # Prepare the features, standardarization of performance variables already done in feature_engineering.
    X_train_perf, y_train, scaler = prepare_features(
        train_df, features=FEATURES, target="top_25"
    )

    # Add the major dummies (unscaled) to the scaled performance features
    X_train_dummies = train_df.loc[X_train_perf.index, MAJOR_COLS]
    X_train = pd.concat([X_train_perf, X_train_dummies], axis=1)

    # Apply the same scaler to the test data (in order that it fits on trained data only and therefore avoid data leakage)
    # Only select the features columns
    test_features = test_df[FEATURES].copy()
    X_test_scaled = scaler.transform(test_features)

    # Add the major dummies (unscaled) to test data
    X_test_perf = pd.DataFrame(
        X_test_scaled, columns=FEATURES, index=test_features.index
    )
    X_test_dummies = test_df.loc[X_test_perf.index, MAJOR_COLS]
    X_test = pd.concat([X_test_perf, X_test_dummies], axis=1)

    y_test = test_df.loc[X_test.index, "top_25"]

    return X_train, X_test, y_train, y_test


# =============================================================================
# Hyperparameter Tuning
# =============================================================================


def tune_random_forest(X_train, y_train):
    """Use GridSearchCV to find the best Random Forest hyperparameters, in order to reduce overfitting by finding optimal complexity through cross-validation."""

    param_grid = {
        "max_depth": [4, 5, 6],
        "min_samples_split": [10, 15, 20],
        "min_samples_leaf": [5, 7, 10],
        "max_features": ["sqrt", "log2"],
    }

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    grid_search = GridSearchCV(
        rf_model,
        param_grid,
        cv=5,  # 5-fold cross-validation on training data only
        scoring="roc_auc",
        n_jobs=-1,
        verbose=0,
    )

    #
    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_


def tune_xgboost(X_train, y_train):
    """Use GridSearchCV to find best XGBoost hyperparameters, in order to reduce overfitting by finding optimal complexity through cross-validation."""

    param_grid = {
        "max_depth": [3, 4, 5],
        "min_child_weight": [3, 5, 7],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.7, 0.8, 0.9],
    }

    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        eval_metric="logloss",
    )

    grid_search = GridSearchCV(
        xgb_model,
        param_grid,
        cv=5,  # 5-fold cross-validation on training data only
        scoring="roc_auc",
        n_jobs=-1,
        verbose=0,
    )

    #
    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_


# =============================================================================
# Model Fitting Functions
# =============================================================================


def fit_random_forest(X_train, X_test, y_train, y_test):
    """
    Fit Random Forest classifier.
    Goal: to capture non-linear relationships and provide robust feature importance.
    """

    # Use GridSearchCV to find best hyperparameters
    model = tune_random_forest(X_train, y_train)

    # Print clean hyperparameter summary
    logger.info(
        f"Random Forest Hyperparameters: max_depth={model.max_depth}, min_samples_split={model.min_samples_split}, min_samples_leaf={model.min_samples_leaf}, max_features={model.max_features}"
    )
    logger.info("Random Forest Cross-validation: 5-fold CV on 2020-2024 training data")
    print()  # Blank line

    # Make the predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Asess the feature importance
    importance_df = pd.DataFrame(
        {"Feature": FEATURES_WITH_MAJOR, "Importance": model.feature_importances_}
    ).sort_values("Importance", ascending=False)

    return {
        "model": model,
        "X_train": X_train,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_pred_proba": y_pred_proba,
        "importance": importance_df,
    }


def fit_xgboost(X_train, X_test, y_train, y_test):
    """Fit the XGBoost classifier (which often outperforms RF bc it corrects its errors and doesnt overfit very easily."""

    # Use GridSearchCV to find best hyperparameters
    model = tune_xgboost(X_train, y_train)

    # Print clean hyperparameter summary
    logger.info(
        f"XGBoost Hyperparameters: max_depth={model.max_depth}, min_child_weight={model.min_child_weight}, subsample={model.subsample}, colsample_bytree={model.colsample_bytree}"
    )
    logger.info("XGBoost Cross-validation: 5-fold CV on 2020-2024 training data")
    print()  # Blank line

    # Add predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Include feature importance (as part of the model's output)
    importance_df = pd.DataFrame(
        {"Feature": FEATURES_WITH_MAJOR, "Importance": model.feature_importances_}
    ).sort_values("Importance", ascending=False)

    return {
        "model": model,
        "X_train": X_train,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_pred_proba": y_pred_proba,
        "importance": importance_df,
    }


# =============================================================================
# SHAP Analysis
# =============================================================================


def compute_shap_values(model, X_test):
    """Compute the SHAP values in order to interpret the model correctly and to show how each feature contributes to predictions."""

    # TreeExplainer for tree-based models
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Since SHAP can return different shapes depending on the model, check whether its:
    # - a list: therefore we have SHAP values for both classes: positive and negative (however, we only need the positive class)
    # - a 3D array: with samples, features and classes (again we only need the positive class)
    if isinstance(shap_values, list):
        # List of arrays with both negative and positive classes
        shap_values = shap_values[1]
    elif len(shap_values.shape) == 3:
        # 3D array (n_samples, n_features, n_classes)
        # Take positive class (index 1 in last dimension)
        shap_values = shap_values[:, :, 1]

    # shap_values shape should now be: (n_samples, n_features)
    # Compute mean absolute SHAP value for each feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    # Make sure that the shape matches
    if len(mean_abs_shap) != len(FEATURES_WITH_MAJOR):
        logger.error(
            "SHAP shape mismatch: got %d values for %d features",
            len(mean_abs_shap),
            len(FEATURES_WITH_MAJOR),
        )
        raise ValueError(
            f"SHAP shape mismatch: {len(mean_abs_shap)} vs {len(FEATURES_WITH_MAJOR)}"
        )

    # Create the required data frame
    shap_importance = pd.DataFrame(
        {"Feature": list(FEATURES_WITH_MAJOR), "SHAP_Importance": list(mean_abs_shap)}
    ).sort_values("SHAP_Importance", ascending=False)

    return {
        "shap_values": shap_values,
        "shap_importance": shap_importance,
        "explainer": explainer,
    }


def compute_shap_per_major(model, X_test):
    """Compute SHAP values separately for each major to understand major-specific feature importance."""

    # Create explainer
    explainer = shap.TreeExplainer(model)

    # Make a dictionary to store the results for each major
    per_major_results = {}

    # Make a list of majors (matching the dummy column names)
    majors = ["The Masters", "PGA Championship", "The Open Championship", "US Open"]

    for major in majors:
        major_col = f"major_{major}"

        # Filter test data for this major only
        major_mask = X_test[major_col] == 1
        X_test_major = X_test[major_mask]

        if len(X_test_major) == 0:
            logger.warning(f"No test data found for {major}, skipping")
            continue

        # Compute SHAP values for this major
        shap_values_major = explainer.shap_values(X_test_major)

        # Handle different SHAP output formats (same logic as before)
        if isinstance(shap_values_major, list):
            shap_values_major = shap_values_major[1]
        elif len(shap_values_major.shape) == 3:
            shap_values_major = shap_values_major[:, :, 1]

        # Compute mean absolute SHAP value for each feature
        mean_abs_shap = np.abs(shap_values_major).mean(axis=0)

        # Create importance DataFrame for this major
        shap_importance_major = pd.DataFrame(
            {
                "Feature": list(FEATURES_WITH_MAJOR),
                "SHAP_Importance": list(mean_abs_shap),
            }
        ).sort_values("SHAP_Importance", ascending=False)

        # Filter out the major dummy variables since they don't provide any meaningful interpretation
        shap_importance_major = shap_importance_major[
            ~shap_importance_major["Feature"].str.startswith("major_")
        ]

        per_major_results[major] = {
            "shap_values": shap_values_major,
            "shap_importance": shap_importance_major,
            "n_samples": len(X_test_major),
        }

    return per_major_results


# =============================================================================
# Main Analysis Function
# =============================================================================


def run_ml_analysis(df, results_dir=None):
    """
    Run machine learning models and save the feature importance to csv.
    Metrics and comparisons are computed separately in evaluation.py.
    """

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
    results["random_forest"] = fit_random_forest(X_train, X_test, y_train, y_test)

    # Save random forest feature importance to its corresponding results file
    results["random_forest"]["importance"].to_csv(
        results_dir / "1_rf_feature_importance.csv", index=False
    )

    # =========================================================================
    # XGBoost
    # =========================================================================
    results["xgboost"] = fit_xgboost(X_train, X_test, y_train, y_test)

    # Save XGBoost feature importance to its corresponding results file
    results["xgboost"]["importance"].to_csv(
        results_dir / "2_xgb_feature_importance.csv", index=False
    )

    # =========================================================================
    # SHAP Analysis
    # =========================================================================
    results["shap"] = compute_shap_values(results["xgboost"]["model"], X_test)

    # Save SHAP importance to its corresponding results file
    results["shap"]["shap_importance"].to_csv(
        results_dir / "3_shap_importance.csv", index=False
    )

    # Per-Major SHAP Analysis
    results["shap_per_major"] = compute_shap_per_major(
        results["xgboost"]["model"], X_test
    )

    # Create combined dataframe with all majors (put features as the index and exclude Major dummies bc not useful)
    all_features = [f for f in FEATURES_WITH_MAJOR if not f.startswith("major_")]
    shap_combined = pd.DataFrame({"Feature": all_features})

    # Add columns for each Major's shap importance and rank
    for major, major_results in results["shap_per_major"].items():
        # Get the SHAP importance for this major (already sorted by importance)
        major_df = major_results["shap_importance"].copy()

        # Filter out major dummies from this dataframe
        major_df = major_df[~major_df["Feature"].str.startswith("major_")]

        # Add rank (1 = highest importance)
        major_df["Rank"] = range(1, len(major_df) + 1)

        # Merge importance and rank into combined dataframe
        shap_combined = shap_combined.merge(
            major_df[["Feature", "SHAP_Importance", "Rank"]], on="Feature", how="left"
        )

        # Rename columns to show which major they're from
        safe_major_name = major.replace(" ", "_")
        shap_combined.rename(
            columns={
                "SHAP_Importance": f"SHAP_{safe_major_name}",
                "Rank": f"Rank_{safe_major_name}",
            },
            inplace=True,
        )

    # Save to single CSV
    shap_combined.to_csv(results_dir / "4_shap_per_major_comparison.csv", index=False)

    return results
