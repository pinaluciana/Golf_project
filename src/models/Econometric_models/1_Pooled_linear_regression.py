"""
Model 1: Pooled Linear Regression

Goal: Identify which performance variables have the strongest relationship
with scoring across all Major Championships combined.

This model treats all tournaments as one dataset to find universal
performance drivers.
"""

import sys
from pathlib import Path
import logging

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add src directory to path for imports
SRC_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(SRC_DIR))

try:
    from data_loader import load_combined_data
except ImportError as e:
    logger.error("Failed to import data_loader: %s", e)
    logger.error("Make sure data_loader.py is in the src/ directory")
    raise


# =============================================================================
# Feature Definitions
# =============================================================================

# Performance metrics grouped by category
# Excluded to avoid multicollinearity:
# - sg_total: sum of all strokes gained metrics
# - sg_t2g: sum of sg_ott + sg_app + sg_arg
# - sg_bs: sum of sg_ott + sg_app

OFF_TEE = ['distance', 'accuracy', 'sg_ott']
APPROACH = ['sg_app', 'prox_fw', 'prox_rgh']
SHORT_GAME = ['sg_arg', 'scrambling']
PUTTING = ['sg_putt']
BALL_STRIKING = ['gir']
SHOT_QUALITY = ['great_shots', 'poor_shots']

FEATURES = OFF_TEE + APPROACH + SHORT_GAME + PUTTING + BALL_STRIKING + SHOT_QUALITY


# =============================================================================
# Model Functions
# =============================================================================

def prepare_data(df):
    """
    Prepare features and target for pooled linear regression.

    Args:
        df: DataFrame with golf performance data

    Returns:
        X_scaled: Standardized feature DataFrame
        y: Target variable (total_score)
        scaler: Fitted StandardScaler for later use

    Raises:
        ValueError: If required columns are missing
    """
    logger.info("Preparing data for pooled linear regression")

    # Validate required columns exist
    missing_features = [f for f in FEATURES if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")

    if 'total_score' not in df.columns:
        raise ValueError("Missing required target column: total_score")

    X = df[FEATURES]
    y = df['total_score']

    # Check for missing values
    if X.isnull().any().any():
        logger.warning("Missing values detected in features, dropping rows")
        valid_idx = X.dropna().index
        X = X.loc[valid_idx]
        y = y.loc[valid_idx]

    # Standardize features since they're on different scales
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=FEATURES,
        index=X.index
    )

    logger.info("Using %d features: %s", len(FEATURES), FEATURES)
    logger.info("Training on %d records", len(X))

    return X_scaled, y, scaler


def fit_pooled_linear_regression(X_scaled, y):
    """
    Fit pooled OLS regression model using statsmodels.

    Args:
        X_scaled: Standardized features
        y: Target variable

    Returns:
        model: Fitted statsmodels OLS model

    Raises:
        ValueError: If model fitting fails
    """
    logger.info("Fitting pooled linear regression model")

    try:
        X_with_const = sm.add_constant(X_scaled)
        model = sm.OLS(y, X_with_const).fit()

        logger.info("Model R²: %.3f", model.rsquared)
        logger.info("Model Adjusted R²: %.3f", model.rsquared_adj)

        return model

    except Exception as err:
        logger.error("Failed to fit model: %s", err)
        raise ValueError(f"Model fitting failed: {err}") from err


def extract_coefficients(model):
    """
    Extract coefficients with significance information.

    Args:
        model: Fitted statsmodels model

    Returns:
        DataFrame with coefficients, std errors, p-values, and significance
    """
    coefficients = pd.DataFrame({
        'Feature': model.params.index[1:],
        'Coefficient': model.params.values[1:],
        'Std_Error': model.bse.values[1:],
        'p_value': model.pvalues.values[1:],
        'CI_lower': model.conf_int().iloc[1:, 0].values,
        'CI_upper': model.conf_int().iloc[1:, 1].values,
        'Significant': model.pvalues.values[1:] < 0.05
    }).sort_values('Coefficient', key=abs, ascending=False)

    return coefficients


# =============================================================================
# Main Analysis Function
# =============================================================================

def run_pooled_linear_regression(df, results_dir=None):
    """
    Run the complete pooled linear regression analysis.

    Args:
        df: DataFrame with golf performance data
        results_dir: Directory to save results (optional)

    Returns:
        Dictionary containing model, coefficients, and metrics

    Raises:
        ValueError: If data preparation or model fitting fails
        IOError: If results cannot be saved
    """
    logger.info("Starting pooled linear regression analysis")

    # Setup results directory
    if results_dir is None:
        root = Path(__file__).parent.parent.parent.parent
        results_dir = root / "results" / "2_Econometric_models" / "1_Pooled_linear_regression"
    else:
        results_dir = Path(results_dir)

    try:
        results_dir.mkdir(parents=True, exist_ok=True)
    except OSError as err:
        logger.error("Failed to create results directory: %s", err)
        raise IOError(f"Cannot create results directory: {err}") from err

    # Prepare data
    X_scaled, y, scaler = prepare_data(df)

    # Fit model
    model = fit_pooled_linear_regression(X_scaled, y)

    # Print summary
    print("\nPOOLED LINEAR REGRESSION RESULTS")
    print("=" * 80)
    print(model.summary())

    # Extract coefficients
    coefficients = extract_coefficients(model)
    print("\nFeature Importance (Standardized Coefficients):")
    print("Note: Coefficients show impact of 1 standard deviation change")
    print(coefficients.to_string(index=False))

    # Save coefficients to CSV
    try:
        coefficients.to_csv(results_dir / "coefficients.csv", index=False)
        logger.info("Saved coefficients to %s", results_dir / "coefficients.csv")
    except OSError as err:
        logger.warning("Could not save coefficients CSV: %s", err)

    # Save model summary to text file
    try:
        with open(results_dir / "model_summary.txt", 'w', encoding='utf-8') as f:
            f.write("POOLED LINEAR REGRESSION RESULTS\n")
            f.write("=" * 80 + "\n")
            f.write(str(model.summary()))
        logger.info("Saved model summary to %s", results_dir / "model_summary.txt")
    except OSError as err:
        logger.warning("Could not save model summary: %s", err)

    # Compile results
    results = {
        'model': model,
        'coefficients': coefficients,
        'scaler': scaler,
        'X_scaled': X_scaled,
        'y': y,
        'y_pred': model.fittedvalues,
        'residuals': y - model.fittedvalues,
        'metrics': {
            'r_squared': model.rsquared,
            'adj_r_squared': model.rsquared_adj,
            'rmse': np.sqrt(model.mse_resid),
            'n_observations': int(model.nobs)
        }
    }

    logger.info("Pooled linear regression analysis complete")
    print(f"\nResults saved to: {results_dir}")

    return results


# =============================================================================
# Script Entry Point
# =============================================================================

if __name__ == "__main__":
    try:
        # Load data
        logger.info("Loading combined dataset")
        data = load_combined_data()
        logger.info("Loaded %d player-tournament records", len(data))

        # Run analysis
        model_results = run_pooled_linear_regression(data)

        # Print summary metrics
        print("\n" + "=" * 60)
        print("MODEL SUMMARY")
        print("=" * 60)
        print(f"R²: {model_results['metrics']['r_squared']:.3f}")
        print(f"Adjusted R²: {model_results['metrics']['adj_r_squared']:.3f}")
        print(f"RMSE: {model_results['metrics']['rmse']:.3f} strokes")
        print(f"Observations: {model_results['metrics']['n_observations']}")

        # Generate plots using evaluation module
        print("\nGenerating plots...")
        try:
            from evaluation import (
                plot_pooled_linear_coefficients,
                plot_residuals,
                plot_actual_vs_predicted
            )

            # Setup figures directory
            project_root = Path(__file__).parent.parent.parent.parent
            fig_dir = project_root / "results" / "2_Econometric_models"
            fig_dir = fig_dir / "1_Pooled_linear_regression" / "figures"
            fig_dir.mkdir(parents=True, exist_ok=True)

            plot_pooled_linear_coefficients(
                model_results['coefficients'],
                save_path=fig_dir / "coefficient_importance.png"
            )
            plot_residuals(
                model_results['y'],
                model_results['y_pred'],
                title='Pooled Linear Regression: Residual Plot',
                save_path=fig_dir / "residual_plot.png"
            )
            plot_actual_vs_predicted(
                model_results['y'],
                model_results['y_pred'],
                title='Pooled Linear Regression: Actual vs Predicted',
                save_path=fig_dir / "actual_vs_predicted.png"
            )

            print(f"Plots saved to: {fig_dir}")

        except ImportError as err:
            logger.warning("Could not import evaluation module: %s", err)
            logger.warning("Skipping plot generation")

    except FileNotFoundError as err:
        logger.error("Data file not found: %s", err)
        sys.exit(1)
    except (ValueError, KeyError) as err:
        logger.error("Data processing error: %s", err)
        sys.exit(1)