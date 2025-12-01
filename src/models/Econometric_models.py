"""
Econometric Models to identify Performance drivers in golf Majors Championships.
Main goals: 
- Explain score based on performance variables
- Identify how skill importance varies across tournaments
- Model the probability of  finishing in the top 25% of the leaderboard

This module contains three regression models:
1. Pooled Linear Regression: universal performance drivers across all majors
2. Per-Major Linear Regression: how skill importance varies by major
3. Logistic Regression (with extension on Major Interactions): predicting top 25% finishers

Using mainly the statsmodels module since, this file's goal is explanatory and not predictive.
"""

import sys
import logging
from pathlib import Path

# Add src directory to path for local imports (to be able to find files inside src)
SRC_DIR = Path(__file__).parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report

logger = logging.getLogger(__name__)

# =============================================================================
# Feature Definitions
# =============================================================================

# Performance metrics that aren't taken into consideration (to avoid multicollinearity):
# - sg_total: bc it's the sum of all strokes gained metrics
# - sg_t2g: sum of sg_ott + sg_app + sg_arg
# - sg_bs: sum of sg_ott + sg_app

OFF_TEE = ['distance', 'accuracy', 'sg_ott']
APPROACH = ['sg_app', 'prox_fw', 'prox_rgh']
SHORT_GAME = ['sg_arg', 'scrambling']
PUTTING = ['sg_putt']
BALL_STRIKING = ['gir']
SHOT_QUALITY = ['great_shots', 'poor_shots']

# Combine all features together
FEATURES = OFF_TEE + APPROACH + SHORT_GAME + PUTTING + BALL_STRIKING + SHOT_QUALITY

# =============================================================================
# Data Preparation
# =============================================================================

def prepare_features(df, features=None):
    """
    Prepare and standardize features for regression. Standardize: so we're able to compare coefficients across features on different scales.
    """
    if features is None:
        features = FEATURES
    
    X = df[features]
    y = df['total_score']
    
    # Check for missing values and drop if found
    if X.isnull().any().any():
        logger.warning("Missing values detected, dropping rows")
        valid_idx = X.dropna().index
        X = X.loc[valid_idx]
        y = y.loc[valid_idx]
    
    # Standardize features since they're on different scales
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=features, index=X.index)
    
    return X_scaled, y, scaler

def create_top25_target(df):
    """
    Create binary target indicating top 25% finish per tournament. Top 25% is calculated per tournament (major & year) based on total_score.
    Note: lower scores are better in golf, so top 25% = score <= 25th percentile.
    """
    df = df.copy()
    
    df['tournament_25th_percentile'] = df.groupby(
        ['major', 'year'])['total_score'].transform(lambda x: x.quantile(0.25))
    
    df['top_25'] = (df['total_score'] <= df['tournament_25th_percentile']).astype(int)
    
    logger.info("Created top 25%% target:")
    logger.info("  Top 25%%: %d (%.1f%%)", df['top_25'].sum(), 
                df['top_25'].sum() / len(df) * 100)
    logger.info("  Rest of field: %d (%.1f%%)", (df['top_25'] == 0).sum(),
                (df['top_25'] == 0).sum() / len(df) * 100)
    
    return df

# =============================================================================
# Model 1: Pooled Linear Regression
# =============================================================================

def fit_pooled_linear(df):
    """
    Fit pooled OLS regression across all majors.
    Goal: identify which performance variables have the strongest relationship with scoring across all Major Championships combined, to get an overall idea.
    """
    logger.info("Fitting pooled linear regression")
    
    X_scaled, y, scaler = prepare_features(df)
    
    # Add constant for intercept
    X_with_const = sm.add_constant(X_scaled)
    
    # Fit model using statsmodels for p-values and confidence intervals
    model = sm.OLS(y, X_with_const).fit()
    
    # Extract coefficients with statistical info
    coefficients = pd.DataFrame({'Feature': model.params.index,
        'Coefficient': model.params.values,
        'Std_Error': model.bse.values,
        't_value': model.tvalues.values,
        'p_value': model.pvalues.values,
        'CI_lower': model.conf_int().iloc[:, 0].values,
        'CI_upper': model.conf_int().iloc[:, 1].values})
    
    logger.info("Pooled linear regression fitted successfully")
    
    return {'model': model,
        'coefficients': coefficients,
        'scaler': scaler,
        'y_pred': model.fittedvalues,
        'y_true': y,
        'n_features': len(FEATURES)}

# =============================================================================
# Model 2: Per-Major Linear Regression
# =============================================================================

def fit_per_major_linear(df):
    """
    Fit a separate OLS regression for each major.
    Goal: identify how skill importance varies across different Major championships, since ach major has unique characteristics that might favor different skills.
    """
    logger.info("Fitting per-major linear regressions")
    
    results = {}
    
    for major in df['major'].unique():
        major_df = df[df['major'] == major]
        
        # Standardize within each major separately
        X_scaled, y, scaler = prepare_features(major_df)
        X_with_const = sm.add_constant(X_scaled)
        model = sm.OLS(y, X_with_const).fit()
        
        # Create comprehensive coefficients DataFrame for this major
        major_coefficients = pd.DataFrame({'Feature': model.params.index,
            'Coefficient': model.params.values,
            'Std_Error': model.bse.values,
            't_value': model.tvalues.values,
            'p_value': model.pvalues.values,
            'CI_lower': model.conf_int().iloc[:, 0].values,
            'CI_upper': model.conf_int().iloc[:, 1].values})
        
        results[major] = {'model': model,
            'y_true': y,
            'y_pred': model.fittedvalues,
            'coefficients_df': major_coefficients,
            'n_obs': len(major_df),
            'n_features': len(FEATURES)}
        
        logger.info("%s: fitted successfully, n=%d", major, len(major_df))
    
    # Create coefficient comparison table (with features as rows and majors as columns)
    # Use only the Coefficient column from each major's DataFrame
    coef_dict = {}
    for major in df['major'].unique():
        coef_df = results[major]['coefficients_df']
        # Create a Series with Feature as index and Coefficient as values (exclude constant)
        coef_series = coef_df[coef_df['Feature'] != 'const'].set_index('Feature')['Coefficient']
        coef_dict[major] = coef_series
    
    results['coefficient_comparison'] = pd.DataFrame(coef_dict)
    return results

# =============================================================================
# Model 3: Logistic Regression (Pooled and extension with Major Interactions)
# =============================================================================

def fit_pooled_logistic(df):
    """
    Fit pooled logistic regression using statsmodels.
    Goal: predict the probability of finishing in the top 25% of each tournament, in order to identify which performance variables distinguish top performers.
    """
    logger.info("Fitting pooled logistic regression")
    
    # Prepare the data
    X_scaled, _, scaler = prepare_features(df)
    y = df.loc[X_scaled.index, 'top_25']
    
    # Add a constant and fit the model
    X_with_const = sm.add_constant(X_scaled)
    model = sm.Logit(y, X_with_const).fit(disp=0)  # disp=0 suppresses output
    
    # Include predictions
    y_pred_proba = model.predict(X_with_const)
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # Extract the coefficients
    coefficients = pd.DataFrame({'Feature': model.params.index,
        'Coefficient': model.params.values,
        'Std_Error': model.bse.values,
        'z_value': model.tvalues.values,
        'p_value': model.pvalues.values,
        'CI_lower': model.conf_int().iloc[:, 0].values,
        'CI_upper': model.conf_int().iloc[:, 1].values})
    
    logger.info("Pooled logistic regression fitted successfully")
    
    return {'model': model,
        'coefficients': coefficients,
        'y_true': y,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba}

def fit_logistic_with_interactions(df):
    """
    Added an extension to the logistic regression with major interaction terms.
    This was done to show how the importance of the performance variables changes per major (which is essential for my project).
    Created interaction terms: metric * major for each combination.
    Note: here I used sklearn instead of statsmodels bc:
    - Many interaction terms create numerical instability in statsmodels
    - Sklearn's regularization handles high-dimensional data better
    - Goal here is comparing coefficients, not statistical inference
    """
    logger.info("Fitting logistic regression with interactions")
    
    # Create interaction features
    majors = df['major'].unique()
    interaction_features = []
    df_copy = df.copy()
    
    for metric in FEATURES:
        for major in majors:
            col_name = f'{metric}_x_{major}'
            df_copy[col_name] = df_copy[metric] * (df_copy['major'] == major).astype(int)
            interaction_features.append(col_name)
    
    logger.info("Created %d interaction terms", len(interaction_features))
    
    X = df_copy[interaction_features]
    y = df_copy['top_25']
    
    # Fit model with sklearn
    model = LogisticRegression(max_iter=10000, random_state=42)
    model.fit(X, y)
    
    # Make predictions
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]

    # Extract coefficients and organize by major
    coef_df = pd.DataFrame({'feature': interaction_features, 'coefficient': model.coef_[0]})
    
    # Split feature name to get metric and major
    coef_df[['metric', 'major']] = coef_df['feature'].str.split('_x_', n=1, expand=True)
    
    # Pivot to create comparison table (metrics as rows, majors as columns)
    comparison_table = coef_df.pivot(index='metric', columns='major', values='coefficient')
    
    logger.info("Logistic with interactions fitted successfully")
    
    return {'model': model,
        'comparison_table': comparison_table,
        'y_true': y,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba}

    # =============================================================================
    # Main Analysis Function
    # =============================================================================
def run_econometric_analysis(df, results_dir=None):
    """
    Run all econometric models and save results to CSV.
    """
    logger.info("Starting econometric analysis")
    
    # Setup results directory (golf_project/results/2_Econometric_models)
    if results_dir is None:
        results_dir = Path(__file__).parent.parent.parent / "results" / "2_Econometric_models"
    else:
        results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}

    # =========================================================================
    # Model 1: Pooled Linear Regression
    # =========================================================================
    logger.info("Running Model 1: Pooled Linear Regression")
    results['pooled_linear'] = fit_pooled_linear(df)
    
    # Save comprehensive coefficients to CSV
    results['pooled_linear']['coefficients'].to_csv(
        results_dir / "1_pooled_linear_coefficients.csv", index=False)
    
    # =========================================================================
    # Model 2: Per-Major Linear Regression
    # =========================================================================
    logger.info("Running Model 2: Per-Major Linear Regression")
    results['per_major_linear'] = fit_per_major_linear(df)
    
    # Save each major's coefficients to separate CSV
    for major in df['major'].unique():
        safe_major_name = major.replace(' ', '_').replace("'", "")
        results['per_major_linear'][major]['coefficients_df'].to_csv(
            results_dir / f"2_per_major_{safe_major_name}_coefficients.csv", index=False)
    
    # Save coefficient comparison table
    results['per_major_linear']['coefficient_comparison'].to_csv(
        results_dir / "2_per_major_comparison.csv")
    
    # =========================================================================
    # Model 3: Logistic Regression
    # =========================================================================
    logger.info("Running Model 3: Logistic Regression")
    
    # Add top 25% target
    df = create_top25_target(df)
    
    # Part 1: Pooled Logistic
    results['pooled_logistic'] = fit_pooled_logistic(df)
    results['pooled_logistic']['coefficients'].to_csv(
        results_dir / "3a_logistic_coefficients.csv", index=False)
    
    # Part 2: Logistic with Major Interactions
    results['interaction_logistic'] = fit_logistic_with_interactions(df)
    results['interaction_logistic']['comparison_table'].to_csv(
        results_dir / "3b_interaction_coefficients.csv")
    
    logger.info("Econometric analysis complete. Results saved to %s", results_dir)
    
    return results

# =============================================================================
# Script Entry Point
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    from data_loader import load_combined_data   

    # Load data
    logger.info("Loading combined dataset")
    data = load_combined_data()
    logger.info("Loaded %d player-tournament records", len(data))
    
    # Run analysis (results saded to CSV)
    results = run_econometric_analysis(data)
    
    logger.info("Done with Econometric models")

