"""
Model 2: Per-Major Linear Regression

Goal: Identify how skill importance varies across different Major championships.

Each major has unique characteristics that might favor different skills:
- The Masters: Augusta National's undulating greens
- US Open: Thick rough and narrow fairways
- The Open Championship: Links-style wind and firm conditions
- PGA Championship: Varies by venue
"""

import sys
from pathlib import Path
import logging

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data_loader import load_combined_data

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# Feature Definitions (same as Model 1 for consistency)
# =============================================================================

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

def fit_per_major_regressions(df):
    """
    Fit separate OLS regressions for each Major championship.

    Args:
        df: DataFrame with golf performance data

    Returns:
        Dictionary with results for each major
    """
    logger.info("Running separate regressions for each major")
    logger.info("Using %d features: %s", len(FEATURES), FEATURES)

    majors = df['major'].unique()
    results = {}

    for major in majors:
        major_df = df[df['major'] == major]

        X_major = major_df[FEATURES]
        y_major = major_df['total_score']

        # Standardize within each major
        scaler_major = StandardScaler()
        X_major_scaled = pd.DataFrame(
            scaler_major.fit_transform(X_major),
            columns=FEATURES,
            index=X_major.index
        )

        # Add constant and fit model
        X_major_const = sm.add_constant(X_major_scaled)
        model_major = sm.OLS(y_major, X_major_const).fit()

        # Store results
        results[major] = {
            'model': model_major,
            'r2': model_major.rsquared,
            'adj_r2': model_major.rsquared_adj,
            'rmse': np.sqrt(model_major.mse_resid),
            'coefficients': dict(zip(FEATURES, model_major.params[1:])),
            'pvalues': dict(zip(FEATURES, model_major.pvalues[1:])),
            'n_obs': len(major_df),
            'scaler': scaler_major
        }

        logger.info(
            "%s: R²=%.3f, RMSE=%.3f, n=%d",
            major, model_major.rsquared,
            np.sqrt(model_major.mse_resid), len(major_df)
        )

    return results


def create_coefficient_comparison(results):
    """
    Create DataFrame comparing coefficients across majors.

    Args:
        results: Dictionary from fit_per_major_regressions()

    Returns:
        DataFrame with features as rows, majors as columns
    """
    coef_df = pd.DataFrame({
        major: results[major]['coefficients']
        for major in results.keys()
    })

    return coef_df


def get_top_features_by_major(results, n_top=3):
    """
    Get top N most important features for each major.

    Args:
        results: Dictionary from fit_per_major_regressions()
        n_top: Number of top features to return

    Returns:
        Dictionary with top features for each major
    """
    top_features = {}

    for major in results.keys():
        coeffs = results[major]['coefficients']
        sorted_coeffs = sorted(
            coeffs.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:n_top]
        top_features[major] = sorted_coeffs

    return top_features


# =============================================================================
# Main Analysis Function
# =============================================================================

def run_per_major_linear_regression(df, results_dir=None):
    """
    Run the complete per-major linear regression analysis.

    Args:
        df: DataFrame with golf performance data
        results_dir: Directory to save results (optional)

    Returns:
        Dictionary containing results for each major
    """
    logger.info("Starting per-major linear regression analysis")

    # Setup results directory
    if results_dir is None:
        root = Path(__file__).parent.parent.parent.parent
        results_dir = root / "results" / "2_Econometric_models"
        results_dir = results_dir / "2_Per_major_linear_regression"
    else:
        results_dir = Path(results_dir)

    results_dir.mkdir(parents=True, exist_ok=True)

    # Fit models
    results = fit_per_major_regressions(df)

    # Print summary for each major
    print("\nPER-MAJOR LINEAR REGRESSION RESULTS")
    print("=" * 80)

    for major in results.keys():
        print(f"\n{major}:")
        print(f"  Observations: {results[major]['n_obs']}")
        print(f"  R²: {results[major]['r2']:.3f}")
        print(f"  Adjusted R²: {results[major]['adj_r2']:.3f}")
        print(f"  RMSE: {results[major]['rmse']:.3f} strokes")

    # Create coefficient comparison
    coef_df = create_coefficient_comparison(results)
    print("\nCoefficient Comparison Across Majors:")
    print(coef_df.round(2))

    # Save coefficient comparison to CSV
    coef_df.to_csv(results_dir / "coefficient_comparison.csv")
    logger.info(
        "Saved coefficient comparison to %s",
        results_dir / "coefficient_comparison.csv"
    )

    # Get and print top features
    top_features = get_top_features_by_major(results)
    print("\nTop 3 Most Important Features by Major:")
    print("(Based on absolute coefficient values)")

    for major, features in top_features.items():
        print(f"\n{major}:")
        for i, (feat, coef) in enumerate(features, 1):
            print(f"  {i}. {feat}: {coef:.2f}")

    # Save summary to text file
    with open(results_dir / "model_summary.txt", 'w', encoding='utf-8') as f:
        f.write("PER-MAJOR LINEAR REGRESSION RESULTS\n")
        f.write("=" * 80 + "\n\n")
        for major in results.keys():
            f.write(f"{major}:\n")
            f.write(f"  Observations: {results[major]['n_obs']}\n")
            f.write(f"  R²: {results[major]['r2']:.3f}\n")
            f.write(f"  Adjusted R²: {results[major]['adj_r2']:.3f}\n")
            f.write(f"  RMSE: {results[major]['rmse']:.3f} strokes\n\n")

        f.write("\nCoefficient Comparison:\n")
        f.write(coef_df.round(3).to_string())

    logger.info("Saved model summary to %s", results_dir / "model_summary.txt")
    logger.info("Per-major linear regression analysis complete")
    print(f"\nResults saved to: {results_dir}")

    # Add coef_df to results for plotting
    results['coefficient_comparison'] = coef_df

    return results


# =============================================================================
# Script Entry Point
# =============================================================================

if __name__ == "__main__":
    # Load data
    logger.info("Loading combined dataset")
    data = load_combined_data()
    logger.info("Loaded %d player-tournament records", len(data))

    # Run analysis
    model_results = run_per_major_linear_regression(data)

    # Print final summary
    print("\n" + "=" * 60)
    print("MODEL FIT SUMMARY")
    print("=" * 60)
    for major in model_results.keys():
        if major != 'coefficient_comparison':
            print(f"{major}: R² = {model_results[major]['r2']:.3f}")

    # Generate plots using evaluation module
    print("\nGenerating plots...")
    from evaluation import (
        plot_per_major_coefficient_heatmap,
        plot_per_major_top_features
    )

    # Setup figures directory
    project_root = Path(__file__).parent.parent.parent.parent
    fig_dir = project_root / "results" / "2_Econometric_models"
    fig_dir = fig_dir / "2_Per_major_linear_regression" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    plot_per_major_coefficient_heatmap(
        model_results['coefficient_comparison'],
        save_path=fig_dir / "coefficient_heatmap.png"
    )
    plot_per_major_top_features(
        model_results,
        save_path=fig_dir / "top_features_comparison.png"
    )

    print(f"Plots saved to: {fig_dir}")