"""
Model 3: Pooled Logistic Regression (with Major Interactions Extension)

Goal: Predict the probability of finishing in the top 25% of each tournament,
identifying which performance metrics best distinguish top performers.

This module includes:
1. Pooled Logistic Regression - baseline model
2. Logistic Regression with Major Interactions - captures tournament-specific effects
"""

import sys
from pathlib import Path
import logging

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

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
# Feature Definitions
# =============================================================================

OFF_TEE = ['distance', 'accuracy', 'sg_ott']
APPROACH = ['sg_app', 'prox_fw', 'prox_rgh']
SHORT_GAME = ['sg_arg', 'scrambling']
PUTTING = ['sg_putt']
BALL_STRIKING = ['gir']
SHOT_QUALITY = ['great_shots', 'poor_shots']

FEATURES = OFF_TEE + APPROACH + SHORT_GAME + PUTTING + BALL_STRIKING + SHOT_QUALITY


# =============================================================================
# Data Preparation Functions
# =============================================================================

def create_top25_target(df):
    """
    Create binary target indicating top 25% finish per tournament.

    Top 25% is calculated per tournament (major + year) based on total_score.
    Lower scores are better in golf, so top 25% = score <= 25th percentile.

    Args:
        df: DataFrame with golf performance data

    Returns:
        DataFrame with 'top_25' column added
    """
    df = df.copy()

    df['tournament_25th_percentile'] = df.groupby(
        ['major', 'year']
    )['total_score'].transform(lambda x: x.quantile(0.25))

    df['top_25'] = (
        df['total_score'] <= df['tournament_25th_percentile']
    ).astype(int)

    logger.info("Created top 25%% target:")
    logger.info(
        "  Top 25%%: %d (%.1f%%)",
        df['top_25'].sum(),
        df['top_25'].sum() / len(df) * 100
    )
    logger.info(
        "  Rest of field: %d (%.1f%%)",
        (df['top_25'] == 0).sum(),
        (df['top_25'] == 0).sum() / len(df) * 100
    )

    return df


def prepare_logistic_data(df):
    """
    Prepare standardized features for logistic regression.

    Args:
        df: DataFrame with 'top_25' column

    Returns:
        X_scaled: Standardized features
        y: Binary target
        scaler: Fitted StandardScaler
    """
    X = df[FEATURES]
    y = df['top_25']

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=FEATURES,
        index=X.index
    )

    return X_scaled, y, scaler


# =============================================================================
# Pooled Logistic Regression
# =============================================================================

def fit_pooled_logistic(X_scaled, y):
    """
    Fit pooled logistic regression using statsmodels.

    Args:
        X_scaled: Standardized features
        y: Binary target

    Returns:
        Fitted statsmodels Logit model
    """
    logger.info("Fitting pooled logistic regression")

    X_with_const = sm.add_constant(X_scaled)
    model = sm.Logit(y, X_with_const).fit(disp=0)

    return model


def evaluate_logistic_model(model, X_scaled, y):
    """
    Evaluate logistic regression model performance.

    Args:
        model: Fitted statsmodels Logit model
        X_scaled: Standardized features
        y: Actual binary target

    Returns:
        Dictionary with evaluation metrics
    """
    X_with_const = sm.add_constant(X_scaled)
    y_pred_proba = model.predict(X_with_const)
    y_pred = (y_pred_proba >= 0.5).astype(int)

    accuracy = (y_pred == y).mean()
    roc_auc = roc_auc_score(y, y_pred_proba)
    fpr, tpr, thresholds = roc_curve(y, y_pred_proba)
    conf_matrix = confusion_matrix(y, y_pred)

    return {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds,
        'confusion_matrix': conf_matrix
    }


def extract_logistic_coefficients(model):
    """
    Extract coefficients from logistic regression model.

    Args:
        model: Fitted statsmodels Logit model

    Returns:
        DataFrame with coefficients and p-values
    """
    coefficients = pd.DataFrame({
        'Feature': model.params.index[1:],
        'Coefficient': model.params.values[1:],
        'p_value': model.pvalues.values[1:]
    }).sort_values('Coefficient', key=abs, ascending=False)

    return coefficients


# =============================================================================
# Logistic Regression with Major Interactions
# =============================================================================

def create_interaction_features(df):
    """
    Create interaction terms between performance metrics and majors.

    This allows each metric to have different importance at each major.

    Args:
        df: DataFrame with golf performance data

    Returns:
        X_interactions: DataFrame with interaction features
        interaction_features: List of interaction feature names
    """
    logger.info("Creating interaction features")

    majors = df['major'].unique()
    interaction_features = []

    df_copy = df.copy()

    for metric in FEATURES:
        for major in majors:
            interaction_name = f'{metric}_x_{major}'
            df_copy[interaction_name] = (
                df_copy[metric] * (df_copy['major'] == major).astype(int)
            )
            interaction_features.append(interaction_name)

    logger.info("Created %d interaction terms", len(interaction_features))

    X_interactions = df_copy[interaction_features]

    return X_interactions, interaction_features


def fit_logistic_with_interactions(X_interactions, y):
    """
    Fit logistic regression with interaction terms using sklearn.

    Note: Using sklearn instead of statsmodels because:
    - Many interaction terms create numerical instability in statsmodels
    - Sklearn's regularization handles high-dimensional data better
    - Goal is comparing coefficients, not statistical inference

    Args:
        X_interactions: DataFrame with interaction features
        y: Binary target

    Returns:
        Fitted sklearn LogisticRegression model
    """
    logger.info("Fitting logistic regression with interactions")

    model = LogisticRegression(max_iter=10000, random_state=42)
    model.fit(X_interactions, y)

    return model


def extract_interaction_coefficients(model, feature_names):
    """
    Extract and organize interaction coefficients by major.

    Args:
        model: Fitted sklearn LogisticRegression
        feature_names: List of interaction feature names

    Returns:
        comparison_table: DataFrame with metrics as rows, majors as columns
    """
    coef_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': model.coef_[0]
    })

    interaction_coefs = coef_df[coef_df['feature'].str.contains('_x_')].copy()

    interaction_coefs[['metric', 'major']] = interaction_coefs[
        'feature'
    ].str.split('_x_', n=1, expand=True)

    comparison_table = interaction_coefs.pivot(
        index='metric',
        columns='major',
        values='coefficient'
    )

    return comparison_table


# =============================================================================
# Main Analysis Function
# =============================================================================

def run_logistic_regression(df, results_dir=None):
    """
    Run the complete logistic regression analysis (pooled + interactions).

    Args:
        df: DataFrame with golf performance data
        results_dir: Directory to save results (optional)

    Returns:
        Dictionary containing all models and results
    """
    logger.info("Starting logistic regression analysis")

    # Setup results directory
    if results_dir is None:
        root = Path(__file__).parent.parent.parent.parent
        results_dir = root / "results" / "2_Econometric_models"
        results_dir = results_dir / "3_Logistic_regression"
    else:
        results_dir = Path(results_dir)

    results_dir.mkdir(parents=True, exist_ok=True)

    # Prepare data
    df = create_top25_target(df)
    X_scaled, y, scaler = prepare_logistic_data(df)

    # =========================================================================
    # Part 1: Pooled Logistic Regression
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 1: POOLED LOGISTIC REGRESSION")
    print("=" * 80)

    pooled_model = fit_pooled_logistic(X_scaled, y)
    print(pooled_model.summary())

    # Evaluate
    pooled_eval = evaluate_logistic_model(pooled_model, X_scaled, y)
    print(f"\nAccuracy: {pooled_eval['accuracy']:.3f}")
    print(f"ROC-AUC Score: {pooled_eval['roc_auc']:.3f}")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(
        y,
        pooled_eval['y_pred'],
        target_names=['Rest of Field', 'Top 25%']
    ))

    # Extract coefficients
    pooled_coefs = extract_logistic_coefficients(pooled_model)
    print("\nFeature Importance (Logistic Regression Coefficients):")
    print(pooled_coefs.to_string(index=False))

    # Confusion matrix details
    conf_mat = pooled_eval['confusion_matrix']
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives: {conf_mat[0, 0]}")
    print(f"  False Positives: {conf_mat[0, 1]}")
    print(f"  False Negatives: {conf_mat[1, 0]}")
    print(f"  True Positives: {conf_mat[1, 1]}")

    # Save pooled results
    pooled_coefs.to_csv(results_dir / "pooled_coefficients.csv", index=False)
    logger.info(
        "Saved pooled coefficients to %s",
        results_dir / "pooled_coefficients.csv"
    )

    # =========================================================================
    # Part 2: Logistic Regression with Major Interactions
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 2: LOGISTIC REGRESSION WITH MAJOR INTERACTIONS")
    print("=" * 80)

    # Create interaction features
    X_interactions, interaction_features = create_interaction_features(df)

    # Fit model
    interaction_model = fit_logistic_with_interactions(X_interactions, y)

    # Evaluate
    y_pred_interact = interaction_model.predict(X_interactions)
    y_pred_proba_interact = interaction_model.predict_proba(X_interactions)[:, 1]
    accuracy_interact = (y_pred_interact == y).mean()
    roc_auc_interact = roc_auc_score(y, y_pred_proba_interact)

    print(f"\nAccuracy: {accuracy_interact:.3f}")
    print(f"ROC-AUC Score: {roc_auc_interact:.3f}")

    # Extract interaction coefficients
    comparison_table = extract_interaction_coefficients(
        interaction_model,
        interaction_features
    )

    print("\nCoefficient Comparison Across Majors:")
    print(comparison_table.round(3))

    # Top 3 metrics by major
    print("\nTop 3 Metrics by Major (Absolute Impact):")
    for major in comparison_table.columns:
        top3 = comparison_table[major].abs().nlargest(3)
        print(f"\n{major}:")
        for metric, coef in top3.items():
            print(f"  {metric}: {coef:.3f}")

    # Save interaction results
    comparison_table.to_csv(results_dir / "interaction_coefficients.csv")
    logger.info(
        "Saved interaction coefficients to %s",
        results_dir / "interaction_coefficients.csv"
    )

    # Save model summary
    with open(results_dir / "model_summary.txt", 'w', encoding='utf-8') as f:
        f.write("LOGISTIC REGRESSION RESULTS\n")
        f.write("=" * 80 + "\n\n")

        f.write("PART 1: POOLED LOGISTIC REGRESSION\n")
        f.write("-" * 40 + "\n")
        f.write(f"Accuracy: {pooled_eval['accuracy']:.3f}\n")
        f.write(f"ROC-AUC: {pooled_eval['roc_auc']:.3f}\n\n")
        f.write("Coefficients:\n")
        f.write(pooled_coefs.to_string(index=False))

        f.write("\n\n" + "=" * 80 + "\n")
        f.write("PART 2: LOGISTIC WITH INTERACTIONS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Accuracy: {accuracy_interact:.3f}\n")
        f.write(f"ROC-AUC: {roc_auc_interact:.3f}\n\n")
        f.write("Coefficient Comparison:\n")
        f.write(comparison_table.round(3).to_string())

    logger.info("Saved model summary to %s", results_dir / "model_summary.txt")

    # Compile all results
    results = {
        'pooled': {
            'model': pooled_model,
            'coefficients': pooled_coefs,
            'evaluation': pooled_eval,
            'y': y
        },
        'interactions': {
            'model': interaction_model,
            'comparison_table': comparison_table,
            'accuracy': accuracy_interact,
            'roc_auc': roc_auc_interact
        },
        'scaler': scaler
    }

    logger.info("Logistic regression analysis complete")
    print(f"\nResults saved to: {results_dir}")

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
    model_results = run_logistic_regression(data)

    # Print final summary
    print("\n" + "=" * 60)
    print("MODEL SUMMARY")
    print("=" * 60)
    print("\nPooled Logistic Regression:")
    print(f"  Accuracy: {model_results['pooled']['evaluation']['accuracy']:.3f}")
    print(f"  ROC-AUC: {model_results['pooled']['evaluation']['roc_auc']:.3f}")
    print("\nLogistic Regression with Interactions:")
    print(f"  Accuracy: {model_results['interactions']['accuracy']:.3f}")
    print(f"  ROC-AUC: {model_results['interactions']['roc_auc']:.3f}")

    # Generate plots using evaluation module
    print("\nGenerating plots...")
    from evaluation import (
        plot_logistic_coefficients,
        plot_confusion_matrix,
        plot_roc_curve,
        plot_interaction_heatmap,
        plot_key_metrics_by_major
    )

    # Setup figures directory
    project_root = Path(__file__).parent.parent.parent.parent
    fig_dir = project_root / "results" / "2_Econometric_models"
    fig_dir = fig_dir / "3_Logistic_regression" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Pooled model plots
    plot_logistic_coefficients(
        model_results['pooled']['coefficients'],
        save_path=fig_dir / "pooled_coefficients.png"
    )
    plot_confusion_matrix(
        model_results['pooled']['y'],
        model_results['pooled']['evaluation']['y_pred'],
        save_path=fig_dir / "confusion_matrix.png"
    )
    plot_roc_curve(
        model_results['pooled']['y'],
        model_results['pooled']['evaluation']['y_pred_proba'],
        model_results['pooled']['evaluation']['roc_auc'],
        save_path=fig_dir / "roc_curve.png"
    )

    # Interaction model plots
    plot_interaction_heatmap(
        model_results['interactions']['comparison_table'],
        save_path=fig_dir / "interaction_heatmap.png"
    )
    plot_key_metrics_by_major(
        model_results['interactions']['comparison_table'],
        save_path=fig_dir / "key_metrics_by_major.png"
    )

    print(f"Plots saved to: {fig_dir}")