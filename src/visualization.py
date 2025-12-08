"""Visualization functions for Econometric and Machine Learning Models. (Note: Exploratory visualizations are in exploratory.py)"""

import logging
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

# =============================================================================
# ECONOMETRIC MODEL VISUALIZATIONS
# =============================================================================

def plot_pooled_linear_coefficients(coefficients_df, save_path):
    """
    Make a bar chart that shows which performance variables matter the most in the pooled linear regression via the coefficients. 
    The goal is to highlight which variables have the strongest impact on scoring
    Note that lower coefficients mean better performance (negative: improves score).
    """
    # Filter out the intercept bc it isnt needed for the plot
    coef_df = coefficients_df[coefficients_df['Feature'] != 'const'].copy()
    
    # Sort by the raw coefficient value (most negative first)
    coef_df = coef_df.sort_values('Coefficient', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Integrate different color bars based on whether the feature is statistically significant or not.
    colors = ['steelblue' if p < 0.05 else 'lightgray' for p in coef_df['p_value']]
    
    ax.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8) # Add a horizontal like at zero for reference
    ax.set_xlabel('Standardized Coefficient (Negative = Improves Score)', fontsize=12)
    ax.set_title('Pooled Linear Regression: Feature Coefficients\n(Blue = Significant at p<0.05)', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3) # Add a light grid to make the chart easier to read
    plt.tight_layout()
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info("Saved pooled linear coefficients to %s", save_path)
    plt.close()

def plot_per_major_coefficient_heatmap(comparison_df, save_path):
    """
    Heatmap showing how coefficients vary across Majors. In order to help visualize which factirs druve performance in each Major.
    The features are ordered by their average absolute coefficient (most impactful first).
    """
    # The performance variables go on the rows and the Major names on the columns.
    avg_abs_coef = comparison_df.abs().mean(axis=1)  # Compute the average absolute coefficient for each variable across Majors
    comparison_df = comparison_df.loc[avg_abs_coef.sort_values(ascending=False).index] # Reaorder the rows so that the most impactful variables appear first.
    
    plt.figure(figsize=(12, 8))
    
    sns.heatmap(comparison_df.T, annot=True, fmt='.2f', cmap='RdBu_r', center=0, cbar_kws={'label': 'Standardized Coefficient'}, linewidths=0.5)
    
    plt.title('Per-Major Linear Regression: Coefficient Comparison\n(Features Ordered by Average Impact)', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Performance Metric', fontsize=12)
    plt.ylabel('Major Championship', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info("Saved per-major coefficient heatmap to %s", save_path)
    plt.close()

def plot_pooled_logistic_coefficients(coefficients_df, save_path):
    """Bar chart of pooled logistic regression coefficients. Which shows which variables are the best predictors of finishing in the top 25% (sorted by raw coefficients value)."""
    # Filter out intercept since it isnt needed for the plot
    coef_df = coefficients_df[coefficients_df['Feature'] != 'const'].copy()
    
    # Sort by the raw coefficient value (not abs), with the most negative shown first, positive coefficients increase the chance of finishing in the top 25%.
    coef_df = coef_df.sort_values('Coefficient', ascending=True)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Include color bars based on the significance of the coefficients, blue: significant, grey: not significant
    colors = ['steelblue' if p < 0.05 else 'lightgray' for p in coef_df['p_value']]
    
    ax.barh(coef_df['Feature'], coef_df['Coefficient'], color=colors)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8) # Add a vertical line at zero for reference
    ax.set_xlabel('Logistic Coefficient (Positive = Increases Top 25% Probability)', fontsize=12)
    ax.set_title('Pooled Logistic Regression: Feature Coefficients\n(Predicting Top 25%, Blue = Significant at p<0.05)', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3) # Add a light grid to make the chart easier to read
    plt.tight_layout()
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info("Saved pooled logistic coefficients to %s", save_path)
    plt.close()

def plot_interaction_logistic_heatmap(comparison_table, save_path):
    """Heatmap showing how each feature incluences the probability of finishing in the top 25% for each Major.)"""
    # Rows: performance variables, columns: Major names
    # Performance varuables are sorted by their average absolute coefficient so that the most impactful goes first.
    avg_abs_coef = comparison_table.abs().mean(axis=1)  # axis=1 means average across columns (majors)
    comparison_table = comparison_table.loc[avg_abs_coef.sort_values(ascending=False).index] # Take the average across Majors
    
    plt.figure(figsize=(14, 6))
    
    sns.heatmap(comparison_table.T, annot=True, fmt='.2f', cmap='RdBu_r', center=0, cbar_kws={'label': 'Interaction Coefficient'}, linewidths=0.5)
    
    plt.title('Logistic Regression with Interactions: Major-Specific Effects\n(How Features Affect Top 25% Probability Per Major)', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Performance Metric', fontsize=12)
    plt.ylabel('Major Championship', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info("Saved interaction logistic heatmap to %s", save_path)
    plt.close()


def create_econometric_visualizations(econ_results, results_dir):
    """Generate all the above mentioned econometric models visualizations."""
    logger.info("Creating econometric visualizations")
    
    results_dir = Path(results_dir)
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Pooled Linear Regression
    plot_pooled_linear_coefficients(econ_results['pooled_linear']['coefficients'], save_path=figures_dir / "1_pooled_linear_coefficients.png")
    
    # 2. Per-Major Linear Regression
    plot_per_major_coefficient_heatmap(econ_results['per_major_linear']['coefficient_comparison'], save_path=figures_dir / "2_per_major_coefficient_heatmap.png")
    
    # 3a. Pooled Logistic Regression
    plot_pooled_logistic_coefficients(econ_results['pooled_logistic']['coefficients'], save_path=figures_dir / "3a_pooled_logistic_coefficients.png")
    
    # 3b. Logistic Regression with Interactions
    plot_interaction_logistic_heatmap(econ_results['interaction_logistic']['comparison_table'], save_path=figures_dir / "3b_interaction_logistic_heatmap.png")
    
    logger.info("Econometric visualizations complete. Saved to %s", figures_dir)


# =============================================================================
# ML MODEL VISUALIZATIONS
# =============================================================================

def plot_feature_importance_comparison(rf_importance, xgb_importance, shap_importance, save_path):
    """Comparison of ML models (Random Forrest & XGBoost) and SHAP feature importance."""
    # Keep only performance variables (exclude Major dummies)
    rf_all = rf_importance[~rf_importance['Feature'].str.startswith('major_')]
    xgb_all = xgb_importance[~xgb_importance['Feature'].str.startswith('major_')]
    shap_all = shap_importance[~shap_importance['Feature'].str.startswith('major_')]
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 8))
    
    # Random Forest feature importance
    ax1.barh(rf_all['Feature'], rf_all['Importance'], color='steelblue')
    ax1.set_xlabel('Importance', fontsize=12)
    ax1.set_title('Random Forest\nFeature Importance', fontsize=13, fontweight='bold')
    ax1.invert_yaxis()
    ax1.grid(axis='x', alpha=0.3)
    
    # XGBoost feature importance
    ax2.barh(xgb_all['Feature'], xgb_all['Importance'], color='darkorange')
    ax2.set_xlabel('Importance', fontsize=12)
    ax2.set_title('XGBoost\nFeature Importance', fontsize=13, fontweight='bold')
    ax2.invert_yaxis()
    ax2.grid(axis='x', alpha=0.3)
    
    # SHAP feature importance
    ax3.barh(shap_all['Feature'], shap_all['SHAP_Importance'], color='forestgreen')
    ax3.set_xlabel('Importance', fontsize=12)
    ax3.set_title('SHAP\nFeature Importance', fontsize=13, fontweight='bold')
    ax3.invert_yaxis()
    ax3.grid(axis='x', alpha=0.3)
    
    plt.suptitle('ML Models and SHAP Feature Importance Comparison', fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info("Saved feature importance comparison to %s", save_path)
    plt.close()

def plot_per_major_shap_heatmap(shap_per_major_df, save_path):
    """
    Heatmap showing how SHAP feature importance varies across Major Championships. 
    Answers the question: "Which skills matter the most at each Major?"
    The features are ordered by their average SHAP importance, in order to show the most important ones first.
    """
    # Keep only the SHAP importance columns (ignore the rankings)
    shap_cols = [col for col in shap_per_major_df.columns if col.startswith('SHAP_')]
    shap_data = shap_per_major_df[['Feature'] + shap_cols].set_index('Feature')
    
    # Remove the SHAP_ prefix and format the column names nicely

    shap_data.columns = [col.replace('SHAP_', '').replace('_', ' ') for col in shap_data.columns]
    
    # Sort the features by their average SHAP importance across Majors (done after transposing)
    avg_importance = shap_data.mean(axis=1)  # axis=1 averages across columns (majors)
    shap_data = shap_data.loc[avg_importance.sort_values(ascending=False).index]
    
    plt.figure(figsize=(14, 8))
    
    sns.heatmap(shap_data.T, annot=True, fmt='.3f', cmap='YlOrRd', cbar_kws={'label': 'SHAP Importance'}, linewidths=0.5)
    
    plt.title('SHAP Feature Importance by Major Championship\n(How Performance Metrics Drive Top 25% Predictions Per Major)', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Performance Metric', fontsize=12)
    plt.ylabel('Major Championship', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info("Saved per-major SHAP heatmap to %s", save_path)
    plt.close()


def plot_confusion_matrices(rf_results, xgb_results, save_path):
    """Confusion matrices for the Random Forrest and the XGBoost models, compared side by side for better visualization. Include prediction quality with percentages and overall accuracy."""
    from sklearn.metrics import confusion_matrix, accuracy_score
    import numpy as np
    
    # Build the confusion matrices for each model
    cm_rf = confusion_matrix(rf_results['y_test'], rf_results['y_pred'])
    cm_xgb = confusion_matrix(xgb_results['y_test'], xgb_results['y_pred'])
    
    # Compute the overall accuracy (as a percentage, per model)
    acc_rf = accuracy_score(rf_results['y_test'], rf_results['y_pred']) * 100
    acc_xgb = accuracy_score(xgb_results['y_test'], xgb_results['y_pred']) * 100
    
    # Turn each confusion matrix into percentages
    cm_rf_pct = cm_rf / cm_rf.sum() * 100
    cm_xgb_pct = cm_xgb / cm_xgb.sum() * 100
    
    # Create cell labels containing both the count and its corresponding percentage
    labels_rf = np.array([[f'{count}\n({pct:.1f}%)' for count, pct in zip(row_counts, row_pcts)] for row_counts, row_pcts in zip(cm_rf, cm_rf_pct)])
    labels_xgb = np.array([[f'{count}\n({pct:.1f}%)' for count, pct in zip(row_counts, row_pcts)] for row_counts, row_pcts in zip(cm_xgb, cm_xgb_pct)])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Random Forest comfusion matrix
    sns.heatmap(cm_rf, annot=labels_rf, fmt='', cmap='Blues', ax=ax1, xticklabels=['Rest', 'Top 25%'], yticklabels=['Rest', 'Top 25%'], cbar_kws={'label': 'Count'})
    ax1.set_title(f'Random Forest\nConfusion Matrix\n(Overall Accuracy: {acc_rf:.1f}%)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Actual', fontsize=12)
    ax1.set_xlabel('Predicted', fontsize=12)
    
    # XGBoost confusion matrix
    sns.heatmap(cm_xgb, annot=labels_xgb, fmt='', cmap='Oranges', ax=ax2, xticklabels=['Rest', 'Top 25%'], yticklabels=['Rest', 'Top 25%'], cbar_kws={'label': 'Count'})
    ax2.set_title(f'XGBoost\nConfusion Matrix\n(Overall Accuracy: {acc_xgb:.1f}%)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Actual', fontsize=12)
    ax2.set_xlabel('Predicted', fontsize=12)
    
    plt.suptitle('ML Model Prediction Quality', fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    logger.info("Saved confusion matrices to %s", save_path)
    plt.close()

def create_ml_visualizations(ml_results, results_dir):
    """Generate all the above mentioned Machine Learning models visualizations."""
    logger.info("Creating ML visualizations")
    
    results_dir = Path(results_dir)
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Feature Importance Comparison
    plot_feature_importance_comparison(ml_results['random_forest']['importance'], ml_results['xgboost']['importance'], ml_results['shap']['shap_importance'], save_path=figures_dir / "1_feature_importance_comparison.png")
    
    # 2. Per-Major SHAP Heatmap (load the csv)
    shap_csv_path = results_dir / "4_shap_per_major_comparison.csv"
    shap_df = pd.read_csv(shap_csv_path)
    plot_per_major_shap_heatmap(shap_df, save_path=figures_dir / "2_per_major_shap_heatmap.png")
    
    # 3. Confusion Matrices for Random Forrest and XGBoost
    plot_confusion_matrices(ml_results['random_forest'], ml_results['xgboost'], save_path=figures_dir / "3_confusion_matrices.png")
    
    logger.info("ML visualizations complete. Saved to %s", figures_dir)


# =============================================================================
# Script Entry Point
# =============================================================================

if __name__ == "__main__":
    import sys
    
    # Add src directory to path
    SRC_DIR = Path(__file__).parent
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("To generate visualizations, import and call:")
    print("  create_econometric_visualizations(econ_results, results_dir)")
    print("  create_ml_visualizations(ml_results, results_dir)")