"""
Evaluation metrics and model comparison for the golf performance models. This file computes metrics for both econometric and Machine Learning models.
"""

import logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (mean_squared_error, r2_score, accuracy_score, roc_auc_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score)

logger = logging.getLogger(__name__)

# =============================================================================
# Regression Metrics (for econometric models)
# =============================================================================

def compute_regression_metrics(y_true, y_pred):
    """Compute R², RMSE, MAE for linear regression models."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return {'r_squared': r2_score(y_true, y_pred), 'rmse': np.sqrt(mean_squared_error(y_true, y_pred)), 'mae': np.mean(np.abs(y_true - y_pred)), 'n_observations': len(y_true)}

def compute_adjusted_r2(r2, n, p):
    """Compute adjusted R² that penalizes for number of predictors."""
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)

# ==================================================================================
# Classification Metrics (for the logistic regression & the Machine Learning models)
# ==================================================================================

def compute_classification_metrics(y_true, y_pred, y_pred_proba=None):
    """Compute all classification metrics: accuracy, precision, recall, F1, ROC-AUC."""
    metrics = {'accuracy': accuracy_score(y_true, y_pred), 'precision': precision_score(y_true, y_pred, zero_division=0), 'recall': recall_score(y_true, y_pred, zero_division=0),'f1_score': f1_score(y_true, y_pred, zero_division=0)}
    if y_pred_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
    return metrics

def compute_confusion_matrix(y_true, y_pred):
    """Compute confusion matrix."""
    return confusion_matrix(y_true, y_pred)

def get_classification_report_dict(y_true, y_pred):
    """Get classification report as dict for saving to CSV."""
    report = classification_report(y_true, y_pred, target_names=['Rest of Field', 'Top 25%'], output_dict=True)
    return pd.DataFrame(report).transpose()

# =============================================================================
# Print Helper Functions
# =============================================================================

def print_regression_metrics(model_name, metrics):
    """Print regression metrics in consistent format."""
    print(f"\n{model_name}:")
    print(f"  R²:   {metrics['r_squared']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  MAE:  {metrics['mae']:.4f}")
    print(f"  N:    {metrics['n_observations']}")

def print_classification_metrics(model_name, metrics, y_true=None, y_pred=None):
    """Print classification metrics in consistent format."""
    print(f"\n{model_name}:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1_score']:.4f}")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    
    # Print confusion matrix 
    if y_true is not None and y_pred is not None: 
        cm = compute_confusion_matrix(y_true, y_pred)
        total = cm.sum()

        print(f"\n  Confusion Matrix:")
        print(f"                       Predicted")
        print(f"                    Rest      Top 25%")
        print(f"  Actual")
        print(f"  Rest              {cm[0,0]:>6}     {cm[0,1]:>6}    (True Negatives: {cm[0,0]}, False Positives: {cm[0,1]})")
        print(f"  Top 25%           {cm[1,0]:>6}     {cm[1,1]:>6}    (False Negatives: {cm[1,0]}, True Positives: {cm[1,1]})")
        print(f"\n  Correctly Predicted: {cm[0,0] + cm[1,1]} out of {total} ({(cm[0,0] + cm[1,1])/total*100:.1f}%)")

def create_combined_feature_importance(rf_importance, xgb_importance, shap_importance, save_path):
    """Create one combined feature importance CSV with all models and rankings."""
    
    # Get top 10 from each model
    rf_top10 = rf_importance.head(10).copy()
    xgb_top10 = xgb_importance.head(10).copy()
    shap_top10 = shap_importance.head(10).copy()
    
    # Add rankings
    rf_top10['RF_Rank'] = range(1, len(rf_top10) + 1)
    xgb_top10['XGB_Rank'] = range(1, len(xgb_top10) + 1)
    shap_top10['SHAP_Rank'] = range(1, len(shap_top10) + 1)
    
    # Rename importance columns
    rf_top10 = rf_top10.rename(columns={'Importance': 'RF_Importance'})
    xgb_top10 = xgb_top10.rename(columns={'Importance': 'XGB_Importance'})
    shap_top10 = shap_top10.rename(columns={'SHAP_Importance': 'SHAP_Importance'})
    
    # Get all unique features from top 10
    all_features = set(rf_top10['Feature']) | set(xgb_top10['Feature']) | set(shap_top10['Feature'])
    
    # Create combined dataframe
    combined = pd.DataFrame({'Feature': sorted(all_features)})
    
    # Merge each model's data
    combined = combined.merge(rf_top10[['Feature', 'RF_Importance', 'RF_Rank']], on='Feature', how='left')
    combined = combined.merge(xgb_top10[['Feature', 'XGB_Importance', 'XGB_Rank']], on='Feature', how='left')
    combined = combined.merge(shap_top10[['Feature', 'SHAP_Importance', 'SHAP_Rank']], on='Feature', how='left')
    
    # Replace NaN with "-" for missing features
    combined = combined.fillna('-')
    
    # Reorder columns
    combined = combined[['Feature', 'RF_Importance', 'RF_Rank', 'XGB_Importance', 'XGB_Rank', 'SHAP_Importance', 'SHAP_Rank']]
    
    # Save top 10 to CSV
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(save_path, index=False)
    logger.info("Saved combined feature importance to %s", save_path)

    # Print only top 5
    print("\n  Top 5 Features by Model:")
    print("\n  Random Forest:")
    for idx, row in rf_importance.head(5).iterrows():
        print(f"    {row['Feature']:<15} {row['Importance']:>8.4f}")
    
    print("\n  XGBoost:")
    for idx, row in xgb_importance.head(5).iterrows():
        print(f"    {row['Feature']:<15} {row['Importance']:>8.4f}")
    
    print("\n  SHAP:")
    for idx, row in shap_importance.head(5).iterrows():
        print(f"    {row['Feature']:<15} {row.iloc[1]:>8.4f}")
    
    return combined

# =============================================================================
# Model Comparison
# =============================================================================

def compare_regression_models(results_dict, save_path=None):
    """Compare the different regression models and save to a csv file."""
    comparison_data = {}
    
    for model_name, results in results_dict.items():
        metrics = compute_regression_metrics(results['y_test'], results['y_pred'])
        comparison_data[model_name] = metrics
    
    # Create comparison dataframe with metrics as rows and models as columns
    comparison_df = comparison_df.reindex(['r_squared', 'rmse', 'mae', 'n_observations'])
    
    # Save comparison of econometric models to corresponding csv file
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(save_path, index=True)
    logger.info("Saved regression comparison to %s", save_path)
    
    return comparison_df


def compare_classification_models(results_dict, save_path=None):
    """Compare the different classification models and save to csv"""
    comparison_data = {}
    
    for model_name, results in results_dict.items():
        metrics = compute_classification_metrics(results['y_test'], results['y_pred'], results.get('y_pred_proba'))
        comparison_data[model_name] = metrics
    
    # Create comparison dataframe with metricss as rows and models as columns
    comparison_df = pd.DataFrame(comparison_data)
    
    # Save comparison of classification models to corresponding csv file
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(save_path, index=True)
    logger.info("Saved model comparison to %s", save_path)
    
    return comparison_df

# =============================================================================
# Evaluation Pipeline Functions
# =============================================================================

def evaluate_econometric_models(econ_results, results_dir):
    """Evaluate econometric models and print metrics to terminal."""
    
    logger.info("Evaluating econometric models")
    
    print("\n" + "="*70)
    print("ECONOMETRIC MODEL EVALUATION")
    print("="*70)

    # Pooled linear regression
    pooled_metrics = compute_regression_metrics(econ_results['pooled_linear']['y_true'], econ_results['pooled_linear']['y_pred'])
    print_regression_metrics("Pooled Linear Regression", pooled_metrics)

    # Pooled logistic regression
    logistic_metrics = compute_classification_metrics(econ_results['pooled_logistic']['y_true'], econ_results['pooled_logistic']['y_pred'], econ_results['pooled_logistic']['y_pred_proba'])
    print_classification_metrics("Pooled Logistic Regression", logistic_metrics)

    # Logistic regression with interactions
    interaction_metrics = compute_classification_metrics(econ_results['interaction_logistic']['y_true'], econ_results['interaction_logistic']['y_pred'], econ_results['interaction_logistic']['y_pred_proba'])
    print_classification_metrics("Logistic with Interactions", interaction_metrics)
    
    print("\n" + "="*70)
    logger.info("Econometric evaluation complete. Saved to %s", results_dir)

def evaluate_ml_models(ml_results, results_dir):
    """Evaluate Machine Learning models and print metrics to terminal."""
    results_dir = Path(results_dir)
    logger.info("Evaluating ML models")

    print("\n" + "="*70)
    print("ML MODEL EVALUATION")
    print("="*70)
    
    # Evaluate the Random Forest model
    rf_metrics = compute_classification_metrics(ml_results['random_forest']['y_test'], ml_results['random_forest']['y_pred'], ml_results['random_forest']['y_pred_proba'])
    print_classification_metrics("Random Forest", rf_metrics, ml_results['random_forest']['y_test'], ml_results['random_forest']['y_pred'])
    
    # Evaluate XGBoost results
    print("\n" + "-"*70)
    xgb_metrics = compute_classification_metrics(ml_results['xgboost']['y_test'], ml_results['xgboost']['y_pred'], ml_results['xgboost']['y_pred_proba'])
    print_classification_metrics("XGBoost", xgb_metrics, ml_results['xgboost']['y_test'], ml_results['xgboost']['y_pred'])
    
    # Combined feature importance (saved top 10 to csv, print only top 5)
    print("\n" + "-"*70)
    print("\nFeature Importance Analysis")
    create_combined_feature_importance(ml_results['random_forest']['importance'], ml_results['xgboost']['importance'], ml_results['shap']['shap_importance'], save_path=results_dir / "combined_feature_importance.csv")
    print("\n(Top 10 saved to combined_feature_importance.csv)")
    
    # Model comparison
    print("\n" + "="*70)
    print("MODEL COMPARISON")
    print("="*70 + "\n")
    
    comparison_df = compare_classification_models({'Random Forest': ml_results['random_forest'], 'XGBoost': ml_results['xgboost']}, save_path=results_dir / "ml_model_comparison.csv")
    
    print()
    for metric_name in comparison_df.index:
        print(f"  {metric_name:<15} RF: {comparison_df.loc[metric_name, 'Random Forest']:.4f}    XGB: {comparison_df.loc[metric_name, 'XGBoost']:.4f}")

    print("\n" + "="*70)
    
    logger.info("ML evaluation complete")

