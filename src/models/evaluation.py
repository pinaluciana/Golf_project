"""
Evaluation metrics and model comparison for the golf performance models.
This file computes metrics for both econometric and Machine Learning models.
"""

import logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (mean_squared_error, r2_score, accuracy_score,  roc_auc_score, confusion_matrix, classification_report,  precision_score, recall_score, f1_score)

logger = logging.getLogger(__name__)

# =============================================================================
# BASIC METRIC COMPUTATION FUNCTIONS
# =============================================================================

def compute_regression_metrics(y_true, y_pred):
    """Compute R², RMSE, MAE for linear regression models."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return {'r_squared': r2_score(y_true, y_pred), 'rmse': np.sqrt(mean_squared_error(y_true, y_pred)), 'mae': np.mean(np.abs(y_true - y_pred)),'n_observations': len(y_true)}

def compute_classification_metrics(y_true, y_pred, y_pred_proba=None):
    """Compute all classification metrics: accuracy, precision, recall, F1, ROC-AUC."""
    metrics = {'accuracy': accuracy_score(y_true, y_pred), 'precision': precision_score(y_true, y_pred, zero_division=0),'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1_score': f1_score(y_true, y_pred, zero_division=0)}
    if y_pred_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
    return metrics

def compute_confusion_matrix(y_true, y_pred):
    """Compute confusion matrix."""
    return confusion_matrix(y_true, y_pred)

# =============================================================================
# ECONOMETRIC MODEL EVALUATION
# =============================================================================

def evaluate_econometric_models(econ_results, data):
    """
    Evaluate econometric models and print metrics organized by subsections.
    Prints evaluation metrics for Section 3.
    """
    
    # Section 3.1: Pooled Linear Regression
    print("-"*100)
    print("SECTION 3.1: Pooled Linear Regression")
    print("-"*100)

    pooled_metrics = compute_regression_metrics(
        econ_results['pooled_linear']['y_true'], 
        econ_results['pooled_linear']['y_pred'])

    logger.info("Pooled Linear Regression:")
    logger.info(f"  R²:          {pooled_metrics['r_squared']:.4f}")
    logger.info(f"  Adjusted R²: {econ_results['pooled_linear']['model'].rsquared_adj:.4f}")
    logger.info(f"  RMSE:        {pooled_metrics['rmse']:.4f} strokes")
    logger.info(f"  N:           {pooled_metrics['n_observations']}")
    
    # Section 3.2: Per-Major Linear Regression
    print("\n" + "-"*100)
    print("SECTION 3.2: Per-Major Linear Regression")
    print("-"*100)
    logger.info("Per-Major Linear Regression:")

    for major in data['major'].unique():
        major_model = econ_results['per_major_linear'][major]['model']
        major_metrics = compute_regression_metrics(
            econ_results['per_major_linear'][major]['y_true'],
            econ_results['per_major_linear'][major]['y_pred'])
    
        logger.info(f"  {major:<25} R² = {major_model.rsquared:.4f}, Adj R² = {major_model.rsquared_adj:.4f}, RMSE = {major_metrics['rmse']:.4f} strokes, n = {major_metrics['n_observations']}")
    
   # Section 3.3: Pooled Logistic Regression
    print("\n" + "-"*100)
    print("SECTION 3.3: Pooled Logistic Regression")
    print("-"*100)

    pooled_metrics = compute_classification_metrics(
        econ_results['pooled_logistic']['y_true'],
        econ_results['pooled_logistic']['y_pred'],
        econ_results['pooled_logistic']['y_pred_proba'])

    logger.info("Pooled Logistic Regression:")
    logger.info(f"  Accuracy:  {pooled_metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {pooled_metrics['precision']:.4f}")
    logger.info(f"  Recall:    {pooled_metrics['recall']:.4f}")
    logger.info(f"  F1 Score:  {pooled_metrics['f1_score']:.4f}")
    logger.info(f"  ROC-AUC:   {pooled_metrics['roc_auc']:.4f}")
    
   # Section 3.4: Extension with Interactions
    print("\n" + "-"*100)
    print("SECTION 3.4: Extension of Pooled Logistic Regression with Interactions")
    print("-"*100)

    inter_metrics = compute_classification_metrics(
        econ_results['interaction_logistic']['y_true'],
        econ_results['interaction_logistic']['y_pred'],
        econ_results['interaction_logistic']['y_pred_proba'])

    logger.info("Logistic Regression with Interactions:")
    logger.info(f"  Accuracy:  {inter_metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {inter_metrics['precision']:.4f}")
    logger.info(f"  Recall:    {inter_metrics['recall']:.4f}")
    logger.info(f"  F1 Score:  {inter_metrics['f1_score']:.4f}")
    logger.info(f"  ROC-AUC:   {inter_metrics['roc_auc']:.4f}")
    
# =============================================================================
# ECONOMETRIC MODEL DETAILED SUMMARY
# =============================================================================

def print_econometric_summary(econ_results, data, results_dir_econ):
    """
    Print detailed summary of econometric analysis results organized by subsections.
    Shows top features and coefficients for each model and the files that were created.
    """
    
    # Section 4.1: Pooled Linear Regression Results
    print("-"*100)
    print("SECTION 4.1: Pooled Linear Regression Results (explain score based on performance metrics)")
    print("-"*100)
    logger.info(f"R² = {econ_results['pooled_linear']['model'].rsquared:.4f}")
    logger.info(f"Adjusted R² = {econ_results['pooled_linear']['model'].rsquared_adj:.4f}")
    logger.info("Top 5 Features (by absolute coefficient):")
    top_coefs = econ_results['pooled_linear']['coefficients'][econ_results['pooled_linear']['coefficients']['Feature'] != 'const'].copy()
    top_coefs = top_coefs.reindex(top_coefs['Coefficient'].abs().sort_values(ascending=False).head(5).index)
    for _, row in top_coefs.iterrows():
        logger.info(f"  {row['Feature']:<15} {row['Coefficient']:>7.3f}  (p={row['p_value']:.3f})")
    
    # Section 4.2: Per-Major Linear Regression Results
    print("\n" + "-"*100)
    print("SECTION 4.2: Per-Major Linear Regression Results (show how skill importance varies across tournaments)")
    print("-"*100)
    for major in data['major'].unique():
        r2 = econ_results['per_major_linear'][major]['model'].rsquared
        logger.info(f"{major}: R² = {r2:.4f}")
    
        # Get top 3 features for this major
        major_coefs = econ_results['per_major_linear'][major]['coefficients_df']
        major_coefs = major_coefs[major_coefs['Feature'] != 'const'].copy()
        top_3 = major_coefs.reindex(major_coefs['Coefficient'].abs().sort_values(ascending=False).head(3).index)
        logger.info("  Top 3 Features:")
        for _, row in top_3.iterrows():
            logger.info(f"    {row['Feature']:<15} {row['Coefficient']:>7.3f}  (p={row['p_value']:.3f})")
        print()  # Blank line between majors
    
    # Section 4.3: Pooled Logistic Regression Results
    print("-"*100)
    print("SECTION 4.3: Pooled Logistic Regression Results (probability of finishing in the top 25% of the leaderboard )")
    print("-"*100)
    logistic_acc = accuracy_score(econ_results['pooled_logistic']['y_true'], econ_results['pooled_logistic']['y_pred'])
    logger.info(f"Accuracy = {logistic_acc:.4f}")
    logger.info("Top 5 Predictors of Top 25%:")
    top_logistic = econ_results['pooled_logistic']['coefficients'][econ_results['pooled_logistic']['coefficients']['Feature'] != 'const'].copy()
    top_logistic = top_logistic.reindex(top_logistic['Coefficient'].abs().sort_values(ascending=False).head(5).index)
    for _, row in top_logistic.iterrows():
        logger.info(f"  {row['Feature']:<15} {row['Coefficient']:>7.3f}  (p={row['p_value']:.3f})")
    
    # Section 4.4: Logistic Regression with Interactions Results
    print("\n" + "-"*100)
    print("SECTION 4.4: Logistic Regression with Interactions Results (probability of finishing in the top 25% of the leaderboard in each Major)")
    print("-"*100)
    inter_acc = accuracy_score(econ_results['interaction_logistic']['y_true'], econ_results['interaction_logistic']['y_pred'])
    logger.info(f"Accuracy = {inter_acc:.4f}")
    logger.info("Top 3 Features by Major:")
    
    comparison_table = econ_results['interaction_logistic']['comparison_table']
    for major in comparison_table.columns:
        logger.info(f"  {major}:")
        # Get top 3 for this major by absolute coefficient
        major_coefs = comparison_table[major].abs().sort_values(ascending=False).head(3)
        for feature in major_coefs.index:
            coef_value = comparison_table.loc[feature, major]
            logger.info(f"    {feature:<15} {coef_value:>7.3f}")
    
     # Files Created section
    print("\n" + "-"*100)
    logger.info("FILES CREATED:")
    
    # List CSV files
    csv_files = sorted(results_dir_econ.glob('*.csv'))
    for csv_file in csv_files:
        logger.info(f"  Saved {csv_file.name}")
    
    # List PNG files in figures directory
    figures_dir = results_dir_econ / "figures"
    png_files = sorted(figures_dir.glob('*.png'))
    for png_file in png_files:
        logger.info(f"  Saved {png_file.name}")
    print("-"*100)

# =============================================================================
# MACHINE LEARNING MODEL EVALUATION
# =============================================================================

def evaluate_ml_models(ml_results, results_dir):
    """
    Evaluate ML models and print metrics organized by subsections.
    Prints evaluation metrics for Section 5.
    """
    results_dir = Path(results_dir)
    
    # Section 5.1: Random Forest
    print("-"*100)
    print("SECTION 5.1: Random Forest")
    print("-"*100)
    
    # Training metrics
    rf_train_pred = ml_results['random_forest']['model'].predict(ml_results['random_forest']['X_train'])
    rf_train_proba = ml_results['random_forest']['model'].predict_proba(ml_results['random_forest']['X_train'])[:, 1]
    rf_train_metrics = compute_classification_metrics(
        ml_results['random_forest']['y_train'],
        rf_train_pred,
        rf_train_proba)

    # Test metrics
    rf_test_metrics = compute_classification_metrics(
        ml_results['random_forest']['y_test'], 
        ml_results['random_forest']['y_pred'], 
        ml_results['random_forest']['y_pred_proba'])

    logger.info("Random Forest:")
    logger.info(f"  Training Accuracy:  {rf_train_metrics['accuracy']:.4f}")
    logger.info(f"  Test Accuracy:      {rf_test_metrics['accuracy']:.4f}")
    logger.info(f"  Overfitting Gap:    {rf_train_metrics['accuracy'] - rf_test_metrics['accuracy']:.4f}")
    print()  # Blank line
    logger.info(f"  Precision: {rf_test_metrics['precision']:.4f}")
    logger.info(f"  Recall:    {rf_test_metrics['recall']:.4f}")
    logger.info(f"  F1 Score:  {rf_test_metrics['f1_score']:.4f}")
    logger.info(f"  ROC-AUC:   {rf_test_metrics['roc_auc']:.4f}")
    
    # Random Forest confusion matrix
    print()  # Blank line
    logger.info("Confusion Matrix:")
    cm_rf = compute_confusion_matrix(ml_results['random_forest']['y_test'], ml_results['random_forest']['y_pred'])
    total_rf = cm_rf.sum()
    logger.info("                       Predicted")
    logger.info("                    Rest      Top 25%")
    logger.info("  Actual")
    logger.info(f"  Rest              {cm_rf[0,0]:>6}     {cm_rf[0,1]:>6}    (True Negative: {cm_rf[0,0]}, False Positive: {cm_rf[0,1]})")
    logger.info(f"  Top 25%           {cm_rf[1,0]:>6}     {cm_rf[1,1]:>6}    (False Negative: {cm_rf[1,0]}, True Positive: {cm_rf[1,1]})")
    print()  # Blank line
    logger.info(f"Correctly Predicted: {cm_rf[0,0] + cm_rf[1,1]} out of {total_rf} ({(cm_rf[0,0] + cm_rf[1,1])/total_rf*100:.1f}%)")
    
    # Section 5.2: XGBoost
    print("\n" + "-"*100)
    print("SECTION 5.2: XGBoost")
    print("-"*100)
    
    # Training metrics
    xgb_train_pred = ml_results['xgboost']['model'].predict(ml_results['xgboost']['X_train'])
    xgb_train_proba = ml_results['xgboost']['model'].predict_proba(ml_results['xgboost']['X_train'])[:, 1]
    xgb_train_metrics = compute_classification_metrics(
        ml_results['xgboost']['y_train'],
        xgb_train_pred,
        xgb_train_proba)

    # Test metrics
    xgb_test_metrics = compute_classification_metrics(
        ml_results['xgboost']['y_test'], 
        ml_results['xgboost']['y_pred'], 
        ml_results['xgboost']['y_pred_proba'])

    logger.info("XGBoost:")
    logger.info(f"  Training Accuracy:  {xgb_train_metrics['accuracy']:.4f}")
    logger.info(f"  Test Accuracy:      {xgb_test_metrics['accuracy']:.4f}")
    logger.info(f"  Overfitting Gap:    {xgb_train_metrics['accuracy'] - xgb_test_metrics['accuracy']:.4f}")
    print()  # Blank line
    logger.info(f"  Precision: {xgb_test_metrics['precision']:.4f}")
    logger.info(f"  Recall:    {xgb_test_metrics['recall']:.4f}")
    logger.info(f"  F1 Score:  {xgb_test_metrics['f1_score']:.4f}")
    logger.info(f"  ROC-AUC:   {xgb_test_metrics['roc_auc']:.4f}")

    # XGBoost confusion matrix
    print()  # Blank line
    logger.info("Confusion Matrix:")
    cm_xgb = compute_confusion_matrix(ml_results['xgboost']['y_test'], ml_results['xgboost']['y_pred'])
    total_xgb = cm_xgb.sum()
    logger.info("                       Predicted")
    logger.info("                    Rest      Top 25%")
    logger.info("  Actual")
    logger.info(f"  Rest              {cm_xgb[0,0]:>6}     {cm_xgb[0,1]:>6}    (True Negative: {cm_xgb[0,0]}, False Positive: {cm_xgb[0,1]})")
    logger.info(f"  Top 25%           {cm_xgb[1,0]:>6}     {cm_xgb[1,1]:>6}    (False Negative: {cm_xgb[1,0]}, True Positive: {cm_xgb[1,1]})")
    print()  # Blank line
    logger.info(f"Correctly Predicted: {cm_xgb[0,0] + cm_xgb[1,1]} out of {total_xgb} ({(cm_xgb[0,0] + cm_xgb[1,1])/total_xgb*100:.1f}%)")

# =============================================================================
# MACHINE LEARNING MODEL DETAILED SUMMARY
# =============================================================================
def print_ml_summary(ml_results, results_dir):
    """
    Print detailed ML analysis summary organized by subsections.
    Shows model comparison, SHAP analysis, and feature importance.
    """
    results_dir = Path(results_dir)
    
    # Section 6.1: Model Comparison
    print("-"*100)
    print("SECTION 6.1: Model Comparison")
    print("-"*100)
    
    # Get metrics for both models
    rf_metrics = compute_classification_metrics(ml_results['random_forest']['y_test'], 
                                               ml_results['random_forest']['y_pred'], 
                                               ml_results['random_forest']['y_pred_proba'])
    xgb_metrics = compute_classification_metrics(ml_results['xgboost']['y_test'], 
                                                ml_results['xgboost']['y_pred'], 
                                                ml_results['xgboost']['y_pred_proba'])
    
    logger.info("Model Comparison:")
    logger.info(f"{'Metric':<15} {'Random Forest':>15} {'XGBoost':>15} {'Winner':>15}")
    for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']:
        rf_val = rf_metrics[metric]
        xgb_val = xgb_metrics[metric]
        winner = "XGBoost" if xgb_val > rf_val else "Random Forest" if rf_val > xgb_val else "Tie"
        logger.info(f"{metric:<15} {rf_val:>15.4f} {xgb_val:>15.4f} {winner:>15}")
    
    baseline_acc = 1 - ml_results['xgboost']['y_test'].mean()
    logger.info(f"Baseline (predict 'Rest' always): {baseline_acc:.4f}")
    logger.info(f"XGBoost improvement: +{xgb_metrics['accuracy'] - baseline_acc:.4f}")
    
    # Section 6.2: SHAP Analysis
    print("\n" + "-"*100)
    print("SECTION 6.2: SHAP Analysis (from XGBoost Model)")
    print("-"*100)
    
    logger.info("Global SHAP Importance (Top 5):")
    shap_top = ml_results['shap']['shap_importance'][~ml_results['shap']['shap_importance']['Feature'].str.startswith('major_')].head(5)
    for _, row in shap_top.iterrows():
        logger.info(f"  {row['Feature']:<15} {row['SHAP_Importance']:>8.4f}")
    
    print()  # Blank line
    logger.info("SHAP Importance by Major (Top 3 per tournament):")
    for major, major_results in ml_results['shap_per_major'].items():
        logger.info(f"  {major}:")
        for _, row in major_results['shap_importance'].head(3).iterrows():
            logger.info(f"    {row['Feature']:<15} {row['SHAP_Importance']:>8.4f}")
    
    # Section 6.3: Feature Importance
    print("\n" + "-"*100)
    print("SECTION 6.3: Feature Importance of Models and SHAP")
    print("-"*100)
    
    # Get top 5 for each, excluding major dummies
    rf_top = ml_results['random_forest']['importance'][~ml_results['random_forest']['importance']['Feature'].str.startswith('major_')].head(5)
    xgb_top = ml_results['xgboost']['importance'][~ml_results['xgboost']['importance']['Feature'].str.startswith('major_')].head(5)
    shap_top = ml_results['shap']['shap_importance'][~ml_results['shap']['shap_importance']['Feature'].str.startswith('major_')].head(5)
    
    logger.info("Random Forest:")
    for _, row in rf_top.iterrows():
        logger.info(f"  {row['Feature']:<15} {row['Importance']:>8.4f}")
    
    print()  # Blank line
    logger.info("XGBoost:")
    for _, row in xgb_top.iterrows():
        logger.info(f"  {row['Feature']:<15} {row['Importance']:>8.4f}")
    
    print()  # Blank line
    logger.info("SHAP:")
    for _, row in shap_top.iterrows():
        logger.info(f"  {row['Feature']:<15} {row['SHAP_Importance']:>8.4f}")
    
    # Files Created section
    print("\n" + "-"*100)
    logger.info("FILES CREATED:")
    
    # List CSV files
    csv_files = sorted(results_dir.glob('*.csv'))
    for csv_file in csv_files:
        logger.info(f"  Saved {csv_file.name}")
    
    # List PNG files in figures directory
    figures_dir = results_dir / "figures"
    png_files = sorted(figures_dir.glob('*.png'))
    for png_file in png_files:
        logger.info(f"  Saved {png_file.name}")
    print("-"*100)