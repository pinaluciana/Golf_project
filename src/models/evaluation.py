"""
Evaluation and visualization functions for the golf analytics project.

This module contains all plotting functions for:
- Exploratory analysis (correlation heatmap, overall performance, top 25% comparison)
- Model evaluation (to be added: coefficient plots, ROC curves, etc.)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Exploratory Visualizations
# =============================================================================

def plot_correlation_heatmap(corr_matrix, title='Correlation Between Performance Metrics', 
                              save_path=None):
    """
    Create a heatmap showing correlations between performance metrics.
    
    Args:
        corr_matrix: Correlation matrix (from compute_winner_correlations or compute_all_players_correlations)
        title: Plot title
        save_path: Path to save the figure (optional)
    """
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr_matrix, 
        annot=True, 
        cmap='coolwarm', 
        center=0,
        fmt='.2f', 
        square=True, 
        linewidths=1, 
        cbar_kws={"shrink": 0.8}
    )
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved correlation heatmap to {save_path}")
    
    plt.show()


def plot_overall_performance(overall_perf_df, title='Overall Performance Metrics by Major',
                              save_path=None):
    """
    Create a grouped bar chart showing mean performance metrics by major for ALL players.
    This visualizes the context for pooled linear regression analysis.
    
    Args:
        overall_perf_df: DataFrame with majors as rows, metrics as columns (from compute_standardized_performance)
        title: Plot title
        save_path: Path to save the figure (optional)
    """
    # Transpose so metrics are on x-axis, majors are the groups
    plot_data = overall_perf_df.T
    
    # Create grouped bar chart
    ax = plot_data.plot(kind='bar', figsize=(16, 8), width=0.8)
    
    plt.title(title, fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Performance Metric', fontsize=13)
    plt.ylabel('Standardized Value (z-score)', fontsize=13)
    
    # Place legend outside plot area so it doesnt cover the bars
    plt.legend(title='Major', bbox_to_anchor=(1.0, 1.0), loc='upper left', fontsize=11)
    
    # Add horizontal line at 0 for reference
    plt.axhline(y=0, color='black', linestyle='--', linewidth=1.2)
    
    # Add horizontal gridlines for easier reading
    plt.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=0, ha='center', fontsize=10)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved overall performance plot to {save_path}")
    
    plt.show()


def plot_top25_comparison(comparison_data, save_path=None):
    """
    Create a bar chart comparing top 25% finishers vs rest of field.
    Shows standardized difference to compare metrics on different scales.
    
    Args:
        comparison_data: Dict from compare_top25_vs_rest() or Series from compute_standardized_top25_difference()
        save_path: Path to save the figure (optional)
    """
    # Handle both dict and Series input
    if isinstance(comparison_data, dict):
        # If dict, we need to compute standardized difference or use raw difference
        if 'difference' in comparison_data:
            diff_data = comparison_data['difference']
        else:
            diff_data = comparison_data
    else:
        # Assume it's already a Series (standardized difference)
        diff_data = comparison_data
    
    # Sort by absolute value to show most impactful metrics first
    diff_sorted = diff_data.reindex(diff_data.abs().sort_values(ascending=True).index)
    
    # Create horizontal bar chart
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Color bars based on direction (positive = top 25% higher, negative = top 25% lower)
    colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in diff_sorted.values]
    
    bars = ax.barh(diff_sorted.index, diff_sorted.values, color=colors)
    
    plt.title('Top 25% vs Rest of Field: Standardized Metric Differences', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Standardized Difference (positive = top 25% higher)', fontsize=12)
    plt.ylabel('Performance Metric', fontsize=12)
    
    # Add vertical line at 0
    plt.axvline(x=0, color='black', linestyle='-', linewidth=1)
    
    # Add value labels on bars
    for bar, val in zip(bars, diff_sorted.values):
        x_pos = val + 0.02 if val >= 0 else val - 0.02
        ha = 'left' if val >= 0 else 'right'
        ax.text(x_pos, bar.get_y() + bar.get_height()/2, f'{val:.2f}', 
                va='center', ha=ha, fontsize=10)
    
    # Add legend explaining colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label='Top 25% higher'),
        Patch(facecolor='#e74c3c', label='Top 25% lower (better for some metrics)')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved top 25% comparison plot to {save_path}")
    
    plt.show()


def plot_exploratory_results(results, df=None, results_dir=None):
    """
    Generate all exploratory visualizations and save to results folder.
    
    Args:
        results: Dict from run_exploratory_analysis()
        df: Original DataFrame (optional, for additional plots)
        results_dir: Directory to save plots (default: results/1_Exploratory/)
    """
    # Setup results directory
    if results_dir is None:
        results_dir = Path(__file__).parent.parent.parent / "results" / "1_Exploratory" / "figures"
    else:
        results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving exploratory plots to {results_dir}")
    
    # 1. Correlation Heatmap (using winner correlations)
    plot_correlation_heatmap(
        results['winner_correlations'],
        title='Correlation Between Winner Performance Metrics',
        save_path=results_dir / "correlation_heatmap.png"
    )
    
    # 2. Overall Performance by Major (standardized, for all players)
    plot_overall_performance(
        results['overall_performance_std'],
        title='Overall Performance Metrics by Major (Standardized)',
        save_path=results_dir / "overall_performance_by_major.png"
    )
    
    # 3. Top 25% vs Rest Comparison (standardized difference)
    plot_top25_comparison(
        results['top25_std_difference'],
        save_path=results_dir / "top25_vs_rest_comparison.png"
    )
    
    logger.info("All exploratory plots saved")


# =============================================================================
# Model Evaluation Functions (to be expanded for regression models)
# =============================================================================

def plot_coefficients(coefficients, title='Feature Coefficients', save_path=None):
    """
    Create a bar chart showing model coefficients.
    
    Args:
        coefficients: Series or dict with feature names and coefficient values
        title: Plot title
        save_path: Path to save the figure (optional)
    """
    if isinstance(coefficients, dict):
        coefficients = pd.Series(coefficients)
    
    # Sort by absolute value
    coef_sorted = coefficients.reindex(coefficients.abs().sort_values(ascending=True).index)
    
    # Create horizontal bar chart
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Color by sign
    colors = ['#3498db' if x >= 0 else '#e74c3c' for x in coef_sorted.values]
    
    ax.barh(coef_sorted.index, coef_sorted.values, color=colors)
    
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Coefficient Value', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.axvline(x=0, color='black', linestyle='-', linewidth=1)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved coefficients plot to {save_path}")
    
    plt.show()


def plot_residuals(y_true, y_pred, title='Residual Plot', save_path=None):
    """
    Create a residual plot to check model fit.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        title: Plot title
        save_path: Path to save the figure (optional)
    """
    residuals = y_true - y_pred
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_pred, residuals, alpha=0.5)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=2)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Values', fontsize=12)
    plt.ylabel('Residuals', fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved residual plot to {save_path}")
    
    plt.show()


def plot_actual_vs_predicted(y_true, y_pred, title='Actual vs Predicted', save_path=None):
    """
    Create a scatter plot of actual vs predicted values.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        title: Plot title
        save_path: Path to save the figure (optional)
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_true, y_pred, alpha=0.5)
    
    # Add perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Actual Values', fontsize=12)
    plt.ylabel('Predicted Values', fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved actual vs predicted plot to {save_path}")
    
    plt.show()


# =============================================================================
# Script Entry Point (for testing)
# =============================================================================

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    import sys
    import os
    # Get the directory containing this script (src/models)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Get the parent directory (src)
    src_dir = os.path.dirname(current_dir)
    # Add src to the system path so Python can find data_loader
    if src_dir not in sys.path:
        sys.path.append(src_dir)

    # Import and run exploratory analysis, then plot
    from data_loader import load_combined_data
    from exploratory import run_exploratory_analysis
    
    # Load data
    df = load_combined_data()
    
    # Run exploratory analysis
    results = run_exploratory_analysis(df)
    
    # Generate and save all exploratory plots
    plot_exploratory_results(results, df)