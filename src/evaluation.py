"""
Evaluation and visualization functions for the golf analytics project.

This module contains all plotting functions for:
- Exploratory analysis (correlation heatmap, overall performance, top 25% comparison)
- Model evaluation (coefficient plots, residual plots, etc.)
"""

import sys
import logging
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving figures
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position
import seaborn as sns  # pylint: disable=wrong-import-position
from matplotlib.patches import Patch  # pylint: disable=wrong-import-position

# Add parent directory to path to allow imports from src/
sys.path.insert(0, str(Path(__file__).parent))

logger = logging.getLogger(__name__)


# =============================================================================
# Exploratory Visualizations
# =============================================================================

def plot_correlation_heatmap(corr_matrix, title='Correlation Between Performance Metrics',
                              save_path=None):
    """
    Create a heatmap showing correlations between performance metrics.

    Args:
        corr_matrix: Correlation matrix (pandas DataFrame or numpy array)
        title: Plot title
        save_path: Path to save the figure (optional)

    Raises:
        ValueError: If corr_matrix is empty or invalid
        IOError: If unable to save the plot
    """
    if corr_matrix is None or corr_matrix.empty:
        raise ValueError("Correlation matrix is empty or None")

    try:
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
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info("Saved correlation heatmap to %s", save_path)

        plt.close()

    except (OSError, ValueError) as err:
        logger.error("Error creating correlation heatmap: %s", err)
        plt.close()
        raise


def plot_overall_performance(overall_perf_df,
                              title='Overall Performance Metrics by Major',
                              save_path=None):
    """
    Create a grouped bar chart showing mean performance metrics by major.

    This visualizes the context for pooled linear regression analysis.

    Args:
        overall_perf_df: DataFrame with majors as rows, metrics as columns
        title: Plot title
        save_path: Path to save the figure (optional)

    Raises:
        ValueError: If DataFrame is empty or invalid
        IOError: If unable to save the plot
    """
    if overall_perf_df is None or overall_perf_df.empty:
        raise ValueError("Performance DataFrame is empty or None")

    try:
        # Transpose so metrics are on x-axis, majors are the groups
        plot_data = overall_perf_df.T

        # Create grouped bar chart
        plot_data.plot(kind='bar', figsize=(16, 8), width=0.8)

        plt.title(title, fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Performance Metric', fontsize=13)
        plt.ylabel('Standardized Value (z-score)', fontsize=13)

        # Place legend outside plot area
        plt.legend(
            title='Major',
            bbox_to_anchor=(1.0, 1.0),
            loc='upper left',
            fontsize=11
        )

        # Add horizontal line at 0 for reference
        plt.axhline(y=0, color='black', linestyle='--', linewidth=1)

        # Add horizontal gridlines for easier reading
        plt.grid(axis='y', alpha=0.3)
        plt.xticks(rotation=0, ha='center', fontsize=10)
        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info("Saved overall performance plot to %s", save_path)

        plt.close()

    except (OSError, ValueError) as err:
        logger.error("Error creating overall performance plot: %s", err)
        plt.close()
        raise


def plot_overall_performance_heatmap(overall_perf_df,
                                      title='Performance Metrics by Major',
                                      save_path=None):
    """
    Create a heatmap showing mean performance metrics by major.

    This is an alternative to the bar chart - displays z-scores in a grid
    format with color coding (green = above average, red = below average).

    Args:
        overall_perf_df: DataFrame with majors as rows, metrics as columns
        title: Plot title
        save_path: Path to save the figure (optional)

    Raises:
        ValueError: If DataFrame is empty or invalid
        IOError: If unable to save the plot
    """
    if overall_perf_df is None or overall_perf_df.empty:
        raise ValueError("Performance DataFrame is empty or None")

    try:
        plt.figure(figsize=(14, 5))
        sns.heatmap(
            overall_perf_df,
            annot=True,
            cmap='RdYlGn',
            center=0,
            fmt='.2f',
            linewidths=0.5,
            cbar_kws={'label': 'Z-Score'}
        )
        plt.title(title, fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Performance Metric', fontsize=12)
        plt.ylabel('Major', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info("Saved overall performance heatmap to %s", save_path)

        plt.close()

    except (OSError, ValueError) as err:
        logger.error("Error creating overall performance heatmap: %s", err)
        plt.close()
        raise


def plot_top25_comparison(comparison_data, save_path=None):
    """
    Create a bar chart comparing top 25% finishers vs rest of field.

    Shows standardized difference to compare metrics on different scales.

    Args:
        comparison_data: Dict from compare_top25_vs_rest() or Series
        save_path: Path to save the figure (optional)

    Raises:
        ValueError: If comparison_data is empty or invalid
        IOError: If unable to save the plot
    """
    # Handle both dict and Series input
    if isinstance(comparison_data, dict):
        if 'difference' in comparison_data:
            diff_data = comparison_data['difference']
        else:
            diff_data = comparison_data
    else:
        diff_data = comparison_data

    if diff_data is None or len(diff_data) == 0:
        raise ValueError("Comparison data is empty or None")

    try:
        # Sort by absolute value to show most impactful metrics first
        diff_sorted = diff_data.reindex(
            diff_data.abs().sort_values(ascending=True).index
        )

        # Create horizontal bar chart
        _, ax = plt.subplots(figsize=(10, 8))

        # Color bars based on direction
        colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in diff_sorted.values]

        bars = ax.barh(diff_sorted.index, diff_sorted.values, color=colors)

        plt.title(
            'Top 25% vs Rest of Field: Standardized Metric Differences',
            fontsize=14,
            fontweight='bold',
            pad=20
        )
        plt.xlabel(
            'Standardized Difference (positive = top 25% higher)',
            fontsize=12
        )
        plt.ylabel('Performance Metric', fontsize=12)

        # Add vertical line at 0
        plt.axvline(x=0, color='black', linestyle='-', linewidth=1)

        # Add value labels on bars
        for single_bar, val in zip(bars, diff_sorted.values):
            x_pos = val + 0.02 if val >= 0 else val - 0.02
            horiz_align = 'left' if val >= 0 else 'right'
            ax.text(
                x_pos,
                single_bar.get_y() + single_bar.get_height() / 2,
                f'{val:.2f}',
                va='center',
                ha=horiz_align,
                fontsize=10
            )

        # Add legend explaining colors
        legend_elements = [
            Patch(facecolor='#2ecc71', label='Top 25% higher'),
            Patch(facecolor='#e74c3c', label='Top 25% lower (better for some)')
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info("Saved top 25%% comparison plot to %s", save_path)

        plt.close()

    except (OSError, ValueError) as err:
        logger.error("Error creating top 25%% comparison plot: %s", err)
        plt.close()
        raise


def plot_exploratory_results(analysis_results, results_dir=None):
    """
    Generate all exploratory visualizations and save to results folder.

    Args:
        analysis_results: Dict from run_exploratory_analysis()
        results_dir: Directory to save plots (default: results/1_Exploratory/figures)

    Raises:
        ValueError: If results dict is missing required keys
        IOError: If unable to create results directory or save plots
    """
    # Validate results dict
    required_keys = [
        'winner_correlations',
        'overall_performance_std',
        'top25_std_difference'
    ]
    missing_keys = [key for key in required_keys if key not in analysis_results]
    if missing_keys:
        raise ValueError(f"Results dict missing required keys: {missing_keys}")

    # Setup results directory
    if results_dir is None:
        root = Path(__file__).parent.parent
        results_dir = root / "results" / "1_Exploratory" / "figures"
    else:
        results_dir = Path(results_dir)

    # Create directory if it doesn't exist
    try:
        results_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Results directory ready: %s", results_dir)
    except OSError as err:
        logger.error("Failed to create results directory %s: %s", results_dir, err)
        raise IOError(f"Cannot create results directory: {err}") from err

    # Track successes and failures
    plots_created = []
    plots_failed = []

    # 1. Correlation Heatmap
    try:
        plot_correlation_heatmap(
            analysis_results['winner_correlations'],
            title='Correlation Between Winner Performance Metrics',
            save_path=results_dir / "correlation_heatmap.png"
        )
        plots_created.append("correlation_heatmap.png")
    except (ValueError, OSError) as err:
        plots_failed.append(("correlation_heatmap.png", str(err)))
        logger.error("Failed to create correlation heatmap: %s", err)

    # 2. Overall Performance by Major (Bar Chart)
    try:
        plot_overall_performance(
            analysis_results['overall_performance_std'],
            title='Overall Performance Metrics by Major (Standardized)',
            save_path=results_dir / "overall_performance_by_major.png"
        )
        plots_created.append("overall_performance_by_major.png")
    except (ValueError, OSError) as err:
        plots_failed.append(("overall_performance_by_major.png", str(err)))
        logger.error("Failed to create overall performance plot: %s", err)

    # 2b. Overall Performance by Major (Heatmap - alternative view)
    try:
        plot_overall_performance_heatmap(
            analysis_results['overall_performance_std'],
            title='Performance Metrics by Major (Z-Scores)',
            save_path=results_dir / "overall_performance_heatmap.png"
        )
        plots_created.append("overall_performance_heatmap.png")
    except (ValueError, OSError) as err:
        plots_failed.append(("overall_performance_heatmap.png", str(err)))
        logger.error("Failed to create overall performance heatmap: %s", err)

    # 3. Top 25% vs Rest Comparison
    try:
        plot_top25_comparison(
            analysis_results['top25_std_difference'],
            save_path=results_dir / "top25_vs_rest_comparison.png"
        )
        plots_created.append("top25_vs_rest_comparison.png")
    except (ValueError, OSError) as err:
        plots_failed.append(("top25_vs_rest_comparison.png", str(err)))
        logger.error("Failed to create top 25%% comparison plot: %s", err)

    # Summary
    logger.info("Successfully created %d plots", len(plots_created))
    if plots_failed:
        logger.warning("Failed to create %d plots", len(plots_failed))
        for plot_name, error in plots_failed:
            logger.warning("  - %s: %s", plot_name, error)

    print(f"\n{'='*60}")
    print("PLOT GENERATION SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Successfully created: {len(plots_created)} plots")
    for plot in plots_created:
        print(f"   - {plot}")
    if plots_failed:
        print(f"\n❌ Failed: {len(plots_failed)} plots")
        for plot_name, error in plots_failed:
            print(f"   - {plot_name}: {error[:50]}...")
    print(f"\nOutput directory: {results_dir}")
    print(f"{'='*60}\n")


# =============================================================================
# Model Evaluation Functions
# =============================================================================

def plot_coefficients(coefficients, title='Feature Coefficients', save_path=None):
    """
    Create a bar chart showing model coefficients.

    Args:
        coefficients: Series or dict with feature names and coefficient values
        title: Plot title
        save_path: Path to save the figure (optional)

    Raises:
        ValueError: If coefficients is empty or invalid
    """
    if isinstance(coefficients, dict):
        coefficients = pd.Series(coefficients)

    if coefficients is None or len(coefficients) == 0:
        raise ValueError("Coefficients data is empty or None")

    try:
        # Sort by absolute value
        coef_sorted = coefficients.reindex(
            coefficients.abs().sort_values(ascending=True).index
        )

        # Create horizontal bar chart
        _, ax = plt.subplots(figsize=(10, 8))

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
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info("Saved coefficients plot to %s", save_path)

        plt.close()

    except (OSError, ValueError) as err:
        logger.error("Error creating coefficients plot: %s", err)
        plt.close()
        raise


def plot_residuals(y_true, y_pred, title='Residual Plot', save_path=None):
    """
    Create a residual plot to check model fit.

    Args:
        y_true: Actual values (array-like)
        y_pred: Predicted values (array-like)
        title: Plot title
        save_path: Path to save the figure (optional)

    Raises:
        ValueError: If inputs are invalid or mismatched
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if len(y_true) == 0 or len(y_pred) == 0:
        raise ValueError("Input arrays are empty")

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}"
        )

    try:
        residuals = y_true - y_pred

        _, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(y_pred, residuals, alpha=0.5)
        ax.axhline(y=0, color='red', linestyle='--', linewidth=2)

        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Values', fontsize=12)
        plt.ylabel('Residuals', fontsize=12)
        plt.grid(alpha=0.3)
        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info("Saved residual plot to %s", save_path)

        plt.close()

    except (OSError, ValueError) as err:
        logger.error("Error creating residual plot: %s", err)
        plt.close()
        raise


def plot_actual_vs_predicted(y_true, y_pred, title='Actual vs Predicted',
                              save_path=None):
    """
    Create a scatter plot of actual vs predicted values.

    Args:
        y_true: Actual values (array-like)
        y_pred: Predicted values (array-like)
        title: Plot title
        save_path: Path to save the figure (optional)

    Raises:
        ValueError: If inputs are invalid or mismatched
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if len(y_true) == 0 or len(y_pred) == 0:
        raise ValueError("Input arrays are empty")

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}"
        )

    try:
        _, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(y_true, y_pred, alpha=0.5)

        # Add perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot(
            [min_val, max_val],
            [min_val, max_val],
            'r--',
            linewidth=2,
            label='Perfect Prediction'
        )

        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel('Actual Values', fontsize=12)
        plt.ylabel('Predicted Values', fontsize=12)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info("Saved actual vs predicted plot to %s", save_path)

        plt.close()

    except (OSError, ValueError) as err:
        logger.error("Error creating actual vs predicted plot: %s", err)
        plt.close()
        raise


# =============================================================================
# Script Entry Point (for testing)
# =============================================================================

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    try:
        # Import required modules
        from data_loader import load_combined_data
        from exploratory import run_exploratory_analysis

        logger.info("Starting exploratory analysis pipeline")

        # Load data
        logger.info("Loading data...")
        data_df = load_combined_data()
        logger.info("Data loaded: %d records", len(data_df))

        # Run exploratory analysis
        logger.info("Running exploratory analysis...")
        exploratory_results = run_exploratory_analysis(data_df)
        logger.info("Exploratory analysis complete")

        # Generate and save all exploratory plots
        logger.info("Generating plots...")
        plot_exploratory_results(exploratory_results)
        logger.info("All tasks complete!")

    except ImportError as err:
        logger.error("Import error: %s", err)
        logger.error(
            "Make sure data_loader.py and exploratory.py are in the same directory"
        )
        sys.exit(1)
    except (ValueError, OSError) as err:
        logger.error("Unexpected error: %s", err)
        import traceback
        traceback.print_exc()
        sys.exit(1)

