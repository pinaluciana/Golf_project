"""
Exploratory Analysis of Golf Performance Variables in Major Championships.
This module computes descriptive statistics and generates exploratory visualizations.
Model-related plots are in visualization.py.
"""

import logging
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # We use this non-interactive back end for saving figures
import matplotlib.pyplot as plt  
import seaborn as sns 

logger = logging.getLogger(__name__)

# All metrics for correlation analysis
ALL_METRICS = ['total_score', 'sg_total', 'sg_ott', 'sg_app', 'sg_arg', 'sg_putt', 'sg_t2g', 'sg_bs', 'distance', 'accuracy', 'gir', 'prox_fw', 'prox_rgh', 'scrambling', 'great_shots', 'poor_shots']

# Features for modeling (excluding composite metrics to avoid multicollinearity)
# Excluded: sg_total (sum of all SG), sg_t2g (sg_ott + sg_app + sg_arg), sg_bs (sg_ott + sg_app)
KEY_METRICS = ['sg_ott', 'sg_app', 'sg_arg', 'sg_putt', 'distance', 'accuracy', 'gir', 'prox_fw', 'prox_rgh', 'scrambling', 'great_shots', 'poor_shots']

# =============================================================================
# Dataset Overview
# =============================================================================

def print_dataset_overview(df):
    """Print basic info about the dataset."""
    print("=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)
    print(f"Dataset shape: {df.shape}")
    print(f"Total player-tournament records: {len(df)}")
    print(f"Unique players: {df['player_name'].nunique()}")
    print(f"Years covered: {df['year'].min()} - {df['year'].max()}")
    print(f"Majors: {df['major'].unique().tolist()}")
    print()

# =============================================================================
# Descriptive Statistics by Major
# =============================================================================

def compute_metrics_by_major(df):
    """Compute descriptive statistics for all performance metrics grouped by major."""
    # Set pandas display options for clean output
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)
    results = {}
    
    # Scoring: include min/max to see the range of winning vs losing scores
    results['scoring'] = df.groupby('major')[['total_score']].agg(['mean', 'std', 'min', 'max']).round(3)
    
    # Stats for each metric group
    for group_name, metrics in FEATURE_GROUPS.items():
        results[group_name] = df.groupby('major')[metrics].agg(['mean', 'std']).round(3)
    return results

# =============================================================================
# Winner Analysis
# =============================================================================

def analyze_winners(df):
    """Analyze winning scores across majors."""
    # Filter by winners (position = '1')
    winners = df[df['position'] == '1'].groupby('major')['total_score'].describe()
    return winners.round(2)

# =============================================================================
# Top 25% vs Rest of Field Comparison
# =============================================================================

def add_top25_flag(df):
    """
    Add flag indicating if player finished in top 25% of their tournament.
    
    Top 25% is calculated per tournament (major + year) based on total_score.
    Lower scores are better in golf, so top 25% = score <= 25th percentile.
    """
    df = df.copy()
    
    # Calculate the 25th percentile per tournament
    df['tournament_25th_percentile'] = df.groupby(
        ['major', 'year'])['total_score'].transform(lambda x: x.quantile(0.25))
    
    # Mark the top 25% (score <= threshold bc lower is better in golf)
    df['is_top_25'] = df['total_score'] <= df['tournament_25th_percentile']
    return df

def compare_top25_vs_rest(df):
    """Compare performance metrics between top 25% finishers and the rest of the field."""
    if 'is_top_25' not in df.columns:
        df = add_top25_flag(df)
    
    # Compare top 25% vs the rest
    top_performers = df[df['is_top_25']]
    rest_of_field = df[~df['is_top_25']]
    
    return {'top_25_means': top_performers[KEY_METRICS].mean().round(3),
        'rest_means': rest_of_field[KEY_METRICS].mean().round(3),
        'difference': (top_performers[KEY_METRICS].mean() - 
                      rest_of_field[KEY_METRICS].mean()).round(3),
        'counts': {'top_25': len(top_performers),
            'rest': len(rest_of_field),
            'total': len(df)}}

def compute_standardized_top25_difference(df):
    """
    Compute standardized difference between top 25% and rest of field.
    This shows how many standard deviations top performers differ from the rest.
    Useful for comparing impact across metrics on different scales.
    """
    if 'is_top_25' not in df.columns:
        df = add_top25_flag(df)
    
    top_performers = df[df['is_top_25']]
    rest_of_field = df[~df['is_top_25']]
    
    # For each metric, compute the standardized difference
    std_diff = {}
    for metric in KEY_METRICS:
        pooled_std = df[metric].std()
        mean_diff = top_performers[metric].mean() - rest_of_field[metric].mean()
        std_diff[metric] = mean_diff / pooled_std
    
    return pd.Series(std_diff).round(3)

# =============================================================================
# Correlation Analysis
# =============================================================================

def compute_correlations(df):
    """Compute correlation matrix for all players."""
    return df[ALL_METRICS].corr()

def compute_winner_correlations(df):
    """Compute correlation matrix for tournament winners only."""
    winners_data = df[df['position'] == '1']
    return winners_data[ALL_METRICS].corr()

# =============================================================================
# Exploratory Visualizations
# =============================================================================

def plot_correlation_heatmap(corr_matrix, save_path=None):
    """Create a heatmap showing correlations between performance metrics."""
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                fmt='.2f', square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Between Performance Metrics', 
              fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info("Saved correlation heatmap to %s", save_path)
    plt.close()

def plot_top25_comparison(std_difference, save_path=None):
    """
    Create bar chart comparing top 25% vs rest of field.
    Shows standardized difference so we can compare metrics on different scales.
    """
    # Sort by absolute value to show most impactful metrics first
    diff_sorted = std_difference.reindex(
        std_difference.abs().sort_values(ascending=True).index)
    
    _, ax = plt.subplots(figsize=(10, 8))
    
    # Color bars based on direction (green = top 25% higher, red = lower)
    colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in diff_sorted.values]
    
    ax.barh(diff_sorted.index, diff_sorted.values, color=colors)
    plt.title('Top 25% vs Rest: Standardized Differences',
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Standardized Difference (positive = top 25% higher)', fontsize=12)
    plt.axvline(x=0, color='black', linestyle='-', linewidth=1)
    
    # Add value labels on bars
    for i, (metric, val) in enumerate(diff_sorted.items()):
        ax.text(val + 0.02 if val > 0 else val - 0.02, i,
                f'{val:.2f}', va='center', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info("Saved top 25%% comparison to %s", save_path)
    plt.close()

def plot_performance_heatmap(df, save_path=None):
    """
    Create heatmap of standardized performance metrics by major.
    Z-scores allow comparing metrics on different scales.
    """
    # Standardize each metric (z-score) then compute mean by major
    standardized = df.copy()
    for metric in KEY_METRICS:
        mean = df[metric].mean()
        std = df[metric].std()
        standardized[metric] = (df[metric] - mean) / std
    
    perf_std = standardized.groupby('major')[KEY_METRICS].mean().round(3)
    
    plt.figure(figsize=(14, 5))
    sns.heatmap(perf_std, annot=True, cmap='RdYlGn', center=0,
                fmt='.2f', linewidths=0.5, cbar_kws={'label': 'Z-Score'})
    plt.title('Performance Metrics by Major (Z-Scores)',
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Performance Metric', fontsize=12)
    plt.ylabel('Major', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info("Saved performance heatmap to %s", save_path)
    plt.close()

def plot_distribution_dashboard(df, save_path=None):
    """
    Create a single dashboard with boxplots for all performance metrics to show the distribution of each variable (instead of having 12 separate plots).
    """
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(KEY_METRICS):
        ax = axes[i]
        
        # Create boxplot by major
        df.boxplot(column=metric, by='major', ax=ax, grid=False)
        
        ax.set_title(metric, fontsize=11, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('')
        
        # Rotate x labels for readability
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    
    # Remove the automatic suptitle that pandas adds
    plt.suptitle('Performance Metrics Distribution by Major', 
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info("Saved distribution dashboard to %s", save_path)
    plt.close()

# =============================================================================
# Main Analysis Function
# =============================================================================

def run_exploratory_analysis(df, results_dir=None):
    """Run exploratory analysis, save csv data inside results and save plots inside visualizations."""
    
    logger.info("Starting exploratory analysis")
    
    # Setup results directory
    if results_dir is None:
        results_dir = Path(__file__).parent.parent / "results" / "1_Exploratory"
    else:
        results_dir = Path(results_dir)
    
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # 1. Dataset Overview (print to terminal)
    print_dataset_overview(df)
    
    # 2. Metrics by Major (save to csv)
    logger.info("Computing performance metrics by major...")
    metrics_df = df.groupby('major')[ALL_METRICS].agg(['mean', 'std']).round(3)
    metrics_df.to_csv(results_dir / "metrics_by_major.csv")
    logger.info("Saved metrics to %s", results_dir / "metrics_by_major.csv")
    
    # 3. Winner Analysis (save to csv)
    results['winner_stats'] = analyze_winners(df)
    results['winner_stats'].to_csv(results_dir / "winning_scores_by_major.csv")
    logger.info("Saved winning scores to %s", results_dir / "winning_scores_by_major.csv")
    
    # 4. Correlation Analysis (create heatmap)
    results['correlations'] = compute_correlations(df)
    plot_correlation_heatmap(results['correlations'],
        save_path=figures_dir / "correlation_heatmap.png")
    
    # 5. Performance by Major (create heatmap)
    plot_performance_heatmap(df, save_path=figures_dir / "performance_by_major.png")

     # 6. Distribution Dashboard (one figure with all metrics boxplots)
    plot_distribution_dashboard(df, save_path=figures_dir / "distribution_dashboard.png")

    # 7. Top 25% vs Rest Analysis
    df_with_flag = add_top25_flag(df)
    results['top25_comparison'] = compare_top25_vs_rest(df_with_flag)
    results['top25_std_diff'] = compute_standardized_top25_difference(df_with_flag)
    
    plot_top25_comparison(results['top25_std_diff'], save_path=figures_dir / "top25_comparison.png")
    
    logger.info("Exploratory analysis complete. Results saved to %s", results_dir)
    return results

# =============================================================================
# Script Entry Point
# =============================================================================

if __name__ == "__main__":
    import sys
    # Add src directory to path for imports
    SRC_DIR = Path(__file__).parent
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    
    logging.basicConfig(level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s')
    
    from data_loader import load_combined_data
    
    # Load data
    data = load_combined_data()
    
    # Run analysis
    results = run_exploratory_analysis(data)
    
    print("\nExploratory analysis complete!")