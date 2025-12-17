"""
Exploratory analysis of golf performance data in Major Championships.
This module computes basic descriptive statistics and produces exploratory plots used to understand performance patterns before modeling.
"""

import logging
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # We use this non-interactive back end for saving figures
import matplotlib.pyplot as plt  
import seaborn as sns 

# To import "ALL_METRICS" for correlation analysis
from feature_engineering import FEATURES, ALL_METRICS 

logger = logging.getLogger(__name__)

# Features for modeling (excluding composite metrics to avoid multicollinearity)
# Excluded: sg_total (sum of all SG), sg_t2g (sg_ott + sg_app + sg_arg), sg_bs (sg_ott + sg_app)
KEY_METRICS = ['sg_ott', 'sg_app', 'sg_arg', 'sg_putt', 'distance', 'accuracy', 'gir', 'prox_fw', 'prox_rgh', 'scrambling', 'great_shots', 'poor_shots']

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
    df['tournament_25th_percentile'] = df.groupby(['major', 'year'])['total_score'].transform(lambda x: x.quantile(0.25))
    
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
    plt.close()

# =============================================================================
# Main Analysis Function
# =============================================================================

def run_exploratory_analysis(df, results_dir=None):
    """Run exploratory analysis, save csv data inside results and save plots inside visualizations."""
        
    # Setup results directory
    if results_dir is None:
        results_dir = Path(__file__).parent.parent / "results" / "1_Exploratory"
    else:
        results_dir = Path(results_dir)
    
    figures_dir = results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # 1. Metrics by Major (save to csv)
    metrics_df = df.groupby('major')[ALL_METRICS].agg(['mean', 'std']).round(3)
    metrics_df.to_csv(results_dir / "metrics_by_major.csv")
    
    # 2. Winner Analysis (save to csv)
    results['winner_stats'] = analyze_winners(df)
    results['winner_stats'].to_csv(results_dir / "winning_scores_by_major.csv")
    
    # 3. Correlation Analysis (create heatmap)
    results['correlations'] = compute_correlations(df)
    plot_correlation_heatmap(results['correlations'],
        save_path=figures_dir / "correlation_heatmap.png")
    
    # 4. Performance by Major (create heatmap)
    plot_performance_heatmap(df, save_path=figures_dir / "performance_by_major.png")

     # 5. Distribution Dashboard (one figure with all metrics boxplots)
    plot_distribution_dashboard(df, save_path=figures_dir / "distribution_dashboard.png")

    # 6. Top 25% vs Rest Analysis
    df_with_flag = add_top25_flag(df)
    results['top25_comparison'] = compare_top25_vs_rest(df_with_flag)
    results['top25_std_diff'] = compute_standardized_top25_difference(df_with_flag)
    
    plot_top25_comparison(results['top25_std_diff'], save_path=figures_dir / "top25_comparison.png")
    
    return results

def print_exploratory_summary(df, exploratory_results, results_dir):
    """
    Print exploratory analysis summary using results from run_exploratory_analysis().
    This reuses existing computed results rather than recalculating.
    """
    
    # Dataset Overview
    logger.info("DATASET OVERVIEW:")
    logger.info(f"  Dataset shape: {df.shape}")
    logger.info(f"  Total player-tournament records: {len(df)}")
    logger.info(f"  Unique players: {df['player_name'].nunique()}")
    logger.info(f"  Years covered: {df['year'].min()} - {df['year'].max()}")
    logger.info(f"  Majors: {df['major'].unique().tolist()}")
    
    # Exploratory Summary
    print()  # Blank line
    logger.info("EXPLORATORY SUMMARY:")
    
    # Winning Scores - computed fresh from data
    logger.info("Winning Scores per Major:")
    winners = df[df['position'] == '1'].groupby('major')['total_score'].agg(['mean', 'min', 'max'])
    for major in winners.index:
        logger.info(f"  {major:<25} Mean: {winners.loc[major, 'mean']:>6.2f}  "
              f"Best: {int(winners.loc[major, 'min']):>3}  "
              f"Worst: {int(winners.loc[major, 'max']):>3}")
    
    # Top 25% Distribution - reuses exploratory_results
    print()  # Blank line
    logger.info("Top 25% Distribution:")
    top25_count = exploratory_results['top25_comparison']['counts']
    logger.info(f"  Top 25%:       {top25_count['top_25']:>4} players ({top25_count['top_25']/top25_count['total']*100:.1f}%)")
    logger.info(f"  Rest of field: {top25_count['rest']:>4} players ({top25_count['rest']/top25_count['total']*100:.1f}%)")
    
    # Correlations - reuses exploratory_results['correlations']
    print()  # Blank line
    logger.info("Top 5 Correlations with Total Score:")
    score_corr = exploratory_results['correlations']['total_score'].drop('total_score').abs().sort_values(ascending=False).head(5)
    for feature in score_corr.index:
        corr_val = exploratory_results['correlations'].loc[feature, 'total_score']
        direction = "negative" if corr_val < 0 else "positive"
        logger.info(f"  {feature:<15} {corr_val:>7.3f} ({direction})")
    
    # Standardized Differences - reuses exploratory_results['top25_std_diff']
    print()  # Blank line
    logger.info("Top 5 Standardized Differences (Top 25% vs Rest):")
    for feature, diff in exploratory_results['top25_std_diff'].head(5).items():
        logger.info(f"  {feature:<15} {diff:>7.3f} std deviations")
    
    # Files Created - lists files in results_dir
    print("\n" + "-"*100)
    logger.info("FILES CREATED:")
    figures_dir = results_dir / "figures"
    csv_files = sorted(results_dir.glob('*.csv'))
    png_files = sorted(figures_dir.glob('*.png'))
    
    for csv_file in csv_files:
        logger.info(f"  Saved {csv_file.name}")
    for png_file in png_files:
        logger.info(f"  Saved {png_file.name}")
    print("-"*100)
