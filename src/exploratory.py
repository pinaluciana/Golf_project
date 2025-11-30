"""
Exploratory analysis of performance metrics across Major Championships.

This module analyzes golf performance data across the four majors:
PGA Championship, US Open, The Masters, and The Open Championship.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# =============================================================================
# Feature Group Definitions
# =============================================================================

# Define feature groups by category
# These groupings match how golf performance is typically analyzed
FEATURE_GROUPS = {
    'off_tee': ['distance', 'accuracy', 'sg_ott'],
    'approach': ['sg_app', 'prox_fw', 'prox_rgh'],
    'short_game': ['sg_arg', 'scrambling'],
    'putting': ['sg_putt'],
    'tee_to_green': ['sg_bs', 'gir', 'sg_t2g'],
    'overall': ['sg_total', 'great_shots', 'poor_shots']
}

# All metrics used for correlation and comparison
ALL_METRICS = [
    'total_score', 'sg_total', 'sg_ott', 'sg_app', 'sg_arg', 'sg_putt',
    'sg_t2g', 'sg_bs', 'distance', 'accuracy', 'gir',
    'prox_fw', 'prox_rgh', 'scrambling', 'great_shots', 'poor_shots'
]


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
    """
    Compute descriptive statistics for all performance metrics grouped by major.
    Returns a dict with stats for each metric category.
    """
    # Set pandas display options for clean output
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)
    
    results = {}
    
    # Scoring - include min/max to see the range of winning vs losing scores
    results['scoring'] = df.groupby('major')[['total_score']].agg(
        ['mean', 'std', 'min', 'max']
    ).round(3)
    
    # Off the Tee
    results['off_tee'] = df.groupby('major')[FEATURE_GROUPS['off_tee']].agg(
        ['mean', 'std']
    ).round(3)
    
    # Approach
    results['approach'] = df.groupby('major')[FEATURE_GROUPS['approach']].agg(
        ['mean', 'std']
    ).round(3)
    
    # Around the Green (Short Game)
    results['short_game'] = df.groupby('major')[FEATURE_GROUPS['short_game']].agg(
        ['mean', 'std']
    ).round(3)
    
    # Putting
    results['putting'] = df.groupby('major')[FEATURE_GROUPS['putting']].agg(
        ['mean', 'std']
    ).round(3)
    
    # Tee to Green
    results['tee_to_green'] = df.groupby('major')[FEATURE_GROUPS['tee_to_green']].agg(
        ['mean', 'std']
    ).round(3)
    
    # Overall Performance
    results['overall'] = df.groupby('major')[FEATURE_GROUPS['overall']].agg(
        ['mean', 'std']
    ).round(3)
    
    return results


def print_metrics_by_major(df):
    """Print descriptive statistics for all metrics grouped by major."""
    results = compute_metrics_by_major(df)
    
    print("=" * 60)
    print("PERFORMANCE METRICS BY MAJOR")
    print("=" * 60)
    
    # Group metrics by type for organized output
    print("\nSCORING")
    print(results['scoring'])
    print()
    
    print("OFF THE TEE")
    print(results['off_tee'])
    print()
    
    print("APPROACH")
    print(results['approach'])
    print()
    
    print("AROUND THE GREEN")
    print(results['short_game'])
    print()
    
    print("PUTTING")
    print(results['putting'])
    print()
    
    print("TEE TO GREEN")
    print(results['tee_to_green'])
    print()
    
    print("OVERALL PERFORMANCE")
    print(results['overall'])
    print()


# =============================================================================
# Winner Analysis
# =============================================================================

def analyze_winners(df):
    """
    Analyze winning scores across majors.
    Returns descriptive stats of winning scores by major.
    """
    # Filter by winners (position = '1')
    winners = df[df['position'] == '1'].groupby('major')['total_score'].describe()
    return winners.round(2)


def print_winner_analysis(df):
    """Print winning scores analysis."""
    print("=" * 60)
    print("WINNING SCORES BY MAJOR")
    print("=" * 60)
    print(analyze_winners(df))
    print()


# =============================================================================
# Top 25% vs Rest of Field Comparison
# =============================================================================

def add_top25_flag(df):
    """
    Add a flag indicating whether each player finished in the top 25% of their tournament.
    
    Top 25% is calculated per tournament (major + year) based on total_score.
    Lower scores are better in golf, so top 25% = score <= 25th percentile.
    """
    # Create a copy to avoid modifying the original
    df = df.copy()
    
    # Calculate the 25th percentile per tournament
    df['tournament_25th_percentile'] = df.groupby(['major', 'year'])['total_score'].transform(
        lambda x: x.quantile(0.25)
    )
    
    # Mark the top 25% (score <= threshold bc lower is better)
    df['is_top_25'] = df['total_score'] <= df['tournament_25th_percentile']
    
    return df


def compare_top25_vs_rest(df):
    """
    Compare performance metrics between top 25% finishers and the rest of the field.
    Returns tuple: (top_25_means, rest_means, counts)
    """
    # Add flag if not present
    if 'is_top_25' not in df.columns:
        df = add_top25_flag(df)
    
    # Compare top 25% vs the rest
    top_performers = df[df['is_top_25'] == True]
    rest_of_field = df[df['is_top_25'] == False]
    
    # Compare performance variables
    comparison_metrics = [
        'total_score', 'sg_total', 'sg_ott', 'sg_app', 'sg_arg', 'sg_putt',
        'sg_t2g', 'sg_bs', 'distance', 'accuracy', 'gir',
        'prox_fw', 'prox_rgh', 'scrambling', 'great_shots', 'poor_shots'
    ]
    
    top_25_means = top_performers[comparison_metrics].mean().round(3)
    rest_means = rest_of_field[comparison_metrics].mean().round(3)
    
    counts = {
        'top_25': len(top_performers),
        'rest': len(rest_of_field),
        'total': len(df)
    }
    
    return top_25_means, rest_means, counts


def print_top25_comparison(df):
    """Print comparison between top 25% finishers and rest of field."""
    df = add_top25_flag(df)
    top_25_means, rest_means, counts = compare_top25_vs_rest(df)
    
    print("=" * 60)
    print("TOP 25% vs REST OF FIELD COMPARISON")
    print("=" * 60)
    print(f"Top 25% per tournament: {counts['top_25']} player-records")
    print(f"Rest of field: {counts['rest']} player-records")
    
    # Verify if the top 25% + the rest equals actual total players
    print(f"Verification: {counts['top_25']} + {counts['rest']} = {counts['top_25'] + counts['rest']} (should equal {counts['total']})")
    
    print("\nTop 25% average (across all tournaments):")
    print(top_25_means)
    
    print("\nRest of field average:")
    print(rest_means)
    print()


# =============================================================================
# Correlation Analysis
# =============================================================================

def compute_winner_correlations(df):
    """
    Compute correlation matrix for performance metrics among tournament winners.
    This helps identify which metrics tend to move together for winning players.
    """
    # Filter by winners
    winners_data = df[df['position'] == '1']
    
    # Include all metrics to compute correlation
    winner_corr = winners_data[ALL_METRICS].corr()
    
    return winner_corr


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_metrics_boxplots(df, save_path=None):
    """
    Create boxplots comparing performance metrics across majors.
    One figure per category, with subplots for each metric.
    """
    # Set style
    sns.set_style("whitegrid")
    
    # Define metrics by category (matching the analysis structure)
    categories = {
        'Scoring': ['total_score'],
        'Off the Tee': FEATURE_GROUPS['off_tee'],
        'Approach': FEATURE_GROUPS['approach'],
        'Around the Green': FEATURE_GROUPS['short_game'],
        'Putting': FEATURE_GROUPS['putting'],
        'Tee to Green': FEATURE_GROUPS['tee_to_green'],
        'Overall Performance': FEATURE_GROUPS['overall']
    }
    
    # Create plots for each category
    for category, metrics in categories.items():
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
        fig.suptitle(f'{category}', fontsize=16, fontweight='bold')
        
        # Handle single metric (axes is not array)
        if n_metrics == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            sns.boxplot(data=df, x='major', y=metric, ax=axes[i])
            axes[i].set_title(metric.replace('_', ' ').title(), fontweight='bold')
            axes[i].set_xlabel('')
            axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            filename = f"boxplot_{category.lower().replace(' ', '_')}.png"
            plt.savefig(Path(save_path) / filename, dpi=150, bbox_inches='tight')
            logger.info(f"Saved {filename}")
        
        plt.show()


def plot_winner_correlation_heatmap(df, save_path=None):
    """
    Create a heatmap showing correlations between performance metrics for winners.
    This helps visualize which metrics are related for winning players.
    """
    # Calculate correlation matrix
    winner_corr = compute_winner_correlations(df)
    
    # Create a heatmap to visualize correlations
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        winner_corr, 
        annot=True, 
        cmap='coolwarm', 
        center=0,
        fmt='.2f', 
        square=True, 
        linewidths=1, 
        cbar_kws={"shrink": 0.8}
    )
    plt.title(
        'Correlation Between Winner Performance Metrics',
        fontsize=14, 
        fontweight='bold', 
        pad=20
    )
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved correlation heatmap to {save_path}")
    
    plt.show()


# =============================================================================
# Main Function - Run All Exploratory Analysis
# =============================================================================

def run_exploratory_analysis(df, save_plots=False, results_dir=None):
    """
    Run the complete exploratory analysis pipeline.
    
    This includes:
    - Dataset overview
    - Metrics by major (descriptive stats)
    - Winner analysis
    - Top 25% vs rest comparison
    - Winner correlations
    - Visualizations (boxplots and heatmap)
    
    Args:
        df: DataFrame with combined majors data
        save_plots: Whether to save plots to disk
        results_dir: Directory to save results (if save_plots=True)
    
    Returns:
        dict with all analysis results
    """
    logger.info("Starting exploratory analysis")
    
    # Setup results directory if saving
    if save_plots:
        if results_dir is None:
            results_dir = Path(__file__).parent.parent / "results" / "1_Exploratory"
        else:
            results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # 1. Dataset Overview
    print_dataset_overview(df)
    
    # 2. Metrics by Major
    print_metrics_by_major(df)
    results['metrics_by_major'] = compute_metrics_by_major(df)
    
    # 3. Winner Analysis
    print_winner_analysis(df)
    results['winner_stats'] = analyze_winners(df)
    
    # 4. Top 25% Comparison
    df_with_flag = add_top25_flag(df)
    print_top25_comparison(df_with_flag)
    top_25_means, rest_means, counts = compare_top25_vs_rest(df_with_flag)
    results['top25_comparison'] = {
        'top_25_means': top_25_means,
        'rest_means': rest_means,
        'counts': counts
    }
    
    # 5. Winner Correlations
    results['winner_correlations'] = compute_winner_correlations(df)
    
    # 6. Visualizations
    if save_plots:
        plot_metrics_boxplots(df, save_path=results_dir)
        plot_winner_correlation_heatmap(
            df, 
            save_path=results_dir / "winner_correlation_heatmap.png"
        )
    else:
        plot_metrics_boxplots(df)
        plot_winner_correlation_heatmap(df)
    
    logger.info("Exploratory analysis complete")
    
    return results


# =============================================================================
# Script Entry Point
# =============================================================================

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Load data
    root = Path(__file__).parent.parent
    df = pd.read_csv(root / "data" / "processed" / "all_majors_combined.csv")
    
    # Run analysis (save plots to results/exploratory/)
    results = run_exploratory_analysis(df, save_plots=True)