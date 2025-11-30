"""
Exploratory analysis of performance metrics across Major Championships.

This module computes statistics and prepares data for visualization.
Plotting functions are in evaluation.py.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# =============================================================================
# Feature Group Definitions (used across the project)
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

# Key performance metrics for visualizations (excluding composite metrics)
KEY_METRICS = [
    'sg_ott', 'sg_app', 'sg_arg', 'sg_putt',
    'distance', 'accuracy', 'gir', 'prox_fw', 'prox_rgh', 'scrambling',
    'great_shots', 'poor_shots'
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
# Overall Performance by Major (for all players)
# =============================================================================

def compute_overall_performance(df):
    """
    Compute mean performance metrics by major for ALL players.
    This is for the pooled analysis context (not just top 25%).
    
    Returns a DataFrame with majors as rows and metrics as columns.
    """
    # Compute mean of key metrics by major
    overall_perf = df.groupby('major')[KEY_METRICS].mean().round(3)
    return overall_perf


def compute_standardized_performance(df):
    """
    Compute standardized (z-score) mean performance by major.
    Standardization allows comparing metrics on different scales.
    
    Returns a DataFrame with majors as rows and standardized metrics as columns.
    """
    # Standardize each metric (z-score) then compute mean by major
    standardized = df.copy()
    for metric in KEY_METRICS:
        mean = df[metric].mean()
        std = df[metric].std()
        standardized[metric] = (df[metric] - mean) / std
    
    overall_std = standardized.groupby('major')[KEY_METRICS].mean().round(3)
    return overall_std


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
    Returns dict with top_25_means, rest_means, difference, and counts.
    """
    # Add flag if not present
    if 'is_top_25' not in df.columns:
        df = add_top25_flag(df)
    
    # Compare top 25% vs the rest
    top_performers = df[df['is_top_25'] == True]
    rest_of_field = df[df['is_top_25'] == False]
    
    # Compute means for key metrics
    top_25_means = top_performers[KEY_METRICS].mean().round(3)
    rest_means = rest_of_field[KEY_METRICS].mean().round(3)
    
    # Compute difference (top 25% - rest)
    difference = (top_25_means - rest_means).round(3)
    
    counts = {
        'top_25': len(top_performers),
        'rest': len(rest_of_field),
        'total': len(df)
    }
    
    return {
        'top_25_means': top_25_means,
        'rest_means': rest_means,
        'difference': difference,
        'counts': counts
    }


def compute_standardized_top25_difference(df):
    """
    Compute standardized difference between top 25% and rest of field.
    This shows how many standard deviations top performers differ from the rest.
    
    Useful for comparing impact across metrics on different scales.
    """
    if 'is_top_25' not in df.columns:
        df = add_top25_flag(df)
    
    top_performers = df[df['is_top_25'] == True]
    rest_of_field = df[df['is_top_25'] == False]
    
    # For each metric, compute standardized difference
    std_diff = {}
    for metric in KEY_METRICS:
        # Use pooled standard deviation
        pooled_std = df[metric].std()
        mean_diff = top_performers[metric].mean() - rest_of_field[metric].mean()
        std_diff[metric] = mean_diff / pooled_std
    
    return pd.Series(std_diff).round(3)


def print_top25_comparison(df):
    """Print comparison between top 25% finishers and rest of field."""
    df = add_top25_flag(df)
    comparison = compare_top25_vs_rest(df)
    
    print("=" * 60)
    print("TOP 25% vs REST OF FIELD COMPARISON")
    print("=" * 60)
    print(f"Top 25% per tournament: {comparison['counts']['top_25']} player-records")
    print(f"Rest of field: {comparison['counts']['rest']} player-records")
    
    # Verify if the top 25% + the rest equals actual total players
    print(f"Verification: {comparison['counts']['top_25']} + {comparison['counts']['rest']} = {comparison['counts']['top_25'] + comparison['counts']['rest']} (should equal {comparison['counts']['total']})")
    
    print("\nTop 25% average (across all tournaments):")
    print(comparison['top_25_means'])
    
    print("\nRest of field average:")
    print(comparison['rest_means'])
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


def compute_all_players_correlations(df):
    """
    Compute correlation matrix for all players (not just winners).
    Useful for understanding general relationships in the data.
    """
    return df[ALL_METRICS].corr()


# =============================================================================
# Main Function - Run All Exploratory Analysis
# =============================================================================

def run_exploratory_analysis(df):
    """
    Run the complete exploratory analysis pipeline.
    Returns dict with all computed results (no plotting here).
    
    Plotting is done separately via evaluation.py functions.
    """
    logger.info("Starting exploratory analysis")
    
    results = {}
    
    # 1. Dataset Overview
    print_dataset_overview(df)
    
    # 2. Metrics by Major
    print_metrics_by_major(df)
    results['metrics_by_major'] = compute_metrics_by_major(df)
    
    # 3. Winner Analysis
    print_winner_analysis(df)
    results['winner_stats'] = analyze_winners(df)
    
    # 4. Overall Performance (all players, by major)
    results['overall_performance'] = compute_overall_performance(df)
    results['overall_performance_std'] = compute_standardized_performance(df)
    
    # 5. Top 25% Comparison
    df_with_flag = add_top25_flag(df)
    print_top25_comparison(df_with_flag)
    results['top25_comparison'] = compare_top25_vs_rest(df_with_flag)
    results['top25_std_difference'] = compute_standardized_top25_difference(df_with_flag)
    
    # 6. Correlations
    results['winner_correlations'] = compute_winner_correlations(df)
    results['all_correlations'] = compute_all_players_correlations(df)
    
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
    
    # Import data loader
    from data_loader import load_combined_data
    
    # Load data
    df = load_combined_data()
    
    # Run analysis (returns results dict, no plots)
    results = run_exploratory_analysis(df)
    
    # To generate plots, use evaluation.py functions:
    # from evaluation import plot_exploratory_results
    # plot_exploratory_results(results, df)