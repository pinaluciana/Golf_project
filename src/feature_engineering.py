"""
Feature Engineering for Golf Performance Analysis.
This file contains shared feature definitions and data preparation functions that will be used for both econometric and ML models. 
In order to avoid code duplication.

Note: Missing value handling is already taken care of in data_loader.py and combine_all_majors.py.
"""

import logging
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# =============================================================================
# Feature Definitions
# =============================================================================

# Performance metrics grouped by category (for reference and analysis)
# These groupings match how golf performance is typically analyzed

OFF_TEE = ['distance', 'accuracy', 'sg_ott']
APPROACH = ['sg_app', 'prox_fw', 'prox_rgh']
SHORT_GAME = ['sg_arg', 'scrambling']
PUTTING = ['sg_putt']
BALL_STRIKING = ['gir']
SHOT_QUALITY = ['great_shots', 'poor_shots']

# Combined list of all features used in modeling
# Excluded to avoid multicollinearity:
# - sg_total: bc it's the sum of all strokes gained metrics
# - sg_t2g: sum of sg_ott + sg_app + sg_arg
# - sg_bs: sum of sg_ott + sg_app
FEATURES = OFF_TEE + APPROACH + SHORT_GAME + PUTTING + BALL_STRIKING + SHOT_QUALITY

# All metrics including composite ones (for exploratory analysis)
ALL_METRICS = ['total_score', 'sg_total', 'sg_ott', 'sg_app', 'sg_arg', 'sg_putt',
    'sg_t2g', 'sg_bs', 'distance', 'accuracy', 'gir',
    'prox_fw', 'prox_rgh', 'scrambling', 'great_shots', 'poor_shots']

# =============================================================================
# Data Preparation Functions
# =============================================================================

def prepare_features(df, features=None, target='total_score'):
    """
    Prepare and standardize features for modeling. Standardize: so we're able to compare coefficients across features with different scales.
    
    Args:
        df: DataFrame with golf performance data
        features: List of feature columns (default: FEATURES)
        target: Target column name (default: 'total_score')
    
    Returns:
        X_scaled: Standardized features as DataFrame
        y: Target variable as Series
        scaler: Fitted StandardScaler (for later use if needed)
    """
    if features is None:
        features = FEATURES
    
    X = df[features]
    y = df[target]
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=features, index=X.index)
    
    return X_scaled, y, scaler

def create_top25_target(df):
    """
    Create binary target indicating top 25% finish per tournament. Top 25% is calculated per tournament (major & year) based on total_score.
    Note: lower scores are better in golf, so top 25% = score <= 25th percentile.
    
    Args:
        df: DataFrame with golf performance data (must have 'major', 'year', 'total_score' columns)
    
    Returns:
        DataFrame with 'top_25' column added (1 = top 25%, 0 = rest of field)
    """
    df = df.copy()
    
    # Calculate 25th percentile per tournament
    df['tournament_25th_percentile'] = df.groupby(
        ['major', 'year'])['total_score'].transform(lambda x: x.quantile(0.25))
    
    # Mark top 25% (score <= threshold bc lower is better in golf)
    df['top_25'] = (df['total_score'] <= df['tournament_25th_percentile']).astype(int)
    
    # Log the distribution
    logger.info("Created top 25%% target:")
    logger.info("  Top 25%%: %d (%.1f%%)", df['top_25'].sum(), 
                df['top_25'].sum() / len(df) * 100)
    logger.info("  Rest of field: %d (%.1f%%)", (df['top_25'] == 0).sum(),
                (df['top_25'] == 0).sum() / len(df) * 100)
    
    return df

def get_feature_groups():
    """
    Return dictionary of feature groups for reference. Useful for analysis and reporting.
    """
    return {'off_tee': OFF_TEE,
        'approach': APPROACH,
        'short_game': SHORT_GAME,
        'putting': PUTTING,
        'ball_striking': BALL_STRIKING,
        'shot_quality': SHOT_QUALITY}