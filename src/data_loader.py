import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def load_data():
    """
    Load the pre-processed combined majors dataset.
    
    This dataset includes:
    - PGA Championship (2020-2025)
    - US Open (2020-2021, 2023-2025, 2022 removed bc of missing values)
    - The Masters (2021-2025)
    - The Open Championship (2022-2025)
    
    Data has been pre-processed with:
    - Removed players who didn't make the cut
    - Removed Disqualified and Withdrawn players 
    - Removed US Open 2022 because the following variables were missing: gir, prox_fw, prox_rgh, scrambling, great_shot, poor_shot
    - Year column added to each dataset
    - Major column added to identify the tournament
    
    Returns:
        pd.DataFrame: Combined dataset with all majors
    """
    logger.info("Loading majors datasets")
    
    root = Path(__file__).parent.parent
    
    # Load each major's dataset
    pga = pd.read_csv(root / "data/processed/PGA_Championship/PGA_combined_data.csv")
    us_open = pd.read_csv(root / "data/processed/US_Open/US_Open_combined_data.csv")
    the_masters = pd.read_csv(root / "data/processed/The_Masters/Masters_combined_data.csv")
    the_open = pd.read_csv(root / "data/processed/The_Open/Open_combined_data.csv")
    
    # Add major column to each dataset
    pga['major'] = 'PGA Championship'
    us_open['major'] = 'US Open'
    the_masters['major'] = 'The Masters'
    the_open['major'] = 'The Open Championship'
    
    # Combine all datasets
    df = pd.concat([pga, us_open, the_masters, the_open], ignore_index=True)
    
    # Save combined file
    output_path = root / "data/processed/all_majors_combined.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Combined data saved to {output_path}")

    logger.info(f"Loaded {len(df)} player & tournament records")
    logger.info(f"Years included: {sorted(df['year'].unique())}")
    logger.info(f"Majors: {df['major'].unique().tolist()}")
    
    return df

def load_combined_data():
    """
    Load the already-combined majors dataset: all_majors_combined.csv,
    to be able to reuse data easily later on.
    """
    root = Path(__file__).parent.parent
    data_path = root / "data" / "processed" / "all_majors_combined.csv"
    
    # Add check to help debug if path is wrong
    if not data_path.exists():
        logger.error(f"Cannot find data at: {data_path}")
        logger.error(f"Current working directory: {Path.cwd()}")
        logger.error(f"Script location: {Path(__file__)}")
        raise FileNotFoundError(f"Data file not found. Expected at: {data_path}")
    
    df = pd.read_csv(data_path) 
    logger.info(f"Loaded {len(df)} player & tournament records")
    return df

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = load_data()
    print(f"\nDataset shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head())