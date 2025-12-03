import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def load_combined_data():
    """
    Load the pre-processed & combined majors dataset: all_majors_combined.csv,
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