import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def combine_all_majors():
    """
    Load individual tournament datasets (including all available years) and combine them.

    This dataset includes:
    - PGA Championship (2020-2025)
    - US Open (2020-2021, 2023-2025, 2022 removed bc of missing values)
    - The Masters (2021-2025)
    - The Open Championship (2022-2025)

    Data has been already pre-processed with:
    - Removed players who didn't make the cut
    - Removed Disqualified and Withdrawn players
    - Removed US Open 2022 because the following variables were missing: gir, prox_fw, prox_rgh, scrambling, great_shot, poor_shot
    - Year column added to each dataset

    This file creates a combined dataset with all majors placed in: data/processed/all_majors_combined.csv
    """
    logger.info("Combining all majors datasets")

    root = Path(__file__).parent.parent.parent

    # Load each major's dataset
    pga = pd.read_csv(root / "data/processed/PGA_Championship/PGA_combined_data.csv")
    us_open = pd.read_csv(root / "data/processed/US_Open/US_Open_combined_data.csv")
    the_masters = pd.read_csv(
        root / "data/processed/The_Masters/Masters_combined_data.csv"
    )
    the_open = pd.read_csv(root / "data/processed/The_Open/Open_combined_data.csv")

    # Add major column to each dataset
    pga["major"] = "PGA Championship"
    us_open["major"] = "US Open"
    the_masters["major"] = "The Masters"
    the_open["major"] = "The Open Championship"

    # Combine all datasets
    df = pd.concat([pga, us_open, the_masters, the_open], ignore_index=True)

    # Save combined file
    output_path = root / "data/processed/all_majors_combined.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Combined data saved to {output_path}")
    logger.info(f"Total records: {len(df)}")

    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    combine_all_majors()
