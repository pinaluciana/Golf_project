"""
Main Analysis Pipeline for Golf Performance in Major Championships.
This script executes the complete analysis from exploratory analysis to final results.

Project: Performance Drivers in Major Championships
Course: Data Science and Advanced Programming 2025
"""

import sys
import logging
from pathlib import Path

# Add src directory to path
SRC_DIR = Path(__file__).parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Silence matplotlib font manager info logs to have a clean terminal output
logging.getLogger('matplotlib').setLevel(logging.WARNING)

# Added function to count lines of code per file to include at the end of the terminal output
def count_lines_per_file(base_dir):
    ignore_dirs = {".venv", "venv", "__pycache__", "results", ".git", ".pytest_cache", "dist", "build"}
    total = 0
    paths = [base_dir / "main.py", base_dir / "src"]
    for p in paths:
        for py_file in sorted(p.rglob("*.py")) if p.is_dir() else [p]:
            # skip ignored dirs
            if any(part in ignore_dirs for part in py_file.parts):
                continue

            with open(py_file, "r", encoding="utf-8") as f:
                loc = sum(1 for line in f)

            rel = py_file.relative_to(base_dir)
            print(f"  {rel}: {loc}")
            total += loc

    return total

# Main execution function
def main():
    """Execute whole project."""
    
    # Project and student info
    print("\n" + "="*100)
    print("PROJECT: Performance drivers in golf Majors Championships")
    print("COURSE: Data Science and Advanced Programming 2025")
    print("NAME: Luciana Piña Strzelecki")
    print("STUDENT ID: 20382701")
    print("="*100 + "\n")

    # Note on runtime 
    print("\n" + "="*100)
    logger.info("NOTE: Complete project runtime: approximately 5-7 minutes, includes:")
    logger.info("  Section 1: Exploratory Analysis" )
    logger.info("  Section 2: Feature Engineering" )
    logger.info("  Section 3: Econometric Models Testing and Evaluation" )
    logger.info("  Section 4: Econometric Analysis" )
    logger.info("  Section 5: Machine Learning Models Testing and Evaluation" )
    logger.info("  Section 6: Machine Learning Analysis" )
    print("="*100 + "\n")
    
    # Data preprocessing note
    logger.info("DATA PREPROCESSING:")
    logger.info("  Data was preprocessed by major by merging each year's CSVs")
    logger.info("  Removed players who didn't make the cut, got disqualified (DQ), or withdrawn (WD)")
    logger.info("  US Open 2022 was removed due to missing variables")
    logger.info("  All majors were merged into one combined CSV")
    logger.info("  Data loaded via data_loader.py for all subsequent analyses")
    
    # Import modules
    from data_loader import load_combined_data
    from exploratory import run_exploratory_analysis, print_exploratory_summary
    import feature_engineering
    from models.Econometric_models import run_econometric_analysis
    from models.ML_models import run_ml_analysis
    from models.evaluation import (evaluate_ml_models, print_econometric_summary, evaluate_econometric_models, print_ml_summary)
    from visualization import create_econometric_visualizations, create_ml_visualizations
    
    # Load data
    data = load_combined_data()

    # =========================================================================
    # SECTION 1: EXPLORATORY ANALYSIS
    # =========================================================================
    print("\n" + "="*100)
    print("SECTION 1: EXPLORATORY ANALYSIS")
    print("="*100)
    
    results_dir_exp = Path(__file__).parent / "results" / "1_Exploratory"
    exploratory_results = run_exploratory_analysis(data, results_dir=results_dir_exp)
    
    # Print exploratory summary
    print()  # Blank line for readability
    print_exploratory_summary(data, exploratory_results, results_dir_exp)
    
    # =========================================================================
    # SECTION 2: FEATURE ENGINEERING
    # =========================================================================
    print("\n" + "="*100)
    print("SECTION 2: FEATURE ENGINEERING")
    print("="*100 +"\n")
    
    logger.info("FEATURE GROUPS (included for interpretation/reference. However, all 12 features are used in the models):")
    groups = feature_engineering.get_feature_groups()
    for group_name, features in groups.items():
        feature_list = ', '.join(features)
        logger.info(f"  {group_name.upper().replace('_', ' ')}: {feature_list}")
    
    print()  # Blank line for readability
    logger.info("EXCLUDED COMPOSITE METRICS TO AVOID MULTICOLLINEARITY:")
    logger.info("  sg_total   (sum of all strokes gained metrics)")
    logger.info("  sg_t2g     (sum of sg_ott + sg_app + sg_arg)")
    logger.info("  sg_bs      (sum of sg_ott + sg_app)")
    
    print()  # Blank line for readability
    logger.info(f"FINAL FEATURE SET ({len(feature_engineering.FEATURES)} features):")
    for i, feature in enumerate(feature_engineering.FEATURES, 1):
        logger.info(f"  {i:2d}. {feature}")
    
    # =========================================================================
    # SECTION 3: ECONOMETRIC MODELS TESTING AND EVALUATION
    # =========================================================================
    print("\n" + "="*100)
    print("SECTION 3: ECONOMETRIC MODELS TESTING AND EVALUATION")
    print("="*100)
    
    results_dir_econ = Path(__file__).parent / "results" / "2_Econometric_models"
    
    # Run all econometric models
    econ_results = run_econometric_analysis(data, results_dir=results_dir_econ)

    # Print evaluation metrics
    evaluate_econometric_models(econ_results, data)
    
    # =========================================================================
    # SECTION 4: ECONOMETRIC ANALYSIS
    # =========================================================================
    print("\n" + "="*100)
    print("SECTION 4: ECONOMETRIC ANALYSIS")
    print("="*100)
    
    # Print detailed summary
    print_econometric_summary(econ_results, data, results_dir_econ)

    # Generate visualizations
    create_econometric_visualizations(econ_results, results_dir_econ)

    # =========================================================================
    # SECTION 5: MACHINE LEARNING MODELS TESTING AND EVALUATION
    # =========================================================================
    print("\n" + "="*100)
    print("SECTION 5: MACHINE LEARNING MODELS TESTING AND EVALUATION")
    print("="*100)

    results_dir_ml = Path(__file__).parent / "results" / "3_ML_models"

    # Run ML models
    ml_results = run_ml_analysis(data, results_dir=results_dir_ml)

    # Print evaluation metrics
    evaluate_ml_models(ml_results, results_dir_ml)
    
    # =========================================================================
    # SECTION 6: MACHINE LEARNING ANALYSIS
    # =========================================================================
    print("\n" + "="*100)
    print("SECTION 6: MACHINE LEARNING ANALYSIS")
    print("="*100)

    # Print ML summary
    print_ml_summary(ml_results, results_dir_ml)

    # Generate visualizations
    create_ml_visualizations(ml_results, results_dir_ml)
    
    # =========================================================================
    # PROJECT COMPLETE
    # =========================================================================
    print("\n" + "="*100)
    print("END OF RESULTS AND ANALYSIS. THE GOLF PROJECT WAS COMPLETED SUCCESSFULLY!")
    print("="*100 + "\n")

    print("-"*100)
    print("SUMMARY OF FILES CREATED AND LINES OF CODE:")
    print("-"*100 + "\n")

    results_base = Path(__file__).parent / "results"
    total_csvs = len(list(results_base.rglob('*.csv')))
    total_figs = len(list(results_base.rglob('*.png')))
    
    print(f"TOTAL FILES CREATED:")
    print(f" Created {total_csvs} CSV result files")
    print(f" Created {total_figs} visualization figures")
    print(f" All files saved to: {results_base}")

    print("\nTOTAL LINES OF CODE (main.py + src/):")
    base_dir = Path(__file__).parent
    total_loc = count_lines_per_file(base_dir)

    print(f"  TOTAL LINES OF CODE: {total_loc}")
    print("\n" + "="*100 + "\n")


if __name__ == "__main__":
    main()