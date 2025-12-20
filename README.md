# Performance drivers in golf Majors Championships: sports analytics

**Course:** Data Science and Advanced Programming 2025  
**Student:** Luciana Piña Strzelecki  
**Student ID:** 20382701


## Research Question
Which performance metrics drive success in golf's Major Championships, and how does skill importance vary across different tournaments (The Masters, PGA Championship, US Open, The Open Championship)?


## Background: Golf & Major Championships

**Golf scoring:** In golf lower scores mean better performance. The score represents the number of strokes per round. Tournaments consist of usually 4 rounds (played over 4 days), with each round covering 18 holes (length of a usual course). A typical round should be completed in 70-72 strokes depending on the course, called "par". Scores are shown relative to par (for example, -2 means 2 strokes below par, 2 strokes better than expected). This project analyzes total tournament scores from all 4 rounds combined, which typically range from -1 (worse) to -21 (best), with average winners around -10 (depending on the tournament).

**The Four Major Championships:**
- **The Masters:** Played every year in the same course: Augusta National Golf Club (Georgia USA). Very exclusive and small (#players), golfers can only assist if invited or if they fullfill certain criteria (past Masters winners, recent Major winners, top world-wide players and special invitations). Played in April.
- **PGA Championship:** Played in the USA only but in different courses every year.
The only Major that is limited to professional golfers, mainly from the PGA tour (PGA pros, Major winners, and top-ranked players). Played in May.
- **US Open:** Played in rotating courses accross the USA, they tend to be very difficult courses. Called Open because any player (professional or amateur) can try to qualify. Played in June.
- **The Open Championship:** Played on coastal courses in the UK and Irland. Players  from all major professional tours from around the world can qualify. Played in July.

Each Major has unique characteristics that reward different skills. This project aims to analyze which skills matter the most at each tournament.


## Performance Variables

Golf performance is measured across different categories:

### Off the Tee
- **distance:** Average driving distance
- **accuracy:** Fairways hit percentage
- **sg_ott:** Strokes Gained Off the Tee (driver performance vs. field average)

### Approach Play
- **sg_app:** Strokes Gained in Approach game (quality of shots into the green)
- **prox_fw:** Proximity from the fairway (average distance to pin from fairway)
- **prox_rgh:** Proximity from rough (average distance to pin from rough)

### Around the Green
- **sg_arg:** Strokes Gained Around the Green (short-game performance: chips, bunker shots)
- **scrambling:** % of times saving par after missing the green

### Putting
- **sg_putt:** Strokes Gained Putting (putting performance vs. field average)

### Ball Striking
- **gir:** Greens in Regulation (% of holes reaching green in expected number of strokes)

### Shot Quality
- **great_shots:** Number of high-quality shots per round
- **poor_shots:** Number of poor/penal shots per round

### Composite Metrics (excluded to avoid multicollinearity)
- **sg_total** - Overall strokes gained (sum of all SG components)
- **sg_t2g** - Tee to Green (sg_ott + sg_app + sg_arg)
- **sg_bs** - Ball Striking (sg_ott + sg_app)

**Note:** Strokes Gained metrics measure performance relative to the field average, positive SG means better than average and negative worse.


## Setup

### Create Environment
Using Conda dependencies (through environment.yml) or pip dependencies (through requirements.txt) 

#### Using Conda (with environment.yml)
conda env create -f environment.yml

conda activate Golf_project

#### Using pip (with requirements.txt)
python -m venv .venv

source .venv/bin/activate

pip install -r requirements.txt


## Usage
Run main.py file

**Runtime:** between 5 and 7 minutes

**Expected output:** 16 csv files and 11 png visualizations all saved under results/

**Sections executued**:
- Section 1: Exploratory Analysis
- Section 2: Feature Engineering
- Section 3: Econometric Models Testing and Evaluation
- Section 4: Econometric Analysis
- Section 5: Machine Learning Models Testing and Evaluation
- Section 6: Machine Learning Analysis

## Project Structure

Golf_project/
├── main.py                    # Main entry point
├── environment.yml            # Conda dependencies
├── requirements.txt           # Pip dependencies
├── proposal.md                # Project proposal
├── README.md                  # This file
├── src/                       # Source code
│   ├── data_loader.py         # Data loading
│   ├── exploratory.py         # Exploratory analysis
│   ├── feature_engineering.py # Feature preparation
│   ├── visualization.py       # Plotting functions
│   ├── data_preparation/      # Data preprocessing scripts
│   └── models/                # Econometric and ML models
│       ├── Econometric_models.py
│       ├── ML_models.py
│       └── evaluation.py
├── data/                      # Data directory
│   └── processed/             # Raw csvs per major and year
│   └── raw/                   # Combined datasets per major and all_majors_combined.csv
└── results/                   # Outputs in csvs and figures
    ├── 1_Exploratory/
    ├── 2_Econometric_models/
    └── 3_ML_models/


## Results
**Econometric Models:**
- Pooled Linear Regression: R² = 0.811
- Per-Major Linear Regressions:
    - PGA Championship: R² = 0.805
    - US Open: R² = 0.904
    - The Masters: R² = 0.920
    - The Open Championship: R² = 0.935
- Pooled Logistic Regression: accuracy = 96.53%,, ROC-AUC = 0.996
- Extension Logistic Regression with interactions: accuracy = 97.04%, ROC-AUC = 0.997

**Machine Learning Models:**
- Random Forest: accuracy = 80.99%, overfitting gap = 12.95pp, precision = 93.02%, recall = 45.98%
- XGBoost: accuracy = 90.87%, overfitting gap = 8.77pp, precision = 92.00%, recall = 79.31%

**Validation for ML Models:** Temporal split (to avoid data leakage), trained on data between 2020-2024 and tested in 2025 data. Hyperparameter tuning using 5-fold CV.


## Requirements
- Python 3.11
- pandas, numpy, scikit-learn, xgboost, statsmodels, matplotlib, seaborn, scipy, shap


# Data
**Source:** DataGolf website (Scratch Plus account) manual download of all data for the 4 tournaments between 2020-2025 

**Data preparation**: 
- Removed all players who didn't make the cut 
- Removed players who got disqualified (DQ) or withdrawn (WD)
- Excluded US Open 2022 because the following metrics were unavailable: gir, prox_fw, prox_rgh, scrambling, great_shot, poor_shot.

**Sample:** 1,384 player-tournament records

**Tournaments:** The Masters, PGA Championship, US Open, The Open Championship  


## Reproducibility
All the results are reproducible using random_state=42.