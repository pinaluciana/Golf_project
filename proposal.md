# Performance drivers in golf Majors Championships:  

## Sports Analytics Tool: regression and machine learning 

## Problem statement / motivation:  
Golf performance analytics usually focus on simple averages, like scoring average and driving distance, fairways hit. These metrics are useful but don’t explain why a player finishes at the top of the leaderboard nor why different tournaments reward different skill profiles. 
This project focuses on identifying which performance metrics drive success in the four Majors 
Championships: The Masters, PGA Championship, U.S. Open, The Open Championship, using all players who made the cut between 2020 and 2025. The goal is to understand what separates stronger from weaker performers and how the importance of key variables changes across tournaments. 

## Data:  
The data will come from the DataGolf’s website (Scratch Plus plan), which provides detailed tournament level performance metrics for each of the four Majors. Available results from 2020 to 2024 will be used for model training, while 2025 data (already available for all events) will be used for testing.  
Only players who made the cut after the first two rounds (36 holes, half the tournament) will be included, usually 50-70 players depending on the event. This ensures complete data and removes noise from amateurs, injured players and withdrawals. Data availability varies per Major (full for the PGA Championship and US Open, one year missing for The Masters and two for The Open Championship). 

### Final sample sizes including 2025:  
• PGA Championship: 474 
• US Open: 407 
• The Open Championship: 312 
• The Masters: 277 
• Total: 1,400+ 

### Main variables: 
• Scoring: tournament score  
• Performance variables: driving distance, driving accuracy, greens in regulation, fairway proximity, scrambling, great shots, poor shots,
-  Strokes gained metrics: off the tee, approach the green, around the green, putting, ball striking (Off the Tee + Approach), Tee to Green (everything except putting), total strokes gained 
• Context: player names, major name, year 
• Course variables: course par and yardage 

### Approach: 
#### Data preparation (in pandas): 
• Extract tournament level and player level data for all Majors between 2020-2025 
• Keep only players who made the cut, remove eventual withdrawn or disqualified players 
• Compute total score relative to par (normalize data for different par courses) 
• Clean, standardize and merge yearly data into one dataset per Major 

#### Analysis plan: 
1. Exploratory Analysis: 
• Compare distributions of performance metrics across Majors 
• Visualize how top performers differ from the others 
• Evaluate correlations between strokes-gained components 

2. Econometric models: 
• Pooled linear regression to explain score based on performance metrics 
• Per-Major linear regression to identify how skill importance varies across tournaments 
• Pooled logistic regression to model the probability of finishing in the top 25% of the leaderboard 

3. Machine Learning Models: 
• Pooled Random Forest and XGBoost classifiers using all players and all Majors 
• Include tournament name as a categorical feature so the models learn Major specific patterns 
• Define the classification target as finishing in the top 25% 
• Train models on 2020-2024 data and evaluate them in 2025 results 
• Evaluate performance using AUC, accuracy, confusion matrix and calibration curves to measure reliability 

## Expected challenges and solutions: 
• Potential overfitting: use a reduced number of relevant performance variables and apply simple regularization or cross-validation to confirm model robustness 
• Course differences: most Majors are played in different courses each year. Normalize scores relative to par and standardizing variables 
• Correlation between metrics: remove or regularize redundant variables when multicollinearity is high   

## Success criteria:  
• Regression and ML results broadly align and identify which performance variables most influence scoring in each Major 
• Linear regression achieves 𝑅2 between 0.3-0.5 (acceptable in sports analytics) 
• Logistic regression and Machine Learning models achieve an AUC above 0.7 

## Stretch goals: 
• SHAP value analysis to provide deeper interpretability of ML models 
• Clustering analysis (k-means): to identify skill-based player archetypes 
• Hyperparameter turning for improved ML performance