# NFL Draft Prediction

Predicts the probability that a college football player gets selected in the NFL Draft, based on his position, school and NFL Combine results.

Final project for the competition in the **GCI World Data Science Program** (Matsuo-Iwasawa Lab, University of Tokyo).

## Data

- **Training set:** 2,781 players, labeled as drafted or not
- **Test set:** 696 players
- **Features:** position and position group, school, age, height, weight, and Combine drill results (40-yard sprint, vertical jump, bench press reps, broad jump, 3-cone drill, shuttle)

The dataset was provided by the course and is not included in this repository.

## Approach

**Feature engineering**
- **Position-relative performance:** each drill result compared to the median for the player's position and position group, since a "good" 40-yard time is very different for a lineman and a wide receiver
- **Missing drills:** a count of skipped Combine drills per player, since skipping tests can itself be a signal
- **Derived athletic metrics:** BMI, weight-adjusted speed score, combined jump distance, bench reps relative to body weight
- **Encoding:** frequency encoding for school and position, plus a missing-age indicator

**Modeling**

| Model | Best CV AUC |
|-------|-------------|
| Support Vector Machine (tuned) | 0.726 |
| Random Forest (tuned, with engineered features) | 0.819 |
| XGBoost (tuned, with engineered features) | 0.833 |
| **CatBoost (final model)** | **0.836** |

The tree-based boosting models clearly outperformed SVM and Random Forest. XGBoost and CatBoost scored almost the same; CatBoost was chosen for its slightly higher score and its native handling of categorical features (position, position group, player type), which avoids manual encoding. Trained with 5-fold stratified cross-validation; final test predictions average the 5 fold models.

## Results

| Fold | AUC |
|------|-----|
| 1 | 0.807 |
| 2 | 0.853 |
| 3 | 0.869 |
| 4 | 0.799 |
| 5 | 0.851 |
| **Mean** | **0.836 ± 0.027** |

## Tech stack

Python, pandas, NumPy, scikit-learn, CatBoost, Matplotlib, Seaborn

## How to run

1. Install dependencies: `pip install -r requirements.txt`
2. Put `train.csv`, `test.csv` and `sample_submission.csv` in an `input/` folder
3. Run `nfl-draft-prediction.ipynb` (originally built in Google Colab)
