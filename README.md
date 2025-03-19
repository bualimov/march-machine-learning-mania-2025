# March Machine Learning Mania 2025

This repository contains a machine learning solution for predicting NCAA basketball tournament outcomes as part of the "March Machine Learning Mania 2025" competition.

Collaborators: [Bakhtiyor Alimov](github.com/bualimov), [Dimitrios Papazekos](github.com/dimitrios06)

## Project Overview

The goal of this project is to predict the probability of each possible matchup in the 2025 NCAA men's and women's basketball tournaments. Our model uses historical tournament data along with regular season statistics to generate predictions that optimize for the Brier score.

### Key Features

- Comprehensive feature engineering using multiple data sources
- Empirical weight calculation based on historical tournament outcomes
- Season-specific performance metrics with time-weighted historical context
- Conference strength analysis
- Coach performance evaluation
- Optimized prediction range for tournament-style competitions

## Data Sources

The project uses official NCAA basketball data, including:
- Regular season game results
- Tournament outcomes from 2003-2024
- Team seedings
- Conference affiliations
- Coaches information
- Team ranking systems (Massey Ordinals)

## Methodology

Our prediction model employs a weighted feature approach with empirically derived weights:

| Feature | Weight |
|---------|--------|
| Tournament Seeding | 22% |
| Point Differential | 14% |
| End-of-season Ranking | 13% |
| Win Percentage | 11% |
| Conference Strength | 9% |
| Strength of Schedule | 8% |
| Recent Form (Last 10 games) | 6% |
| Coach Success Rate | 5% |
| Historical Tournament Success | 5% |
| Tournament Appearance Consistency | 4% |
| Coach Experience | 3% |

Each feature is normalized and combined to produce a balanced prediction that avoids extreme probabilities. Historical data is weighted by recency, giving more importance to recent tournament performance.

## Repository Structure

```
march-machine-learning-mania-2025/
├── march-machine-learning-mania-2025-final-dataset/  # Data files (not in repo)
├── src/                                              # Source code
│   ├── march_madness_2025_comprehensive_model.py     # Main prediction model
│   ├── march_madness_2025_empirical_weights.py       # Historical weight analysis
│   ├── consolidate_predictions.py                    # Combines men's/women's predictions
│   └── other model files                             # Additional model implementations
├── predictions_M_2025_only.csv                       # Men's tournament predictions
├── predictions_W_2025_only.csv                       # Women's tournament predictions
├── predictions_2025_final_submission.csv             # Combined final submission
└── README.md                                         # This file
```

## How to Run

### Prerequisites

- Python 3.8+
- pandas
- numpy
- scikit-learn

Install required packages using:

```bash
pip install -r requirements.txt
```

### Step 1: Calculate Empirical Weights

First, analyze historical tournament outcomes to derive feature weights:

```bash
python src/march_madness_2025_empirical_weights.py
```

This will generate `empirical_weights_M.json` and `empirical_weights_W.json` files containing the importance of each feature based on historical outcomes.

### Step 2: Generate Predictions

Run the comprehensive model to generate predictions for 2025:

```bash
python src/march_madness_2025_comprehensive_model.py
```

This will produce separate prediction files for men's and women's tournaments with IDs in the format `2025_XXXX_YYYY`.

### Step 3: Consolidate Predictions (Optional)

Combine men's and women's predictions into a single submission file:

```bash
python src/consolidate_predictions.py
```

The final predictions will be saved to `predictions_2025_final_submission.csv`.

## Results

Our model produces balanced predictions with:
- 4,556 total matchup predictions (2,278 men's, 2,278 women's)
- Prediction range: 0.15 to 0.85
- Mean probability: ~0.54
- Standard deviation: ~0.22
- Distribution: Even spread across probabilities with most falling in the 0.3-0.7 range

The predictions follow the required format for submission with IDs in the format `2025_XXXX_YYYY` and probabilities indicating the likelihood of the lower-ID team beating the higher-ID team.

## Acknowledgments

- NCAA for providing the historical tournament data
- All contributors to the sklearn, pandas, and numpy libraries 