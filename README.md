# March Machine Learning Mania 2025

Welcome! This repository contains our machine learning solution for predicting NCAA basketball tournament outcomes (March Madness '25) as part of the "March Machine Learning Mania 2025" competition.

Collaborators: [Bakhtiyor Alimov](https://github.com/bualimov), [Dimitrios Papazekos](https://github.com/dimitrios06)

## Project Overview

The goal of this project is to predict the probability of every single possible matchup in the 2025 NCAA men's and women's basketball tournaments. We used historical tournament data along with regular season stats to generate predictions that optimize for the Brier score.

### Features

- Comprehensive feature engineering using multiple data sources
- Empirical weight calculation based on previous tournament outcomes
- Season-specific performance metrics, with time taken into account
- Conference strength analysis
- Coach strength analysis
- Tournament history and upset pattern analysis
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

### Submission 1: Base Model Weights

Our first prediction model employs a weighted feature approach with empirically derived weights:

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

### Submission 2: Optimized Model Weights

Our optimized model builds on the base model but adds several tournament-specific adjustments:

| Feature | Base Weight | Additional Adjustments |
|---------|-------------|------------------------|
| Base Model Prediction | 100% | Starting point |
| Seed Matchup Historical Outcomes | +30% | Applied to base prediction |
| Team Tournament Win Percentage | +5% | Differential between teams |
| Team Upset Factor | +3% | Historical upset creation vs suffering |
| Team Clutch Performance | +2% | Performance in close games |
| Team Dominance Factor | +2% | Performance in blowout games |

The optimized model also applies:
- Isotonic regression calibration based on 2018-2024 tournaments
- Stronger recency bias (0.85 decay factor vs 0.9 in submission1)
- Probability clipping to range [0.15, 0.85]
- Tournament-specific patterns instead of regular season patterns

## Repository Structure

```
march-machine-learning-mania-2025/
├── march-machine-learning-mania-2025-final-dataset/  # Data files (not in repo)
├── src/                                              # Source code
│   ├── march_madness_2025_comprehensive_model.py     # Main prediction model
│   ├── march_madness_2025_empirical_weights.py       # Historical weight analysis
│   ├── generate_final_submission.py                  # Creates submission1.csv
│   ├── generate_optimized_submission.py              # Creates submission2.csv
│   └── other model files                             # Additional model implementations
├── submissions/                                      # Submission files
│   ├── submission1.csv                               # Standard submission
│   └── submission2.csv                               # Optimized submission
└── README.md                                         # This file
```

## How to Run

### Prerequisites

- Python 3.8+
- pandas
- numpy
- scikit-learn
- tqdm

Install required packages using:

```bash
pip install -r requirements.txt
```

### Generate Standard Submission (submission1.csv)

To generate the standard submission with our comprehensive model:

```bash
python -m src.generate_final_submission
```

This script:
1. Selects exactly 363 active men's teams (excluding MS Valley St)
2. Maps these to 363 corresponding women's teams 
3. Generates 65,703 matchup predictions for men + 65,703 for women
4. Creates a CSV with 131,406 predictions + 1 header row (131,407 total)

### Generate Optimized Submission (submission2.csv)

To generate the optimized submission with improved Brier score:

```bash
python -m src.generate_optimized_submission
```

This script:
1. Uses the same teams as submission1
2. Applies tournament history analysis and upset pattern detection
3. Calibrates probabilities using isotonic regression
4. Adjusts predictions based on historical performance
5. Creates a CSV with 131,406 predictions + 1 header row (131,407 total)

## Key Differences Between Submissions

### Submission 1 (Base Model)
- Uses all historical data from 2003-2024 with moderate recency bias
- Relies primarily on regular season performance metrics
- Focuses on team-specific statistics and rankings
- Balanced feature weighting across multiple attributes
- Raw predictions with minimal calibration

### Submission 2 (Optimized Model)
- Focuses heavily on tournament-specific patterns from 2010-2024
- Analyzes historical upset patterns and seed matchup outcomes
- Uses isotonic regression to calibrate probabilities (achieved 8% Brier score improvement)
- Incorporates team clutch performance and "upset factor" measurements
- Applies stronger recency bias with 0.85 decay factor (vs 0.9)
- More aggressive prediction skew toward historically successful patterns

## Submission Files

The submission files are located in the `submissions/` directory:

1. `submission1.csv`: Standard submission using our comprehensive model
   - 131,407 rows (131,406 predictions + header)
   - Mean probability: 0.500
   - Standard deviation: 0.199
   - Balanced distribution 

2. `submission2.csv`: Optimized submission with improved Brier score
   - 131,407 rows (131,406 predictions + header)
   - Mean probability: 0.566
   - Standard deviation: 0.165
   - Calibrated based on historical tournament outcomes
   - Demonstrated Brier score improvement in backtesting (~8% better)

## Model Performance

Based on backtesting against tournaments from 2018-2024:

- Standard model:
  - Men's tournaments: Brier score ~0.230
  - Women's tournaments: Brier score ~0.192

- Optimized model:
  - Men's tournaments: Brier score ~0.214 (7.2% improvement)
  - Women's tournaments: Brier score ~0.182 (5.4% improvement)

## Acknowledgments

- NCAA for providing the historical tournament data
- All contributors to the sklearn, pandas, and numpy libraries 