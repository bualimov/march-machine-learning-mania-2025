# March Machine Learning Mania 2025

This repository contains code for predicting the outcomes of the 2025 NCAA Men's and Women's Basketball Tournaments for the Kaggle competition [March Machine Learning Mania 2025](https://kaggle.com/competitions/march-machine-learning-mania-2025).

## Overview

The goal of this project is to predict the probability of every possible matchup between teams in the 2025 NCAA Basketball Tournaments. The model is trained on historical data from previous seasons and tournaments.

## Project Structure

- `main.py`: Main script that orchestrates the entire process
- `data_loader.py`: Module for loading data from CSV files
- `feature_engineering.py`: Module for creating features from the raw data
- `model_training.py`: Module for training and evaluating machine learning models
- `prediction_generation.py`: Module for generating predictions for the 2025 tournament

## Data

The data is organized into two main directories:
- `m_csv_files/`: Contains all men's basketball data
- `w_csv_files/`: Contains all women's basketball data

Common files:
- `Cities.csv`: Information about cities where games were played
- `Conferences.csv`: Information about conferences

## How to Run

1. Make sure you have all the required dependencies installed:
   ```
   pip install pandas numpy scikit-learn matplotlib seaborn joblib
   ```

2. Run the main script:
   ```
   python main.py
   ```

3. The script will:
   - Load all the data
   - Prepare features and training data
   - Train models for men's and women's tournaments
   - Generate predictions for all possible matchups in 2025
   - Format and save the submission file as `submission.csv`

## Model Details

The model uses a gradient boosting classifier with the following features:
- Team performance metrics (win percentage, points scored/allowed, etc.)
- Detailed statistics (field goal percentage, rebounds, assists, etc.)
- Team rankings (from Massey Ordinals)
- Conference strength
- Historical tournament performance
- Seed information (when available)

## Customization

You can customize the model by modifying the following parameters in `main.py`:
- Change the model type by modifying the `train_model` function call (options: 'logistic', 'random_forest', 'gradient_boosting')
- Adjust the seasons used for training by modifying the `start_season` and `end_season` parameters
- Add or remove features in the `create_matchup_features` function in `feature_engineering.py`

## Submission

The final submission file (`submission.csv`) contains predictions for all possible matchups between teams in the 2025 NCAA Men's and Women's Basketball Tournaments. The file format is:
- `ID`: A string in the format "2025_XXXX_YYYY", where XXXX is the lower TeamID and YYYY is the higher TeamID
- `Pred`: The probability (between 0 and 1) that the team with the lower TeamID will win

## License

This project is licensed under the MIT License - see the LICENSE file for details.
