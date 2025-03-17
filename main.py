#!/usr/bin/env python3
# March Machine Learning Mania 2025
# Main script to orchestrate the entire process

import os
import pandas as pd
import numpy as np
from sklearn.metrics import log_loss, brier_score_loss, accuracy_score, roc_auc_score

from data_loader import load_mens_data, load_womens_data, load_common_data
from feature_engineering import prepare_team_stats, prepare_training_data
from model_training import train_model, evaluate_model, save_model, load_model
from prediction_generation import generate_predictions, format_submission

def load_data():
    """
    Load all required data files.
    
    Returns:
    --------
    dict
        Dictionary containing all loaded dataframes
    """
    print("Loading men's data...")
    mens_data = load_mens_data()
    
    print("Loading women's data...")
    womens_data = load_womens_data()
    
    print("Loading common data...")
    common_data = load_common_data()
    
    # Combine all data into a single dictionary
    dfs = {}
    dfs.update(mens_data)
    dfs.update(womens_data)
    dfs.update(common_data)
    
    print(f"Loaded {len(mens_data)} men's datasets, {len(womens_data)} women's datasets, and {len(common_data)} common datasets")
    
    return dfs

def prepare_features(dfs, gender='M', seasons=range(2010, 2025)):
    """
    Prepare features and target for model training.
    
    Parameters:
    -----------
    dfs : dict
        Dictionary containing loaded dataframes
    gender : str
        'M' for men's data, 'W' for women's data
    seasons : range or list
        Seasons to include in the training data
    
    Returns:
    --------
    tuple
        (features, target, team_stats)
    """
    print(f"Preparing features for {gender} basketball, seasons {min(seasons)}-{max(seasons)}")
    
    # Prepare team statistics
    team_stats = prepare_team_stats(dfs, gender=gender, start_season=min(seasons), end_season=max(seasons))
    
    # Prepare training data
    features, target = prepare_training_data(dfs, team_stats, gender=gender, start_season=min(seasons), end_season=max(seasons))
    
    print(f"Prepared {len(features)} training examples with {len(features.columns)} features")
    
    return features, target, team_stats

def main():
    """
    Main function to run the entire pipeline.
    """
    print("Step 1: Loading data")
    # Load data
    dfs = load_data()
    
    print("\nStep 2: Preparing features and training data")
    # Prepare features for men's basketball (for training)
    mens_features, mens_target, mens_team_stats_training = prepare_features(dfs, gender='M', seasons=range(2010, 2021))
    
    # Prepare features for women's basketball (for training)
    womens_features, womens_target, womens_team_stats_training = prepare_features(dfs, gender='W', seasons=range(2010, 2021))
    
    # Prepare team statistics for prediction seasons (2021-2024)
    print("\nPreparing team statistics for prediction seasons (2021-2024)")
    mens_team_stats_prediction = prepare_team_stats(dfs, gender='M', start_season=2021, end_season=2024)
    womens_team_stats_prediction = prepare_team_stats(dfs, gender='W', start_season=2021, end_season=2024)
    
    # Debug: Print the available seasons in each dataset
    print("\nAvailable seasons in training team stats:")
    print(sorted(mens_team_stats_training['Season'].unique()))
    
    print("\nAvailable seasons in prediction team stats:")
    print(sorted(mens_team_stats_prediction['Season'].unique()))
    
    # Combine training and prediction team statistics
    mens_team_stats = pd.concat([mens_team_stats_training, mens_team_stats_prediction], ignore_index=True)
    womens_team_stats = pd.concat([womens_team_stats_training, womens_team_stats_prediction], ignore_index=True)
    
    # Debug: Print the available seasons after concatenation
    print("\nAvailable seasons in combined team stats:")
    print(sorted(mens_team_stats['Season'].unique()))
    print("Season data types in combined team stats:")
    for season in sorted(mens_team_stats['Season'].unique()):
        print(f"  Season {season}: {type(season)}")
    
    print("\nStep 3: Training models")
    # Train or load men's model
    mens_model_path = 'models/mens_model.pkl'
    if os.path.exists(mens_model_path):
        print("Loading existing men's model")
        mens_model_info = load_model(mens_model_path)
        # Evaluate the model
        mens_metrics = evaluate_model(mens_model_info, mens_features, mens_target)
        print(f"Men's model metrics: {mens_metrics}")
    else:
        print("Training new men's model")
        mens_model_info = train_model(mens_features, mens_target)
        # Save the model
        save_model(mens_model_info, mens_model_path)
    
    # Train or load women's model
    womens_model_path = 'models/womens_model.pkl'
    if os.path.exists(womens_model_path):
        print("Loading existing women's model")
        womens_model_info = load_model(womens_model_path)
        # Evaluate the model
        womens_metrics = evaluate_model(womens_model_info, womens_features, womens_target)
        print(f"Women's model metrics: {womens_metrics}")
    else:
        print("Training new women's model")
        womens_model_info = train_model(womens_features, womens_target)
        # Save the model
        save_model(womens_model_info, womens_model_path)
    
    print("\nStep 4: Generating predictions for seasons 2021-2024")
    # Generate predictions for the specified seasons (2021-2024)
    all_mens_predictions = []
    all_womens_predictions = []
    
    for season in range(2021, 2025):  # Focus on 2021-2024 for testing
        print(f"\nGenerating predictions for season {season}")
        
        # Men's predictions
        mens_predictions = generate_predictions(mens_model_info, dfs, mens_team_stats, gender='M', season=season)
        all_mens_predictions.append(mens_predictions)
        
        # Women's predictions
        womens_predictions = generate_predictions(womens_model_info, dfs, womens_team_stats, gender='W', season=season)
        all_womens_predictions.append(womens_predictions)
    
    # Combine all predictions
    combined_mens_predictions = pd.concat(all_mens_predictions)
    combined_womens_predictions = pd.concat(all_womens_predictions)
    
    print("\nStep 5: Formatting and saving submission")
    # Format and save submission
    submission = format_submission(combined_mens_predictions, combined_womens_predictions)
    
    # Create submissions directory if it doesn't exist
    os.makedirs('submissions', exist_ok=True)
    
    # Save submission
    submission_path = 'submissions/submission_2021_2024.csv'
    submission.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
    print(f"Total predictions: {len(submission)}")
    
    # Additional analysis
    print("\nAdditional Analysis:")
    
    # Analyze seed matchups for the most recent completed tournament (2024)
    print("\nAnalyzing seed matchups for 2024 tournament:")
    mens_seed_analysis = analyze_seed_matchups(dfs, gender='M', season=2024)
    womens_seed_analysis = analyze_seed_matchups(dfs, gender='W', season=2024)

def analyze_seed_matchups(dfs, gender='M', season=2023):
    """
    Analyze the performance of teams based on their tournament seeds.
    
    Parameters:
    -----------
    dfs : dict
        Dictionary containing loaded dataframes
    gender : str
        'M' for men's data, 'W' for women's data
    season : int
        Season to analyze
    
    Returns:
    --------
    DataFrame
        Analysis of seed matchups
    """
    print(f"Analyzing seed matchups for {gender} basketball, season {season}")
    
    # Check if we have tournament results and seeds
    if 'tourney_results' not in dfs or 'tourney_seeds' not in dfs:
        print("Tournament results or seeds not available")
        return None
    
    # Get tournament results for the specified season
    # Try both integer and float versions of the season
    season_numeric = float(season)
    tourney_results = dfs['tourney_results']
    season_results = tourney_results[(tourney_results['Season'].astype(float) == season_numeric)]
    
    if len(season_results) == 0:
        print(f"No tournament results found for season {season}")
        return None
    
    # Get tournament seeds for the specified season
    seeds = dfs['tourney_seeds']
    season_seeds = seeds[(seeds['Season'].astype(float) == season_numeric)]
    
    if len(season_seeds) == 0:
        print(f"No tournament seeds found for season {season}")
        return None
    
    # Extract seed numbers
    season_seeds['SeedNumber'] = season_seeds['Seed'].apply(lambda x: int(''.join(filter(str.isdigit, x))))
    
    # Create a mapping from team ID to seed
    team_to_seed = dict(zip(season_seeds['TeamID'], season_seeds['SeedNumber']))
    
    # Add seed information to tournament results
    season_results['WTeamSeed'] = season_results['WTeamID'].map(team_to_seed)
    season_results['LTeamSeed'] = season_results['LTeamID'].map(team_to_seed)
    
    # Create seed matchup column
    season_results['SeedMatchup'] = season_results.apply(
        lambda row: f"{row['WTeamSeed']} vs {row['LTeamSeed']}", axis=1
    )
    
    # Count occurrences of each seed matchup
    seed_matchup_counts = season_results['SeedMatchup'].value_counts().reset_index()
    seed_matchup_counts.columns = ['SeedMatchup', 'Count']
    
    # Calculate upset rate (lower seed beating higher seed)
    season_results['IsUpset'] = season_results['WTeamSeed'] > season_results['LTeamSeed']
    upset_rate = season_results['IsUpset'].mean()
    
    print(f"Upset rate: {upset_rate:.2f}")
    
    # Group by winning seed and count
    winning_seed_counts = season_results['WTeamSeed'].value_counts().sort_index().reset_index()
    winning_seed_counts.columns = ['Seed', 'Wins']
    
    return seed_matchup_counts

if __name__ == "__main__":
    main() 