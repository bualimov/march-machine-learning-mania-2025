#!/usr/bin/env python3
# March Machine Learning Mania 2025
# Main script to orchestrate the entire process

import os
import pandas as pd
import numpy as np
from data_loader import load_mens_data, load_womens_data, load_common_data
from feature_engineering import prepare_team_stats, prepare_training_data
from model_training import train_model, evaluate_model, load_model
from prediction_generation import generate_predictions, format_submission

def main():
    print("Starting March Machine Learning Mania 2025 prediction process...")
    
    # Create directories for models and submissions
    os.makedirs('models', exist_ok=True)
    os.makedirs('submissions', exist_ok=True)
    
    # Step 1: Load data
    print("\nStep 1: Loading data...")
    mens_data = load_mens_data()
    womens_data = load_womens_data()
    common_data = load_common_data()
    
    # Step 2: Prepare features and training data
    print("\nStep 2: Preparing features and training data...")
    
    # Men's data - focus on more recent seasons (2010-2024) as per notes
    print("Processing men's data...")
    mens_team_stats = prepare_team_stats(mens_data, gender='M', start_season=2010, end_season=2024)
    mens_X, mens_y = prepare_training_data(mens_data, mens_team_stats, gender='M', start_season=2010, end_season=2024)
    
    # Women's data - focus on more recent seasons (2010-2024) as per notes
    print("Processing women's data...")
    womens_team_stats = prepare_team_stats(womens_data, gender='W', start_season=2010, end_season=2024)
    womens_X, womens_y = prepare_training_data(womens_data, womens_team_stats, gender='W', start_season=2010, end_season=2024)
    
    # Step 3: Train models with time-based weighting
    print("\nStep 3: Training models...")
    
    # Check if models already exist
    mens_model_path = 'models/mens_model.pkl'
    womens_model_path = 'models/womens_model.pkl'
    
    # Men's model
    if os.path.exists(mens_model_path):
        print(f"Loading existing men's model from {mens_model_path}")
        mens_model = load_model(mens_model_path)
    else:
        print("Training men's model...")
        try:
            # Apply time-based weighting to give more importance to recent seasons
            mens_model = train_model(mens_X, mens_y, model_type='gradient_boosting', 
                                    save_model=True, model_dir='models', time_weight=True)
            # Save with a specific name
            os.rename('models/gradient_boosting_model.pkl', mens_model_path)
        except Exception as e:
            print(f"Error training men's model: {e}")
            print("Falling back to logistic regression...")
            mens_model = train_model(mens_X, mens_y, model_type='logistic', 
                                    save_model=True, model_dir='models', time_weight=True)
            # Save with a specific name
            os.rename('models/logistic_model.pkl', mens_model_path)
    
    # Evaluate men's model
    print("Evaluating men's model...")
    mens_score = evaluate_model(mens_model, mens_X, mens_y)
    print(f"Men's model evaluation score: {mens_score:.4f}")
    
    # Women's model
    if os.path.exists(womens_model_path):
        print(f"Loading existing women's model from {womens_model_path}")
        womens_model = load_model(womens_model_path)
    else:
        print("Training women's model...")
        try:
            # Apply time-based weighting to give more importance to recent seasons
            womens_model = train_model(womens_X, womens_y, model_type='gradient_boosting', 
                                      save_model=True, model_dir='models', time_weight=True)
            # Save with a specific name
            os.rename('models/gradient_boosting_model.pkl', womens_model_path)
        except Exception as e:
            print(f"Error training women's model: {e}")
            print("Falling back to logistic regression...")
            womens_model = train_model(womens_X, womens_y, model_type='logistic', 
                                      save_model=True, model_dir='models', time_weight=True)
            # Save with a specific name
            os.rename('models/logistic_model.pkl', womens_model_path)
    
    # Evaluate women's model
    print("Evaluating women's model...")
    womens_score = evaluate_model(womens_model, womens_X, womens_y)
    print(f"Women's model evaluation score: {womens_score:.4f}")
    
    # Step 4: Generate predictions for all required seasons
    print("\nStep 4: Generating predictions for all required seasons...")
    
    # Seasons to generate predictions for (based on sample submission)
    seasons = [2021, 2022, 2023, 2024, 2025]
    
    all_mens_predictions = []
    all_womens_predictions = []
    
    for season in seasons:
        print(f"\nGenerating predictions for season {season}...")
        
        # Men's predictions
        print(f"Generating men's predictions for {season}...")
        mens_predictions = generate_predictions(
            mens_model, 
            mens_data, 
            mens_team_stats,
            gender='M',
            season=season
        )
        all_mens_predictions.append(mens_predictions)
        
        # Women's predictions
        print(f"Generating women's predictions for {season}...")
        womens_predictions = generate_predictions(
            womens_model, 
            womens_data, 
            womens_team_stats,
            gender='W',
            season=season
        )
        all_womens_predictions.append(womens_predictions)
    
    # Combine all predictions
    combined_mens_predictions = pd.concat(all_mens_predictions)
    combined_womens_predictions = pd.concat(all_womens_predictions)
    
    # Step 5: Format and save submission
    print("\nStep 5: Formatting and saving submission...")
    
    # Use the sample submission file as a template to ensure correct format
    submission = format_submission(
        combined_mens_predictions, 
        combined_womens_predictions,
        sample_submission_path='SampleSubmissionStage1.csv'
    )
    
    # Save submission
    submission_path = 'submissions/submission.csv'
    submission.to_csv(submission_path, index=False)
    
    print(f"\nPrediction process complete! Submission file saved as '{submission_path}'")
    print(f"Total predictions: {len(submission)}")
    
    # Additional analysis based on notes
    print("\nAdditional Analysis:")
    
    # Check how model performs for different seed matchups
    if 'tourney_seeds' in mens_data and 'tourney_results' in mens_data:
        print("\nAnalyzing model performance by seed matchup (Men's):")
        analyze_seed_matchups(mens_model, mens_data, mens_team_stats)
    
    if 'tourney_seeds' in womens_data and 'tourney_results' in womens_data:
        print("\nAnalyzing model performance by seed matchup (Women's):")
        analyze_seed_matchups(womens_model, womens_data, womens_team_stats)

def analyze_seed_matchups(model, dfs, team_stats_df, season=2024):
    """
    Analyze model performance for different seed matchups.
    
    Parameters:
    -----------
    model : dict or object
        Trained model information
    dfs : dict
        Dictionary containing loaded dataframes
    team_stats_df : DataFrame
        Team statistics DataFrame
    season : int
        Season to analyze
    """
    # Get tournament results for the specified season
    tourney_results = dfs['tourney_results'][dfs['tourney_results']['Season'] == season].copy()
    
    if len(tourney_results) == 0:
        print(f"No tournament results found for season {season}")
        return
    
    # Get tournament seeds for the specified season
    tourney_seeds = dfs['tourney_seeds'][dfs['tourney_seeds']['Season'] == season].copy()
    tourney_seeds['SeedNumber'] = tourney_seeds['Seed'].apply(lambda x: int(x[1:3]) if isinstance(x, str) and len(x) >= 3 else None)
    
    # Create a dictionary to map TeamID to seed
    team_to_seed = dict(zip(tourney_seeds['TeamID'], tourney_seeds['SeedNumber']))
    
    # Add seed information to tournament results
    tourney_results['WTeamSeed'] = tourney_results['WTeamID'].map(team_to_seed)
    tourney_results['LTeamSeed'] = tourney_results['LTeamID'].map(team_to_seed)
    
    # Group matchups by seed difference
    seed_diff_results = []
    
    for _, game in tourney_results.iterrows():
        if pd.isna(game['WTeamSeed']) or pd.isna(game['LTeamSeed']):
            continue
        
        # Calculate seed difference (lower seed - higher seed)
        seed_diff = game['LTeamSeed'] - game['WTeamSeed']
        
        # Generate prediction for this matchup
        team1_stats = team_stats_df[(team_stats_df['Season'] == season) & (team_stats_df['TeamID'] == game['WTeamID'])]
        team2_stats = team_stats_df[(team_stats_df['Season'] == season) & (team_stats_df['TeamID'] == game['LTeamID'])]
        
        if len(team1_stats) > 0 and len(team2_stats) > 0:
            # Create a single matchup for prediction
            matchup_df = pd.DataFrame([{
                'Team1ID': game['WTeamID'],
                'Team2ID': game['LTeamID'],
                'Team1Seed': game['WTeamSeed'],
                'Team2Seed': game['LTeamSeed'],
                'SeedDiff': game['WTeamSeed'] - game['LTeamSeed'],
                'Season': season
            }])
            
            # Generate prediction
            try:
                pred = generate_predictions(model, dfs, team_stats_df, season=season)
                pred_value = pred['Pred'].iloc[0] if len(pred) > 0 else 0.5
            except:
                pred_value = 0.5
            
            seed_diff_results.append({
                'SeedDiff': seed_diff,
                'WTeamSeed': game['WTeamSeed'],
                'LTeamSeed': game['LTeamSeed'],
                'ActualResult': 1,  # WTeam won
                'PredictedProb': pred_value
            })
    
    # Group by seed difference and calculate accuracy
    if seed_diff_results:
        seed_diff_df = pd.DataFrame(seed_diff_results)
        seed_diff_groups = seed_diff_df.groupby('SeedDiff')
        
        print(f"Seed difference analysis for {season} tournament:")
        for seed_diff, group in seed_diff_groups:
            accuracy = np.mean((group['PredictedProb'] > 0.5) == group['ActualResult'])
            avg_prob = group['PredictedProb'].mean()
            count = len(group)
            print(f"  Seed diff {seed_diff}: {count} games, Accuracy: {accuracy:.2f}, Avg Prob: {avg_prob:.2f}")
    else:
        print("No seed matchup data available for analysis")

if __name__ == "__main__":
    main() 