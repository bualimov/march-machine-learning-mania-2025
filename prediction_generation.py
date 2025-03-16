#!/usr/bin/env python3
# Prediction generation module for March Machine Learning Mania 2025

import pandas as pd
import numpy as np
from feature_engineering import create_matchup_features
from data_loader import extract_seed_number

def generate_predictions(model_info, dfs, team_stats_df, gender='M', season=2025):
    """
    Generate predictions for all possible matchups in the specified season.
    
    Parameters:
    -----------
    model_info : dict or object
        Trained model information (dict with 'model' and 'feature_names' keys) or just the model
    dfs : dict
        Dictionary containing loaded dataframes
    team_stats_df : DataFrame
        Team statistics DataFrame
    gender : str
        'M' for men's data, 'W' for women's data
    season : int
        Season to generate predictions for
    
    Returns:
    --------
    DataFrame
        Predictions for all possible matchups
    """
    print(f"Generating predictions for {gender} basketball, season {season}")
    
    # Extract model and feature names
    if isinstance(model_info, dict) and 'model' in model_info:
        model = model_info['model']
        feature_names = model_info.get('feature_names', None)
    else:
        model = model_info
        feature_names = None
    
    # Get all teams for the specified gender
    teams_df = dfs['teams'].copy()
    
    # Check if we can filter by active seasons
    if 'FirstD1Season' in teams_df.columns and 'LastD1Season' in teams_df.columns:
        # Filter teams that were active in the specified season or later
        active_teams = teams_df[
            (teams_df['FirstD1Season'] <= season) & 
            ((teams_df['LastD1Season'] >= season) | (teams_df['LastD1Season'] == 2025))
        ]['TeamID'].unique()
    else:
        # If we don't have season information, use all teams
        active_teams = teams_df['TeamID'].unique()
    
    print(f"Found {len(active_teams)} active teams for {gender} basketball in season {season}")
    
    # Filter team stats for the specified season
    season_stats = team_stats_df[team_stats_df['Season'] == season]
    
    # If no stats for the specified season, use the most recent season
    if len(season_stats) == 0:
        most_recent_season = team_stats_df['Season'].max()
        print(f"No stats found for season {season}, using season {most_recent_season} instead")
        season_stats = team_stats_df[team_stats_df['Season'] == most_recent_season].copy()
        # Update the season column to the requested season
        season_stats.loc[:, 'Season'] = season
    
    # Get tournament seeds if available
    seeds = None
    if 'tourney_seeds' in dfs and season in dfs['tourney_seeds']['Season'].unique():
        seeds = dfs['tourney_seeds'][dfs['tourney_seeds']['Season'] == season].copy()
        seeds['SeedNumber'] = seeds['Seed'].apply(extract_seed_number)
    
    # Generate all possible matchups
    matchups = []
    
    for i, team1_id in enumerate(active_teams):
        for team2_id in active_teams[i+1:]:  # Only consider each pair once
            # Get stats for both teams
            team1_stats = season_stats[season_stats['TeamID'] == team1_id]
            team2_stats = season_stats[season_stats['TeamID'] == team2_id]
            
            # Create matchup features
            if len(team1_stats) > 0 and len(team2_stats) > 0:
                # If we have stats for both teams, create features
                matchup_features = create_matchup_features(team1_stats.iloc[0], team2_stats.iloc[0])
            else:
                # If we don't have stats for one or both teams, use default features
                matchup_features = {}
                
                # Add basic default features
                for stat in ['WinPercentageDiff', 'AvgPointsScoredDiff', 'AvgPointsAllowedDiff', 'AvgPointDifferentialDiff']:
                    matchup_features[stat] = 0
                
                # Add conference strength default features
                for stat in ['ConfAvgWinPercentageDiff', 'ConfAvgPointDifferentialDiff']:
                    matchup_features[stat] = 0
                
                # Add historical tournament performance default features
                for stat in ['TourneyAppearancesDiff', 'TourneyWinsDiff', 'TourneyWinPercentageDiff']:
                    matchup_features[stat] = 0
                
                # Add detailed stats default features
                for stat in ['FGPercentageDiff', 'FG3PercentageDiff', 'FTPercentageDiff', 'AvgReboundsDiff', 
                            'AvgAstDiff', 'AvgTODiff', 'AvgStlDiff', 'AvgBlkDiff']:
                    matchup_features[stat] = 0
                
                # Add opponent stats default features
                for stat in ['OpponentFGPercentageDiff', 'OpponentFG3PercentageDiff', 'OpponentFTPercentageDiff', 
                            'AvgOpponentReboundsDiff']:
                    matchup_features[stat] = 0
                
                # Add ranking default feature
                matchup_features['RankDiff'] = 0
            
            # Add seed information if available
            if seeds is not None:
                team1_seed = seeds[seeds['TeamID'] == team1_id]['SeedNumber'].values
                team2_seed = seeds[seeds['TeamID'] == team2_id]['SeedNumber'].values
                
                if len(team1_seed) > 0 and len(team2_seed) > 0:
                    matchup_features['Team1Seed'] = team1_seed[0]
                    matchup_features['Team2Seed'] = team2_seed[0]
                    matchup_features['SeedDiff'] = team1_seed[0] - team2_seed[0]
                else:
                    # If seeds are not available, use default values
                    matchup_features['Team1Seed'] = np.nan
                    matchup_features['Team2Seed'] = np.nan
                    matchup_features['SeedDiff'] = np.nan
            else:
                # If seeds are not available, use default values
                matchup_features['Team1Seed'] = np.nan
                matchup_features['Team2Seed'] = np.nan
                matchup_features['SeedDiff'] = np.nan
            
            # Add season
            matchup_features['Season'] = season
            
            # Add team IDs
            matchup_features['Team1ID'] = team1_id
            matchup_features['Team2ID'] = team2_id
            
            matchups.append(matchup_features)
    
    # Convert to DataFrame
    X_pred = pd.DataFrame(matchups)
    
    # Check for NaN values
    if X_pred.isna().any().any():
        print(f"Warning: Prediction data contains {X_pred.isna().sum().sum()} NaN values")
        print("Columns with NaN values:")
        print(X_pred.isna().sum()[X_pred.isna().sum() > 0])
    
    # Remove non-numeric columns for prediction
    non_numeric_cols = ['Team1ID', 'Team2ID']
    X_pred_numeric = X_pred.drop(columns=[col for col in non_numeric_cols if col in X_pred.columns])
    
    # Ensure feature order matches training if feature names are available
    if feature_names is not None:
        print(f"Using {len(feature_names)} features from the model")
        
        # Add missing columns with zeros
        for col in feature_names:
            if col not in X_pred_numeric.columns:
                X_pred_numeric[col] = 0
        
        # Reorder columns to match training
        X_pred_numeric = X_pred_numeric[feature_names]
    
    # Make predictions
    try:
        y_pred_proba = model.predict_proba(X_pred_numeric)[:, 1]
    except Exception as e:
        print(f"Error making predictions: {e}")
        print("Falling back to default predictions (0.5)")
        y_pred_proba = np.ones(len(X_pred)) * 0.5
    
    # Add predictions to the matchups DataFrame
    X_pred['Pred'] = y_pred_proba
    
    # Create submission format
    predictions = []
    
    for _, row in X_pred.iterrows():
        team1_id = int(row['Team1ID'])
        team2_id = int(row['Team2ID'])
        
        # Ensure team1_id < team2_id for submission format
        if team1_id < team2_id:
            id_str = f"{season}_{team1_id}_{team2_id}"
            pred = row['Pred']
        else:
            id_str = f"{season}_{team2_id}_{team1_id}"
            pred = 1 - row['Pred']  # Flip the prediction
        
        predictions.append({
            'ID': id_str,
            'Pred': pred
        })
    
    # Convert to DataFrame
    predictions_df = pd.DataFrame(predictions)
    
    print(f"Generated {len(predictions_df)} predictions for season {season}")
    
    return predictions_df

def format_submission(mens_predictions, womens_predictions, sample_submission_path='SampleSubmissionStage1.csv'):
    """
    Format predictions for submission, ensuring it matches the expected format.
    
    Parameters:
    -----------
    mens_predictions : DataFrame
        Predictions for men's tournament
    womens_predictions : DataFrame
        Predictions for women's tournament
    sample_submission_path : str
        Path to the sample submission file
    
    Returns:
    --------
    DataFrame
        Formatted submission
    """
    # Combine men's and women's predictions
    all_predictions = pd.concat([mens_predictions, womens_predictions])
    
    # Remove any duplicates
    all_predictions = all_predictions.drop_duplicates(subset=['ID'])
    
    # Sort by ID
    all_predictions = all_predictions.sort_values('ID')
    
    # Ensure predictions are between 0 and 1
    all_predictions['Pred'] = all_predictions['Pred'].clip(0.001, 0.999)
    
    # Check if we need to match the sample submission format
    try:
        sample_submission = pd.read_csv(sample_submission_path)
        print(f"Sample submission contains {len(sample_submission)} rows")
        
        # Create a dictionary of our predictions for fast lookup
        pred_dict = dict(zip(all_predictions['ID'], all_predictions['Pred']))
        
        # Create a new DataFrame with the same IDs as the sample submission
        final_predictions = []
        
        for id_str in sample_submission['ID']:
            if id_str in pred_dict:
                pred = pred_dict[id_str]
            else:
                # If we don't have a prediction for this ID, use 0.5
                pred = 0.5
                print(f"Warning: No prediction for {id_str}, using default (0.5)")
            
            final_predictions.append({
                'ID': id_str,
                'Pred': pred
            })
        
        final_df = pd.DataFrame(final_predictions)
        print(f"Final submission contains {len(final_df)} predictions (matched to sample submission)")
        
        return final_df
        
    except Exception as e:
        print(f"Error matching sample submission format: {e}")
        print("Using our generated predictions instead")
        
        print(f"Final submission contains {len(all_predictions)} predictions")
        return all_predictions 