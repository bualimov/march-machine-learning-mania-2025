#!/usr/bin/env python3
# Prediction generation module for March Machine Learning Mania 2025

import pandas as pd
import numpy as np
from feature_engineering import create_matchup_features
from data_loader import extract_seed_number
import os

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
    
    # Ensure season is treated as a numeric value for comparison
    season_numeric = float(season)
    season_int = int(season)
    
    # Get all available seasons in the team stats
    available_seasons = team_stats_df['Season'].unique()
    
    # Debug available seasons and their types
    print(f"Available seasons in team statistics: {sorted(available_seasons)}")
    print("Season types in team statistics:")
    for s in sorted(available_seasons):
        print(f"  Season {s}: {type(s)}")
    
    # Make a copy of team_stats_df and ensure Season is float type for consistent comparison
    team_stats_float = team_stats_df.copy()
    team_stats_float['Season'] = team_stats_float['Season'].astype(float)
    
    # Try to find the exact season match
    season_matches = team_stats_float[team_stats_float['Season'] == season_numeric]
    
    if len(season_matches) > 0:
        print(f"Using statistics from season {season} for predictions")
        season_stats = season_matches.copy()
    else:
        # If exact match not found, find the most recent season that's less than or equal to the target season
        valid_seasons = sorted([float(s) for s in available_seasons if not pd.isna(s) and float(s) <= season_numeric])
        
        if valid_seasons:
            most_recent_season = max(valid_seasons)
            print(f"No exact match for season {season}, using season {most_recent_season} instead")
            
            # Filter team stats for the most recent valid season (using float comparison)
            season_stats = team_stats_float[team_stats_float['Season'] == float(most_recent_season)].copy()
            
            # Update the season column to the requested season
            season_stats.loc[:, 'Season'] = season_numeric
        else:
            print(f"No valid seasons found for {season}, using the earliest available season")
            valid_seasons = sorted([float(s) for s in available_seasons if not pd.isna(s)])
            earliest_season = min(valid_seasons)
            season_stats = team_stats_float[team_stats_float['Season'] == float(earliest_season)].copy()
            season_stats.loc[:, 'Season'] = season_numeric
    
    # Count unique teams in the filtered stats
    unique_teams = season_stats['TeamID'].nunique()
    print(f"Using statistics from {unique_teams} teams for predictions")
    
    # Count how many active teams have stats
    active_teams_with_stats = np.intersect1d(active_teams, season_stats['TeamID'].unique())
    print(f"Found stats for {len(active_teams_with_stats)} out of {len(active_teams)} active teams")
    
    # If we found very few teams with stats (e.g., less than 10%), try a more flexible approach
    if len(active_teams_with_stats) < 0.1 * len(active_teams):
        print("WARNING: Very few active teams have stats. Trying more flexible matching...")
        
        # Try to use the most recent season for each team
        all_team_stats = []
        
        for team_id in active_teams:
            # Get all stats for this team
            team_stats = team_stats_float[team_stats_float['TeamID'] == team_id]
            
            if len(team_stats) > 0:
                # Get the most recent season
                most_recent = team_stats[team_stats['Season'] <= season_numeric]['Season'].max()
                if pd.isna(most_recent):
                    # If no seasons before or equal to target, use the earliest available
                    most_recent = team_stats['Season'].min()
                
                # Get stats from most recent season
                recent_stats = team_stats[team_stats['Season'] == most_recent].copy()
                recent_stats.loc[:, 'Season'] = season_numeric
                
                all_team_stats.append(recent_stats)
        
        # Combine all team stats
        if all_team_stats:
            season_stats = pd.concat(all_team_stats, ignore_index=True)
            
            # Update counts
            unique_teams = season_stats['TeamID'].nunique()
            active_teams_with_stats = np.intersect1d(active_teams, season_stats['TeamID'].unique())
            
            print(f"After flexible matching: Using statistics from {unique_teams} teams")
            print(f"After flexible matching: Found stats for {len(active_teams_with_stats)} out of {len(active_teams)} active teams")
    
    # Get tournament seeds if available
    seeds = None
    if 'tourney_seeds' in dfs:
        # Try both integer and float versions of the season
        seeds_data = dfs['tourney_seeds']
        seeds = seeds_data[(seeds_data['Season'].astype(float) == season_numeric) | 
                          (seeds_data['Season'] == season_int)].copy()
        
        if len(seeds) > 0:
            print(f"Found {len(seeds)} tournament seeds for season {season}")
            seeds['SeedNumber'] = seeds['Seed'].apply(extract_seed_number)
        else:
            print(f"No tournament seeds found for season {season}")
    
    # Generate all possible matchups
    matchups = []
    missing_stats_count = 0
    team_pairs_count = 0
    
    for i, team1_id in enumerate(active_teams):
        for team2_id in active_teams[i+1:]:  # Only consider each pair once
            team_pairs_count += 1
            
            # Get stats for both teams
            team1_stats = season_stats[season_stats['TeamID'] == team1_id]
            team2_stats = season_stats[season_stats['TeamID'] == team2_id]
            
            # Create matchup features
            if len(team1_stats) > 0 and len(team2_stats) > 0:
                # If we have stats for both teams, create features
                matchup_features = create_matchup_features(team1_stats.iloc[0], team2_stats.iloc[0])
            else:
                # If we don't have stats for one or both teams, use default features
                missing_stats_count += 1
                
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
            if seeds is not None and len(seeds) > 0:
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
            matchup_features['Season'] = season_numeric
            
            # Add team IDs
            matchup_features['Team1ID'] = team1_id
            matchup_features['Team2ID'] = team2_id
            
            matchups.append(matchup_features)
    
    print(f"Generated {team_pairs_count} team pairs with {missing_stats_count} missing team stats")
    
    # Create a DataFrame from the matchups
    matchups_df = pd.DataFrame(matchups)
    
    # Prepare predictions DataFrame
    predictions = []
    
    # Generate ID column for each matchup
    for _, row in matchups_df.iterrows():
        team1_id = int(row['Team1ID'])
        team2_id = int(row['Team2ID'])
        
        # Create prediction ID in the format "Season_Team1ID_Team2ID"
        pred_id = f"{int(season)}_{team1_id}_{team2_id}"
        
        # Make prediction
        try:
            # Prepare features for prediction
            features = row.drop(['Team1ID', 'Team2ID']).to_dict()
            
            # Convert to DataFrame
            features_df = pd.DataFrame([features])
            
            # If we have feature names, ensure we're using the same features as during training
            if feature_names is not None:
                # Check for missing features
                missing_features = [f for f in feature_names if f not in features_df.columns]
                if missing_features:
                    # Add missing features with default values
                    for feature in missing_features:
                        features_df[feature] = 0
                
                # Reorder columns to match training feature set
                features_df = features_df[feature_names]
            
            # Make prediction
            pred = model.predict_proba(features_df)[0][1]
            
            # Ensure prediction is within bounds
            pred = max(0.05, min(0.95, pred))
        except Exception as e:
            print(f"Error making prediction for {pred_id}: {e}")
            # Default to 0.5 if prediction fails
            pred = 0.5
        
        # Add prediction to list
        predictions.append({
            'ID': pred_id,
            'Pred': pred
        })
    
    # Create predictions DataFrame
    predictions_df = pd.DataFrame(predictions)
    
    # Generate predictions for the opposite matchups (team2 vs team1)
    opposite_predictions = []
    
    for _, row in predictions_df.iterrows():
        # Parse the ID to get the components
        id_parts = row['ID'].split('_')
        season_id = id_parts[0]
        team1_id = id_parts[1]
        team2_id = id_parts[2]
        
        # Create the opposite matchup ID
        opposite_id = f"{season_id}_{team2_id}_{team1_id}"
        
        # The opposite prediction is 1 - original prediction
        opposite_pred = 1 - row['Pred']
        
        # Add to list
        opposite_predictions.append({
            'ID': opposite_id,
            'Pred': opposite_pred
        })
    
    # Create DataFrame for opposite predictions
    opposite_predictions_df = pd.DataFrame(opposite_predictions)
    
    # Combine original and opposite predictions
    all_predictions = pd.concat([predictions_df, opposite_predictions_df], ignore_index=True)
    
    print(f"Generated {len(all_predictions)} predictions for {gender} basketball, season {season}")
    
    return all_predictions

def format_submission(mens_predictions, womens_predictions, sample_submission_path='SampleSubmissionStage1.csv'):
    """
    Format predictions for submission.
    
    Parameters:
    -----------
    mens_predictions : DataFrame
        Predictions for men's basketball
    womens_predictions : DataFrame
        Predictions for women's basketball
    sample_submission_path : str
        Path to the sample submission file
    
    Returns:
    --------
    DataFrame
        Formatted submission
    """
    print("Formatting predictions for submission")
    
    # Combine men's and women's predictions
    all_predictions = pd.concat([mens_predictions, womens_predictions], ignore_index=True)
    
    # Check if we have a sample submission file
    if os.path.exists(sample_submission_path):
        print(f"Using sample submission file: {sample_submission_path}")
        sample_submission = pd.read_csv(sample_submission_path)
        
        # Create a dictionary of our predictions for fast lookup
        prediction_dict = dict(zip(all_predictions['ID'], all_predictions['Pred']))
        
        # Fill in the sample submission with our predictions
        sample_submission['Pred'] = sample_submission['ID'].apply(
            lambda x: prediction_dict.get(x, 0.5)  # Default to 0.5 if we don't have a prediction
        )
        
        print(f"Filled in {len(prediction_dict)} predictions out of {len(sample_submission)} required")
        
        # Return the filled sample submission
        return sample_submission
    else:
        print(f"Warning: Sample submission file not found at {sample_submission_path}")
        print("Using generated predictions directly")
        
        # Return our predictions directly
        return all_predictions[['ID', 'Pred']] 