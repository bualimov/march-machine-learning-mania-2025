#!/usr/bin/env python3
# March Machine Learning Mania 2025
# Prediction model for NCAA Basketball Tournament outcomes

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import os
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Define paths to data directories
M_DATA_DIR = 'm_csv_files'
W_DATA_DIR = 'w_csv_files'

def load_data(gender='M'):
    """
    Load all necessary data files for the specified gender.
    
    Parameters:
    -----------
    gender : str
        'M' for men's data, 'W' for women's data
    
    Returns:
    --------
    dict
        Dictionary containing all loaded dataframes
    """
    data_dir = M_DATA_DIR if gender == 'M' else W_DATA_DIR
    prefix = gender
    
    # Dictionary to store all dataframes
    dfs = {}
    
    # Load teams data
    teams_file = f'{data_dir}/{prefix}Teams.csv'
    dfs['teams'] = pd.read_csv(teams_file)
    
    # Load regular season results
    reg_season_file = f'{data_dir}/{prefix}RegularSeasonCompactResults.csv'
    dfs['regular_season'] = pd.read_csv(reg_season_file)
    
    # Load detailed regular season results (available from 2003 for men, 2010 for women)
    reg_season_detailed_file = f'{data_dir}/{prefix}RegularSeasonDetailedResults.csv'
    dfs['regular_season_detailed'] = pd.read_csv(reg_season_detailed_file)
    
    # Load tournament seeds
    tourney_seeds_file = f'{data_dir}/{prefix}NCAATourneySeeds.csv'
    dfs['tourney_seeds'] = pd.read_csv(tourney_seeds_file)
    
    # Load tournament results
    tourney_results_file = f'{data_dir}/{prefix}NCAATourneyCompactResults.csv'
    dfs['tourney_results'] = pd.read_csv(tourney_results_file)
    
    # Load detailed tournament results
    tourney_detailed_file = f'{data_dir}/{prefix}NCAATourneyDetailedResults.csv'
    dfs['tourney_detailed'] = pd.read_csv(tourney_detailed_file)
    
    # Load team conferences
    team_conferences_file = f'{data_dir}/{prefix}TeamConferences.csv'
    dfs['team_conferences'] = pd.read_csv(team_conferences_file)
    
    # Load seasons data
    seasons_file = f'{data_dir}/{prefix}Seasons.csv'
    dfs['seasons'] = pd.read_csv(seasons_file)
    
    # Load Massey Ordinals (rankings) if it's men's data
    if gender == 'M':
        massey_file = f'{data_dir}/MMasseyOrdinals.csv'
        if os.path.exists(massey_file):
            dfs['massey_ordinals'] = pd.read_csv(massey_file)
    
    # Load conference tourney games
    conf_tourney_file = f'{data_dir}/{prefix}ConferenceTourneyGames.csv'
    dfs['conf_tourney_games'] = pd.read_csv(conf_tourney_file)
    
    print(f"Loaded {len(dfs)} datasets for {gender} basketball")
    return dfs

# Load common data
def load_common_data():
    """Load data files that are common to both men's and women's tournaments"""
    common_dfs = {}
    
    # Load cities data
    common_dfs['cities'] = pd.read_csv('Cities.csv')
    
    # Load conferences data
    common_dfs['conferences'] = pd.read_csv('Conferences.csv')
    
    print(f"Loaded {len(common_dfs)} common datasets")
    return common_dfs

# Function to load sample submission file
def load_submission_template():
    """Load the sample submission file to understand the required format"""
    return pd.read_csv('SampleSubmissionStage1.csv')

def extract_seed_number(seed):
    """Extract the numeric part of the seed (e.g., 'W01' -> 1)"""
    # The seed format is like 'W01', 'X02', etc.
    # The second and third characters represent the seed number
    return int(seed[1:3])

def prepare_tournament_data(dfs, gender='M'):
    """
    Prepare tournament data for model training.
    
    Parameters:
    -----------
    dfs : dict
        Dictionary containing loaded dataframes
    gender : str
        'M' for men's data, 'W' for women's data
    
    Returns:
    --------
    DataFrame
        Processed tournament data ready for feature engineering
    """
    # Get tournament results
    tourney_results = dfs['tourney_results'].copy()
    
    # Add a column to indicate which team won (1 for WTeamID, 0 for LTeamID)
    tourney_results['WTeamWon'] = 1
    
    # Create a DataFrame with the same structure but flipped teams
    # This gives us examples where the "first" team lost
    tourney_results_flipped = tourney_results.copy()
    tourney_results_flipped['WTeamID'], tourney_results_flipped['LTeamID'] = tourney_results_flipped['LTeamID'], tourney_results_flipped['WTeamID']
    tourney_results_flipped['WScore'], tourney_results_flipped['LScore'] = tourney_results_flipped['LScore'], tourney_results_flipped['WScore']
    tourney_results_flipped['WTeamWon'] = 0
    
    # Combine the original and flipped DataFrames
    all_results = pd.concat([tourney_results, tourney_results_flipped])
    
    # Rename columns to make it clearer
    all_results = all_results.rename(columns={
        'WTeamID': 'Team1ID',
        'LTeamID': 'Team2ID',
        'WScore': 'Team1Score',
        'LScore': 'Team2Score',
        'WTeamWon': 'Team1Won'
    })
    
    # Add seed information
    seeds = dfs['tourney_seeds'].copy()
    seeds['SeedNumber'] = seeds['Seed'].apply(extract_seed_number)
    
    # Merge seed information for both teams
    all_results = pd.merge(
        all_results,
        seeds[['Season', 'TeamID', 'SeedNumber']],
        left_on=['Season', 'Team1ID'],
        right_on=['Season', 'TeamID'],
        how='left'
    )
    all_results = all_results.rename(columns={'SeedNumber': 'Team1Seed'})
    all_results = all_results.drop('TeamID', axis=1)
    
    all_results = pd.merge(
        all_results,
        seeds[['Season', 'TeamID', 'SeedNumber']],
        left_on=['Season', 'Team2ID'],
        right_on=['Season', 'TeamID'],
        how='left'
    )
    all_results = all_results.rename(columns={'SeedNumber': 'Team2Seed'})
    all_results = all_results.drop('TeamID', axis=1)
    
    # Calculate seed difference
    all_results['SeedDiff'] = all_results['Team1Seed'] - all_results['Team2Seed']
    
    return all_results

def calculate_team_stats(dfs, gender='M', season=None):
    """
    Calculate team statistics for a given season.
    
    Parameters:
    -----------
    dfs : dict
        Dictionary containing loaded dataframes
    gender : str
        'M' for men's data, 'W' for women's data
    season : int or None
        Season to calculate stats for. If None, calculate for all seasons.
    
    Returns:
    --------
    DataFrame
        Team statistics for the specified season(s)
    """
    # Get regular season results
    reg_season = dfs['regular_season'].copy()
    
    if season is not None:
        reg_season = reg_season[reg_season['Season'] == season]
    
    # Initialize a dictionary to store team stats
    team_stats = {}
    
    # Get unique seasons and teams
    seasons = reg_season['Season'].unique()
    teams = dfs['teams']['TeamID'].unique()
    
    for s in seasons:
        season_games = reg_season[reg_season['Season'] == s]
        
        for team in teams:
            # Games where the team won
            won_games = season_games[season_games['WTeamID'] == team]
            # Games where the team lost
            lost_games = season_games[season_games['LTeamID'] == team]
            
            # Skip if team didn't play in this season
            if len(won_games) == 0 and len(lost_games) == 0:
                continue
            
            # Calculate basic stats
            wins = len(won_games)
            losses = len(lost_games)
            total_games = wins + losses
            win_percentage = wins / total_games if total_games > 0 else 0
            
            # Calculate points scored and allowed
            points_scored = won_games['WScore'].sum() + lost_games['LScore'].sum()
            points_allowed = won_games['LScore'].sum() + lost_games['WScore'].sum()
            
            # Calculate average points per game
            avg_points_scored = points_scored / total_games if total_games > 0 else 0
            avg_points_allowed = points_allowed / total_games if total_games > 0 else 0
            
            # Calculate point differential
            point_diff = points_scored - points_allowed
            avg_point_diff = point_diff / total_games if total_games > 0 else 0
            
            # Store stats in dictionary
            team_stats[(s, team)] = {
                'Season': s,
                'TeamID': team,
                'Wins': wins,
                'Losses': losses,
                'WinPercentage': win_percentage,
                'PointsScored': points_scored,
                'PointsAllowed': points_allowed,
                'AvgPointsScored': avg_points_scored,
                'AvgPointsAllowed': avg_points_allowed,
                'PointDifferential': point_diff,
                'AvgPointDifferential': avg_point_diff,
                'TotalGames': total_games
            }
    
    # Convert dictionary to DataFrame
    team_stats_df = pd.DataFrame.from_dict(team_stats, orient='index')
    team_stats_df = team_stats_df.reset_index(drop=True)
    
    return team_stats_df

def calculate_detailed_stats(dfs, gender='M', season=None):
    """
    Calculate detailed team statistics for a given season.
    
    Parameters:
    -----------
    dfs : dict
        Dictionary containing loaded dataframes
    gender : str
        'M' for men's data, 'W' for women's data
    season : int or None
        Season to calculate stats for. If None, calculate for all seasons.
    
    Returns:
    --------
    DataFrame
        Detailed team statistics for the specified season(s)
    """
    # Get detailed regular season results
    detailed_results = dfs['regular_season_detailed'].copy()
    
    if season is not None:
        detailed_results = detailed_results[detailed_results['Season'] == season]
    
    # Initialize a dictionary to store team stats
    detailed_stats = {}
    
    # Get unique seasons and teams
    seasons = detailed_results['Season'].unique()
    teams = dfs['teams']['TeamID'].unique()
    
    for s in seasons:
        season_games = detailed_results[detailed_results['Season'] == s]
        
        for team in teams:
            # Games where the team won
            won_games = season_games[season_games['WTeamID'] == team]
            # Games where the team lost
            lost_games = season_games[season_games['LTeamID'] == team]
            
            # Skip if team didn't play in this season
            if len(won_games) == 0 and len(lost_games) == 0:
                continue
            
            # Calculate total games
            total_games = len(won_games) + len(lost_games)
            
            # Initialize stats dictionary
            stats = {
                'Season': s,
                'TeamID': team,
                'TotalGames': total_games
            }
            
            # Calculate offensive stats from games won
            if len(won_games) > 0:
                for stat in ['FGM', 'FGA', 'FGM3', 'FGA3', 'FTM', 'FTA', 'OR', 'DR', 'Ast', 'TO', 'Stl', 'Blk', 'PF']:
                    stats[f'W{stat}Sum'] = won_games[f'W{stat}'].sum()
                    stats[f'W{stat}Avg'] = won_games[f'W{stat}'].mean()
            
            # Calculate defensive stats from games won
            if len(won_games) > 0:
                for stat in ['FGM', 'FGA', 'FGM3', 'FGA3', 'FTM', 'FTA', 'OR', 'DR', 'Ast', 'TO', 'Stl', 'Blk', 'PF']:
                    stats[f'OpponentW{stat}Sum'] = won_games[f'L{stat}'].sum()
                    stats[f'OpponentW{stat}Avg'] = won_games[f'L{stat}'].mean()
            
            # Calculate offensive stats from games lost
            if len(lost_games) > 0:
                for stat in ['FGM', 'FGA', 'FGM3', 'FGA3', 'FTM', 'FTA', 'OR', 'DR', 'Ast', 'TO', 'Stl', 'Blk', 'PF']:
                    stats[f'L{stat}Sum'] = lost_games[f'L{stat}'].sum()
                    stats[f'L{stat}Avg'] = lost_games[f'L{stat}'].mean()
            
            # Calculate defensive stats from games lost
            if len(lost_games) > 0:
                for stat in ['FGM', 'FGA', 'FGM3', 'FGA3', 'FTM', 'FTA', 'OR', 'DR', 'Ast', 'TO', 'Stl', 'Blk', 'PF']:
                    stats[f'OpponentL{stat}Sum'] = lost_games[f'W{stat}'].sum()
                    stats[f'OpponentL{stat}Avg'] = lost_games[f'W{stat}'].mean()
            
            # Calculate overall offensive stats
            for stat in ['FGM', 'FGA', 'FGM3', 'FGA3', 'FTM', 'FTA', 'OR', 'DR', 'Ast', 'TO', 'Stl', 'Blk', 'PF']:
                w_sum = stats.get(f'W{stat}Sum', 0)
                l_sum = stats.get(f'L{stat}Sum', 0)
                stats[f'Total{stat}'] = w_sum + l_sum
                stats[f'Avg{stat}'] = (w_sum + l_sum) / total_games if total_games > 0 else 0
            
            # Calculate overall defensive stats
            for stat in ['FGM', 'FGA', 'FGM3', 'FGA3', 'FTM', 'FTA', 'OR', 'DR', 'Ast', 'TO', 'Stl', 'Blk', 'PF']:
                w_sum = stats.get(f'OpponentW{stat}Sum', 0)
                l_sum = stats.get(f'OpponentL{stat}Sum', 0)
                stats[f'TotalOpponent{stat}'] = w_sum + l_sum
                stats[f'AvgOpponent{stat}'] = (w_sum + l_sum) / total_games if total_games > 0 else 0
            
            # Calculate shooting percentages
            stats['FGPercentage'] = stats['TotalFGM'] / stats['TotalFGA'] if stats.get('TotalFGA', 0) > 0 else 0
            stats['FG3Percentage'] = stats['TotalFGM3'] / stats['TotalFGA3'] if stats.get('TotalFGA3', 0) > 0 else 0
            stats['FTPercentage'] = stats['TotalFTM'] / stats['TotalFTA'] if stats.get('TotalFTA', 0) > 0 else 0
            
            stats['OpponentFGPercentage'] = stats['TotalOpponentFGM'] / stats['TotalOpponentFGA'] if stats.get('TotalOpponentFGA', 0) > 0 else 0
            stats['OpponentFG3Percentage'] = stats['TotalOpponentFGM3'] / stats['TotalOpponentFGA3'] if stats.get('TotalOpponentFGA3', 0) > 0 else 0
            stats['OpponentFTPercentage'] = stats['TotalOpponentFTM'] / stats['TotalOpponentFTA'] if stats.get('TotalOpponentFTA', 0) > 0 else 0
            
            # Calculate rebounding stats
            stats['TotalRebounds'] = stats.get('TotalOR', 0) + stats.get('TotalDR', 0)
            stats['AvgRebounds'] = stats['TotalRebounds'] / total_games if total_games > 0 else 0
            
            stats['TotalOpponentRebounds'] = stats.get('TotalOpponentOR', 0) + stats.get('TotalOpponentDR', 0)
            stats['AvgOpponentRebounds'] = stats['TotalOpponentRebounds'] / total_games if total_games > 0 else 0
            
            # Store stats in dictionary
            detailed_stats[(s, team)] = stats
    
    # Convert dictionary to DataFrame
    detailed_stats_df = pd.DataFrame.from_dict(detailed_stats, orient='index')
    detailed_stats_df = detailed_stats_df.reset_index(drop=True)
    
    return detailed_stats_df

def get_team_rankings(dfs, season, day_num=133):
    """
    Get team rankings from Massey Ordinals for a specific season and day.
    
    Parameters:
    -----------
    dfs : dict
        Dictionary containing loaded dataframes
    season : int
        Season to get rankings for
    day_num : int
        Day number to get rankings for (default is 133, which is right before the tournament)
    
    Returns:
    --------
    DataFrame
        Team rankings for the specified season and day
    """
    if 'massey_ordinals' not in dfs:
        return None
    
    massey = dfs['massey_ordinals'].copy()
    
    # Filter for the specified season and day
    season_massey = massey[(massey['Season'] == season) & (massey['RankingDayNum'] == day_num)]
    
    # If no data for the specified day, get the closest day
    if len(season_massey) == 0:
        available_days = massey[massey['Season'] == season]['RankingDayNum'].unique()
        if len(available_days) == 0:
            return None
        
        # Get the closest day to the specified day
        closest_day = available_days[np.abs(available_days - day_num).argmin()]
        season_massey = massey[(massey['Season'] == season) & (massey['RankingDayNum'] == closest_day)]
    
    # Calculate average ranking for each team
    team_rankings = season_massey.groupby('TeamID')['OrdinalRank'].mean().reset_index()
    team_rankings = team_rankings.rename(columns={'OrdinalRank': 'AvgRank'})
    
    # Add season column
    team_rankings['Season'] = season
    
    return team_rankings

def create_matchup_features(team1_stats, team2_stats):
    """
    Create features for a matchup between two teams.
    
    Parameters:
    -----------
    team1_stats : Series
        Statistics for team 1
    team2_stats : Series
        Statistics for team 2
    
    Returns:
    --------
    dict
        Features for the matchup
    """
    features = {}
    
    # Basic stats differences
    for stat in ['WinPercentage', 'AvgPointsScored', 'AvgPointsAllowed', 'AvgPointDifferential']:
        if stat in team1_stats and stat in team2_stats:
            features[f'{stat}Diff'] = team1_stats[stat] - team2_stats[stat]
    
    # Detailed stats differences
    for stat in ['FGPercentage', 'FG3Percentage', 'FTPercentage', 'AvgRebounds', 'AvgAst', 'AvgTO', 'AvgStl', 'AvgBlk']:
        if stat in team1_stats and stat in team2_stats:
            features[f'{stat}Diff'] = team1_stats[stat] - team2_stats[stat]
    
    # Ranking difference
    if 'AvgRank' in team1_stats and 'AvgRank' in team2_stats:
        features['RankDiff'] = team2_stats['AvgRank'] - team1_stats['AvgRank']  # Lower rank is better
    
    return features

def prepare_training_data(dfs, gender='M', start_season=2003, end_season=2024):
    """
    Prepare training data for the model.
    
    Parameters:
    -----------
    dfs : dict
        Dictionary containing loaded dataframes
    gender : str
        'M' for men's data, 'W' for women's data
    start_season : int
        First season to include in training data
    end_season : int
        Last season to include in training data
    
    Returns:
    --------
    tuple
        X (features) and y (target) for model training
    """
    # Get tournament data
    tourney_data = prepare_tournament_data(dfs, gender)
    
    # Filter for the specified seasons
    tourney_data = tourney_data[(tourney_data['Season'] >= start_season) & (tourney_data['Season'] <= end_season)]
    
    # Calculate team stats for each season
    all_team_stats = []
    all_detailed_stats = []
    all_rankings = []
    
    for season in range(start_season, end_season + 1):
        # Basic stats
        team_stats = calculate_team_stats(dfs, gender, season)
        if team_stats is not None and len(team_stats) > 0:
            all_team_stats.append(team_stats)
        
        # Detailed stats
        detailed_stats = calculate_detailed_stats(dfs, gender, season)
        if detailed_stats is not None and len(detailed_stats) > 0:
            all_detailed_stats.append(detailed_stats)
        
        # Rankings
        if gender == 'M':
            rankings = get_team_rankings(dfs, season)
            if rankings is not None and len(rankings) > 0:
                all_rankings.append(rankings)
    
    # Combine stats from all seasons
    team_stats_df = pd.concat(all_team_stats) if all_team_stats else pd.DataFrame()
    detailed_stats_df = pd.concat(all_detailed_stats) if all_detailed_stats else pd.DataFrame()
    rankings_df = pd.concat(all_rankings) if all_rankings else pd.DataFrame()
    
    # Merge all stats
    if len(detailed_stats_df) > 0:
        team_stats_df = pd.merge(
            team_stats_df,
            detailed_stats_df,
            on=['Season', 'TeamID'],
            how='left',
            suffixes=('', '_detailed')
        )
    
    if len(rankings_df) > 0:
        team_stats_df = pd.merge(
            team_stats_df,
            rankings_df,
            on=['Season', 'TeamID'],
            how='left'
        )
    
    # Create features for each tournament matchup
    features = []
    targets = []
    
    for _, row in tourney_data.iterrows():
        season = row['Season']
        team1_id = row['Team1ID']
        team2_id = row['Team2ID']
        
        # Get stats for both teams
        team1_stats = team_stats_df[(team_stats_df['Season'] == season) & (team_stats_df['TeamID'] == team1_id)]
        team2_stats = team_stats_df[(team_stats_df['Season'] == season) & (team_stats_df['TeamID'] == team2_id)]
        
        # Skip if stats are missing for either team
        if len(team1_stats) == 0 or len(team2_stats) == 0:
            continue
        
        # Create matchup features
        matchup_features = create_matchup_features(team1_stats.iloc[0], team2_stats.iloc[0])
        
        # Add seed information
        matchup_features['Team1Seed'] = row['Team1Seed']
        matchup_features['Team2Seed'] = row['Team2Seed']
        matchup_features['SeedDiff'] = row['SeedDiff']
        
        # Add season
        matchup_features['Season'] = season
        
        # Add team IDs
        matchup_features['Team1ID'] = team1_id
        matchup_features['Team2ID'] = team2_id
        
        features.append(matchup_features)
        targets.append(row['Team1Won'])
    
    # Convert to DataFrame
    X = pd.DataFrame(features)
    y = np.array(targets)
    
    return X, y

print("March Machine Learning Mania 2025 - Model Setup Complete") 