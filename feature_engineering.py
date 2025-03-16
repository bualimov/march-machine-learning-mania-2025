#!/usr/bin/env python3
# Feature engineering module for March Machine Learning Mania 2025

import pandas as pd
import numpy as np
from data_loader import extract_seed_number

def prepare_team_stats(dfs, gender='M', start_season=2003, end_season=2024):
    """
    Calculate team statistics for all seasons.
    
    Parameters:
    -----------
    dfs : dict
        Dictionary containing loaded dataframes
    gender : str
        'M' for men's data, 'W' for women's data
    start_season : int
        First season to include
    end_season : int
        Last season to include
    
    Returns:
    --------
    DataFrame
        Team statistics for all seasons
    """
    # Calculate basic stats for each season
    all_team_stats = []
    all_detailed_stats = []
    all_rankings = []
    
    for season in range(start_season, end_season + 1):
        print(f"  Calculating stats for {season} season...")
        
        # Basic stats
        team_stats = calculate_team_stats(dfs, season)
        if team_stats is not None and len(team_stats) > 0:
            all_team_stats.append(team_stats)
        
        # Detailed stats
        detailed_stats = calculate_detailed_stats(dfs, season)
        if detailed_stats is not None and len(detailed_stats) > 0:
            all_detailed_stats.append(detailed_stats)
        
        # Rankings (only for men's data)
        if gender == 'M' and 'massey_ordinals' in dfs:
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
    
    # Add conference strength
    team_stats_df = add_conference_strength(dfs, team_stats_df)
    
    # Add historical tournament performance
    team_stats_df = add_historical_tournament_performance(dfs, team_stats_df)
    
    return team_stats_df

def calculate_team_stats(dfs, season):
    """
    Calculate basic team statistics for a given season.
    
    Parameters:
    -----------
    dfs : dict
        Dictionary containing loaded dataframes
    season : int
        Season to calculate stats for
    
    Returns:
    --------
    DataFrame
        Basic team statistics for the specified season
    """
    # Get regular season results
    reg_season = dfs['regular_season'].copy()
    reg_season = reg_season[reg_season['Season'] == season]
    
    # Initialize a dictionary to store team stats
    team_stats = {}
    
    # Get unique teams
    teams = dfs['teams']['TeamID'].unique()
    
    for team in teams:
        # Games where the team won
        won_games = reg_season[reg_season['WTeamID'] == team]
        # Games where the team lost
        lost_games = reg_season[reg_season['LTeamID'] == team]
        
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
        team_stats[team] = {
            'Season': season,
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

def calculate_detailed_stats(dfs, season):
    """
    Calculate detailed team statistics for a given season.
    
    Parameters:
    -----------
    dfs : dict
        Dictionary containing loaded dataframes
    season : int
        Season to calculate stats for
    
    Returns:
    --------
    DataFrame
        Detailed team statistics for the specified season
    """
    # Get detailed regular season results
    detailed_results = dfs['regular_season_detailed'].copy()
    detailed_results = detailed_results[detailed_results['Season'] == season]
    
    # If no detailed results for this season, return None
    if len(detailed_results) == 0:
        return None
    
    # Initialize a dictionary to store team stats
    detailed_stats = {}
    
    # Get unique teams
    teams = dfs['teams']['TeamID'].unique()
    
    for team in teams:
        # Games where the team won
        won_games = detailed_results[detailed_results['WTeamID'] == team]
        # Games where the team lost
        lost_games = detailed_results[detailed_results['LTeamID'] == team]
        
        # Skip if team didn't play in this season
        if len(won_games) == 0 and len(lost_games) == 0:
            continue
        
        # Calculate total games
        total_games = len(won_games) + len(lost_games)
        
        # Initialize stats dictionary
        stats = {
            'Season': season,
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
        detailed_stats[team] = stats
    
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
    
    # Filter for the specified season
    season_massey = massey[massey['Season'] == season]
    
    if len(season_massey) == 0:
        return None
    
    # Get the closest day to the specified day
    available_days = season_massey['RankingDayNum'].unique()
    closest_day = available_days[np.abs(available_days - day_num).argmin()]
    
    # Filter for the closest day
    day_massey = season_massey[season_massey['RankingDayNum'] == closest_day]
    
    # Calculate average ranking for each team
    team_rankings = day_massey.groupby('TeamID')['OrdinalRank'].mean().reset_index()
    team_rankings = team_rankings.rename(columns={'OrdinalRank': 'AvgRank'})
    
    # Add season column
    team_rankings['Season'] = season
    
    return team_rankings

def add_conference_strength(dfs, team_stats_df):
    """
    Add conference strength metrics to team statistics.
    
    Parameters:
    -----------
    dfs : dict
        Dictionary containing loaded dataframes
    team_stats_df : DataFrame
        Team statistics DataFrame
    
    Returns:
    --------
    DataFrame
        Team statistics with conference strength metrics added
    """
    # Get team conferences
    team_conferences = dfs['team_conferences'].copy()
    
    # Merge team conferences with team stats
    team_stats_with_conf = pd.merge(
        team_stats_df,
        team_conferences[['Season', 'TeamID', 'ConfAbbrev']],
        on=['Season', 'TeamID'],
        how='left'
    )
    
    # Calculate conference strength for each season
    seasons = team_stats_with_conf['Season'].unique()
    
    for season in seasons:
        season_stats = team_stats_with_conf[team_stats_with_conf['Season'] == season]
        
        # Calculate average win percentage and point differential by conference
        conf_strength = season_stats.groupby('ConfAbbrev')[['WinPercentage', 'AvgPointDifferential']].mean().reset_index()
        conf_strength = conf_strength.rename(columns={
            'WinPercentage': 'ConfAvgWinPercentage',
            'AvgPointDifferential': 'ConfAvgPointDifferential'
        })
        
        # Merge conference strength back to team stats
        team_stats_with_conf.loc[team_stats_with_conf['Season'] == season] = pd.merge(
            team_stats_with_conf[team_stats_with_conf['Season'] == season],
            conf_strength,
            on='ConfAbbrev',
            how='left'
        )
    
    return team_stats_with_conf

def add_historical_tournament_performance(dfs, team_stats_df):
    """
    Add historical tournament performance metrics to team statistics.
    
    Parameters:
    -----------
    dfs : dict
        Dictionary containing loaded dataframes
    team_stats_df : DataFrame
        Team statistics DataFrame
    
    Returns:
    --------
    DataFrame
        Team statistics with historical tournament performance metrics added
    """
    # Get tournament results
    tourney_results = dfs['tourney_results'].copy()
    
    # Calculate tournament wins and appearances for each team and season
    tourney_stats = []
    
    for season in team_stats_df['Season'].unique():
        # Get tournament results up to the previous season
        past_results = tourney_results[tourney_results['Season'] < season]
        
        # Calculate tournament appearances and wins for each team
        team_appearances = pd.concat([
            past_results['WTeamID'].rename('TeamID'),
            past_results['LTeamID'].rename('TeamID')
        ]).value_counts().reset_index()
        team_appearances.columns = ['TeamID', 'TourneyAppearances']
        
        team_wins = past_results['WTeamID'].value_counts().reset_index()
        team_wins.columns = ['TeamID', 'TourneyWins']
        
        # Merge appearances and wins
        team_tourney_stats = pd.merge(
            team_appearances,
            team_wins,
            on='TeamID',
            how='left'
        )
        team_tourney_stats['TourneyWins'] = team_tourney_stats['TourneyWins'].fillna(0)
        
        # Calculate win percentage in tournament
        team_tourney_stats['TourneyWinPercentage'] = team_tourney_stats['TourneyWins'] / team_tourney_stats['TourneyAppearances']
        
        # Add season column
        team_tourney_stats['Season'] = season
        
        tourney_stats.append(team_tourney_stats)
    
    # Combine all seasons
    tourney_stats_df = pd.concat(tourney_stats)
    
    # Merge with team stats
    team_stats_with_history = pd.merge(
        team_stats_df,
        tourney_stats_df,
        on=['Season', 'TeamID'],
        how='left'
    )
    
    # Fill NaN values with 0 (for teams with no tournament history)
    for col in ['TourneyAppearances', 'TourneyWins', 'TourneyWinPercentage']:
        team_stats_with_history[col] = team_stats_with_history[col].fillna(0)
    
    return team_stats_with_history

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
            # Handle missing values
            val1 = team1_stats[stat] if not pd.isna(team1_stats[stat]) else 0
            val2 = team2_stats[stat] if not pd.isna(team2_stats[stat]) else 0
            features[f'{stat}Diff'] = val1 - val2
    
    # Conference strength differences
    for stat in ['ConfAvgWinPercentage', 'ConfAvgPointDifferential']:
        if stat in team1_stats and stat in team2_stats:
            # Handle missing values
            val1 = team1_stats[stat] if not pd.isna(team1_stats[stat]) else 0
            val2 = team2_stats[stat] if not pd.isna(team2_stats[stat]) else 0
            features[f'{stat}Diff'] = val1 - val2
    
    # Historical tournament performance differences
    for stat in ['TourneyAppearances', 'TourneyWins', 'TourneyWinPercentage']:
        if stat in team1_stats and stat in team2_stats:
            # Handle missing values
            val1 = team1_stats[stat] if not pd.isna(team1_stats[stat]) else 0
            val2 = team2_stats[stat] if not pd.isna(team2_stats[stat]) else 0
            features[f'{stat}Diff'] = val1 - val2
    
    # Detailed stats differences
    for stat in ['FGPercentage', 'FG3Percentage', 'FTPercentage', 'AvgRebounds', 'AvgAst', 'AvgTO', 'AvgStl', 'AvgBlk']:
        if stat in team1_stats and stat in team2_stats:
            # Handle missing values
            val1 = team1_stats[stat] if not pd.isna(team1_stats[stat]) else 0
            val2 = team2_stats[stat] if not pd.isna(team2_stats[stat]) else 0
            features[f'{stat}Diff'] = val1 - val2
    
    # Opponent stats differences
    for stat in ['OpponentFGPercentage', 'OpponentFG3Percentage', 'OpponentFTPercentage', 'AvgOpponentRebounds']:
        if stat in team1_stats and stat in team2_stats:
            # Handle missing values
            val1 = team1_stats[stat] if not pd.isna(team1_stats[stat]) else 0
            val2 = team2_stats[stat] if not pd.isna(team2_stats[stat]) else 0
            features[f'{stat}Diff'] = val1 - val2
    
    # Ranking difference
    if 'AvgRank' in team1_stats and 'AvgRank' in team2_stats:
        # Handle missing values
        val1 = team1_stats['AvgRank'] if not pd.isna(team1_stats['AvgRank']) else 0
        val2 = team2_stats['AvgRank'] if not pd.isna(team2_stats['AvgRank']) else 0
        features['RankDiff'] = val2 - val1  # Lower rank is better
    
    return features

def prepare_training_data(dfs, team_stats_df, gender='M', start_season=2003, end_season=2024):
    """
    Prepare training data for the model.
    
    Parameters:
    -----------
    dfs : dict
        Dictionary containing loaded dataframes
    team_stats_df : DataFrame
        Team statistics DataFrame
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