import pandas as pd
import numpy as np
from typing import Dict, List
import os

def analyze_tournament_outcomes(data_path: str, start_year: int = 2019, gender: str = 'M'):
    """Analyze tournament outcomes to determine empirical feature importance."""
    # Load tournament results
    tourney_results = pd.read_csv(os.path.join(data_path, f'{gender}NCAATourneyCompactResults.csv'))
    regular_season = pd.read_csv(os.path.join(data_path, f'{gender}RegularSeasonCompactResults.csv'))
    rankings = pd.read_csv(os.path.join(data_path, f'{gender}MasseyOrdinals.csv')) if gender == 'M' else None
    conferences = pd.read_csv(os.path.join(data_path, f'{gender}TeamConferences.csv'))
    
    # Filter for recent years
    recent_tourney = tourney_results[tourney_results['Season'] >= start_year]
    
    # Initialize stats dictionaries
    seed_diff_wins = {i: 0 for i in range(-16, 17)}
    seed_diff_total = {i: 0 for i in range(-16, 17)}
    conf_strength_impact = []
    point_diff_impact = []
    ranking_impact = []
    win_pct_impact = []
    
    for _, game in recent_tourney.iterrows():
        season = game['Season']
        winner_id = game['WTeamID']
        loser_id = game['LTeamID']
        
        # Get regular season stats for both teams
        winner_season = regular_season[
            (regular_season['Season'] == season) & 
            ((regular_season['WTeamID'] == winner_id) | (regular_season['LTeamID'] == winner_id))
        ]
        loser_season = regular_season[
            (regular_season['Season'] == season) & 
            ((regular_season['WTeamID'] == loser_id) | (regular_season['LTeamID'] == loser_id))
        ]
        
        # Calculate win percentages
        winner_wins = len(winner_season[winner_season['WTeamID'] == winner_id])
        winner_losses = len(winner_season[winner_season['LTeamID'] == winner_id])
        winner_win_pct = winner_wins / (winner_wins + winner_losses)
        
        loser_wins = len(loser_season[loser_season['WTeamID'] == loser_id])
        loser_losses = len(loser_season[loser_season['LTeamID'] == loser_id])
        loser_win_pct = loser_wins / (loser_wins + loser_losses)
        
        win_pct_impact.append(winner_win_pct > loser_win_pct)
        
        # Calculate point differentials
        winner_point_diff = (
            winner_season[winner_season['WTeamID'] == winner_id]['WScore'].sum() +
            winner_season[winner_season['LTeamID'] == winner_id]['LScore'].sum()
        ) / (winner_wins + winner_losses)
        
        loser_point_diff = (
            loser_season[loser_season['WTeamID'] == loser_id]['WScore'].sum() +
            loser_season[loser_season['LTeamID'] == loser_id]['LScore'].sum()
        ) / (loser_wins + loser_losses)
        
        point_diff_impact.append(winner_point_diff > loser_point_diff)
        
        # Get conference strength
        winner_conf = conferences[
            (conferences['Season'] == season) & 
            (conferences['TeamID'] == winner_id)
        ]['ConfAbbrev'].iloc[0]
        
        loser_conf = conferences[
            (conferences['Season'] == season) & 
            (conferences['TeamID'] == loser_id)
        ]['ConfAbbrev'].iloc[0]
        
        # Get rankings if available
        if rankings is not None:
            winner_rank = rankings[
                (rankings['Season'] == season) &
                (rankings['TeamID'] == winner_id) &
                (rankings['RankingDayNum'] == 133)
            ]['OrdinalRank'].mean()
            
            loser_rank = rankings[
                (rankings['Season'] == season) &
                (rankings['TeamID'] == loser_id) &
                (rankings['RankingDayNum'] == 133)
            ]['OrdinalRank'].mean()
            
            ranking_impact.append(winner_rank < loser_rank)  # Lower rank is better
    
    # Calculate feature importance based on prediction accuracy
    importance = {
        'WinPct': np.mean(win_pct_impact),
        'PointDiff': np.mean(point_diff_impact),
        'Ranking': np.mean(ranking_impact) if ranking_impact else None
    }
    
    return importance

def main():
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_path, 'march-machine-learning-mania-2025-final-dataset')
    
    # Analyze both men's and women's tournaments
    for gender in ['M', 'W']:
        print(f"\nAnalyzing {('Men' if gender == 'M' else 'Women')}'s Tournament (2019-2024)")
        importance = analyze_tournament_outcomes(data_path, start_year=2019, gender=gender)
        
        print("\nFeature Importance (% of times feature predicted winner correctly):")
        for feature, value in importance.items():
            if value is not None:
                print(f"{feature}: {value:.4f}")

if __name__ == "__main__":
    main() 