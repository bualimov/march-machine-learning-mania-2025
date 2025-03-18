import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import os
from collections import defaultdict

class EmpiricalWeightAnalyzer:
    def __init__(self, data_path: str, start_year: int = 2010):
        self.data_path = data_path
        self.start_year = start_year
        
    def load_data(self, gender: str = 'M'):
        """Load all necessary data files."""
        self.tourney_results = pd.read_csv(os.path.join(self.data_path, f'{gender}NCAATourneyCompactResults.csv'))
        self.regular_season = pd.read_csv(os.path.join(self.data_path, f'{gender}RegularSeasonCompactResults.csv'))
        self.seeds = pd.read_csv(os.path.join(self.data_path, f'{gender}NCAATourneySeeds.csv'))
        self.conferences = pd.read_csv(os.path.join(self.data_path, f'{gender}TeamConferences.csv'))
        
        # Filter for analysis period
        self.tourney_results = self.tourney_results[self.tourney_results['Season'] >= self.start_year]
        self.regular_season = self.regular_season[self.regular_season['Season'] >= self.start_year]
        self.seeds = self.seeds[self.seeds['Season'] >= self.start_year]
        self.conferences = self.conferences[self.conferences['Season'] >= self.start_year]
        
    def calculate_team_stats(self, season: int, team_id: int) -> Dict:
        """Calculate comprehensive team statistics for a given season."""
        season_games = self.regular_season[self.regular_season['Season'] == season]
        team_games = season_games[
            (season_games['WTeamID'] == team_id) | 
            (season_games['LTeamID'] == team_id)
        ]
        
        if len(team_games) == 0:
            return None
            
        stats = {
            'wins': 0, 'losses': 0,
            'points_for': 0, 'points_against': 0,
            'home_wins': 0, 'away_wins': 0, 'neutral_wins': 0,
            'strength_of_schedule': 0,
            'last_10_games': {'wins': 0, 'losses': 0},
            'conference': self.conferences[
                (self.conferences['Season'] == season) & 
                (self.conferences['TeamID'] == team_id)
            ]['ConfAbbrev'].iloc[0]
        }
        
        # Calculate basic stats
        for _, game in team_games.iterrows():
            is_winner = game['WTeamID'] == team_id
            is_home = game['WLoc'] == 'H' if is_winner else game['WLoc'] == 'A'
            is_neutral = game['WLoc'] == 'N'
            
            stats['wins' if is_winner else 'losses'] += 1
            stats['points_for' if is_winner else 'points_against'] += game['WScore']
            stats['points_against' if is_winner else 'points_for'] += game['LScore']
            
            if is_home:
                stats['home_wins' if is_winner else 'away_wins'] += 1
            elif is_neutral:
                stats['neutral_wins'] += 1
                
        # Calculate advanced stats
        games_played = stats['wins'] + stats['losses']
        stats['win_pct'] = stats['wins'] / games_played
        stats['point_diff'] = (stats['points_for'] - stats['points_against']) / games_played
        stats['home_win_pct'] = stats['home_wins'] / games_played
        stats['away_win_pct'] = stats['away_wins'] / games_played
        stats['neutral_win_pct'] = stats['neutral_wins'] / games_played
        
        # Calculate strength of schedule
        opponent_win_pcts = []
        for _, game in team_games.iterrows():
            opponent_id = game['LTeamID'] if game['WTeamID'] == team_id else game['WTeamID']
            opponent_games = season_games[
                (season_games['WTeamID'] == opponent_id) | 
                (season_games['LTeamID'] == opponent_id)
            ]
            if len(opponent_games) > 0:
                opponent_wins = len(opponent_games[opponent_games['WTeamID'] == opponent_id])
                opponent_win_pcts.append(opponent_wins / len(opponent_games))
        
        stats['strength_of_schedule'] = np.mean(opponent_win_pcts) if opponent_win_pcts else 0.5
        
        # Calculate last 10 games performance
        last_10 = team_games.tail(10)
        stats['last_10_games']['wins'] = len(last_10[last_10['WTeamID'] == team_id])
        stats['last_10_games']['losses'] = len(last_10[last_10['LTeamID'] == team_id])
        stats['last_10_win_pct'] = stats['last_10_games']['wins'] / 10
        
        return stats
        
    def calculate_empirical_weights(self) -> Dict:
        """Calculate empirical weights based on historical performance."""
        feature_importance = defaultdict(list)
        
        for _, game in self.tourney_results.iterrows():
            season = game['Season']
            winner_id = game['WTeamID']
            loser_id = game['LTeamID']
            
            # Get team stats for the season
            winner_stats = self.calculate_team_stats(season, winner_id)
            loser_stats = self.calculate_team_stats(season, loser_id)
            
            if winner_stats is None or loser_stats is None:
                continue
                
            # Calculate feature differences
            features = {
                'win_pct': winner_stats['win_pct'] - loser_stats['win_pct'],
                'point_diff': winner_stats['point_diff'] - loser_stats['point_diff'],
                'strength_of_schedule': winner_stats['strength_of_schedule'] - loser_stats['strength_of_schedule'],
                'last_10_win_pct': winner_stats['last_10_win_pct'] - loser_stats['last_10_win_pct'],
                'home_win_pct': winner_stats['home_win_pct'] - loser_stats['home_win_pct'],
                'away_win_pct': winner_stats['away_win_pct'] - loser_stats['away_win_pct'],
                'neutral_win_pct': winner_stats['neutral_win_pct'] - loser_stats['neutral_win_pct']
            }
            
            # Record feature importance
            for feature, diff in features.items():
                feature_importance[feature].append(1 if diff > 0 else 0)
        
        # Calculate empirical weights
        weights = {}
        total_games = len(self.tourney_results)
        
        for feature, results in feature_importance.items():
            if len(results) > 0:
                accuracy = np.mean(results)
                weights[feature] = accuracy
        
        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        weights = {k: v/total_weight for k, v in weights.items()}
        
        return weights
        
    def analyze_seed_importance(self) -> float:
        """Analyze how important seeding is in predicting outcomes."""
        correct_predictions = 0
        total_games = 0
        
        for _, game in self.tourney_results.iterrows():
            season = game['Season']
            
            winner_seed = self.seeds[
                (self.seeds['Season'] == season) & 
                (self.seeds['TeamID'] == game['WTeamID'])
            ]['Seed'].iloc[0]
            
            loser_seed = self.seeds[
                (self.seeds['Season'] == season) & 
                (self.seeds['TeamID'] == game['LTeamID'])
            ]['Seed'].iloc[0]
            
            # Convert seed strings to numbers
            winner_seed_num = int(winner_seed[1:3])
            loser_seed_num = int(loser_seed[1:3])
            
            if winner_seed_num < loser_seed_num:  # Lower seed number is better
                correct_predictions += 1
            total_games += 1
            
        return correct_predictions / total_games if total_games > 0 else 0.5

def main():
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_path, 'march-machine-learning-mania-2025-final-dataset')
    
    # Analyze both tournaments
    for gender in ['M', 'W']:
        print(f"\nAnalyzing {('Men' if gender == 'M' else 'Women')}'s Tournament (2010-2024)")
        
        analyzer = EmpiricalWeightAnalyzer(data_path, start_year=2010)
        analyzer.load_data(gender)
        
        # Calculate empirical weights
        weights = analyzer.calculate_empirical_weights()
        seed_importance = analyzer.analyze_seed_importance()
        
        # Add seed importance to weights
        weights['seed'] = seed_importance
        
        # Normalize all weights
        total_weight = sum(weights.values())
        weights = {k: v/total_weight for k, v in weights.items()}
        
        print("\nEmpirical Feature Weights:")
        for feature, weight in weights.items():
            print(f"{feature}: {weight:.4f}")
        
        # Save analysis results
        results = {
            'weights': weights,
            'seed_importance': seed_importance
        }
        
        output_file = os.path.join(base_path, f'empirical_weights_{gender}.json')
        with open(output_file, 'w') as f:
            import json
            json.dump(results, f, indent=2)
            
        print(f"\nDetailed analysis results saved to {output_file}")

if __name__ == "__main__":
    main() 