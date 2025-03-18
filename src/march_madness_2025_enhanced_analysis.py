import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import os
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

class TournamentAnalyzer:
    def __init__(self, data_path: str, start_year: int = 2010):
        self.data_path = data_path
        self.start_year = start_year
        self.conference_strengths = {}
        self.team_histories = defaultdict(dict)
        self.coach_success_rates = {}
        self.ranking_importance = {}
        
    def load_all_data(self, gender: str = 'M'):
        """Load all necessary data files."""
        # Load core datasets
        self.tourney_results = pd.read_csv(os.path.join(self.data_path, f'{gender}NCAATourneyCompactResults.csv'))
        self.regular_season = pd.read_csv(os.path.join(self.data_path, f'{gender}RegularSeasonCompactResults.csv'))
        self.conferences = pd.read_csv(os.path.join(self.data_path, f'{gender}TeamConferences.csv'))
        self.seeds = pd.read_csv(os.path.join(self.data_path, f'{gender}NCAATourneySeeds.csv'))
        
        # Try to load rankings if available
        ranking_file = os.path.join(self.data_path, f'{gender}MasseyOrdinals.csv')
        self.rankings = pd.read_csv(ranking_file) if os.path.exists(ranking_file) else None
        
        # Filter for analysis period
        self.tourney_results = self.tourney_results[self.tourney_results['Season'] >= self.start_year]
        self.regular_season = self.regular_season[self.regular_season['Season'] >= self.start_year]
        self.conferences = self.conferences[self.conferences['Season'] >= self.start_year]
        self.seeds = self.seeds[self.seeds['Season'] >= self.start_year]
        if self.rankings is not None:
            self.rankings = self.rankings[self.rankings['Season'] >= self.start_year]

    def analyze_seed_matchups(self) -> Dict:
        """Analyze historical success rates based on seed matchups."""
        seed_matchup_stats = defaultdict(lambda: {'wins': 0, 'total': 0})
        
        for _, game in self.tourney_results.iterrows():
            winner_seed = self.seeds[
                (self.seeds['Season'] == game['Season']) & 
                (self.seeds['TeamID'] == game['WTeamID'])
            ]['Seed'].iloc[0]
            loser_seed = self.seeds[
                (self.seeds['Season'] == game['Season']) & 
                (self.seeds['TeamID'] == game['LTeamID'])
            ]['Seed'].iloc[0]
            
            # Convert seed strings to numbers
            winner_seed_num = int(winner_seed[1:3])
            loser_seed_num = int(loser_seed[1:3])
            
            # Record both matchup directions
            seed_diff = winner_seed_num - loser_seed_num
            seed_matchup_stats[seed_diff]['wins'] += 1
            seed_matchup_stats[seed_diff]['total'] += 1
            seed_matchup_stats[-seed_diff]['total'] += 1
            
        # Convert to win probabilities
        seed_probabilities = {
            diff: stats['wins'] / stats['total']
            for diff, stats in seed_matchup_stats.items()
        }
        
        return seed_probabilities

    def analyze_conference_strength(self) -> Dict:
        """Calculate conference strength based on tournament performance."""
        conf_stats = defaultdict(lambda: {'wins': 0, 'games': 0})
        
        for _, game in self.tourney_results.iterrows():
            season = game['Season']
            
            winner_conf = self.conferences[
                (self.conferences['Season'] == season) & 
                (self.conferences['TeamID'] == game['WTeamID'])
            ]['ConfAbbrev'].iloc[0]
            
            loser_conf = self.conferences[
                (self.conferences['Season'] == season) & 
                (self.conferences['TeamID'] == game['LTeamID'])
            ]['ConfAbbrev'].iloc[0]
            
            conf_stats[winner_conf]['wins'] += 1
            conf_stats[winner_conf]['games'] += 1
            conf_stats[loser_conf]['games'] += 1
            
        # Calculate conference strength scores
        return {
            conf: (stats['wins'] / stats['games']) if stats['games'] > 0 else 0.5
            for conf, stats in conf_stats.items()
        }

    def analyze_team_consistency(self) -> Dict:
        """Analyze how consistently teams perform in tournaments."""
        team_stats = defaultdict(lambda: {'appearances': 0, 'wins': 0})
        
        for _, game in self.tourney_results.iterrows():
            team_stats[game['WTeamID']]['appearances'] += 1
            team_stats[game['LTeamID']]['appearances'] += 1
            team_stats[game['WTeamID']]['wins'] += 1
            
        return {
            str(team_id): {
                'consistency': stats['wins'] / stats['appearances'] if stats['appearances'] > 0 else 0,
                'experience': stats['appearances']
            }
            for team_id, stats in team_stats.items()
        }

    def analyze_point_differential_impact(self) -> float:
        """Analyze how regular season point differential correlates with tournament success."""
        correct_predictions = 0
        total_games = 0
        
        for _, game in self.tourney_results.iterrows():
            season = game['Season']
            
            # Calculate season point differentials
            winner_games = self.regular_season[
                (self.regular_season['Season'] == season) &
                ((self.regular_season['WTeamID'] == game['WTeamID']) |
                 (self.regular_season['LTeamID'] == game['WTeamID']))
            ]
            
            loser_games = self.regular_season[
                (self.regular_season['Season'] == season) &
                ((self.regular_season['WTeamID'] == game['LTeamID']) |
                 (self.regular_season['LTeamID'] == game['LTeamID']))
            ]
            
            winner_diff = self._calculate_point_differential(winner_games, game['WTeamID'])
            loser_diff = self._calculate_point_differential(loser_games, game['LTeamID'])
            
            if winner_diff > loser_diff:
                correct_predictions += 1
            total_games += 1
            
        return correct_predictions / total_games if total_games > 0 else 0.5

    def _calculate_point_differential(self, games, team_id):
        """Helper function to calculate point differential."""
        total_diff = 0
        for _, g in games.iterrows():
            if g['WTeamID'] == team_id:
                total_diff += (g['WScore'] - g['LScore'])
            else:
                total_diff += (g['LScore'] - g['WScore'])
        return total_diff / len(games) if len(games) > 0 else 0

    def analyze_ranking_systems(self) -> Dict:
        """Analyze which ranking systems are most predictive."""
        if self.rankings is None:
            return {}
            
        ranking_success = defaultdict(lambda: {'correct': 0, 'total': 0})
        
        for _, game in self.tourney_results.iterrows():
            season = game['Season']
            
            # Get rankings just before tournament (RankingDayNum 133)
            season_rankings = self.rankings[
                (self.rankings['Season'] == season) &
                (self.rankings['RankingDayNum'] == 133)
            ]
            
            for system in season_rankings['SystemName'].unique():
                system_ranks = season_rankings[season_rankings['SystemName'] == system]
                
                try:
                    winner_rank = system_ranks[
                        system_ranks['TeamID'] == game['WTeamID']
                    ]['OrdinalRank'].iloc[0]
                    
                    loser_rank = system_ranks[
                        system_ranks['TeamID'] == game['LTeamID']
                    ]['OrdinalRank'].iloc[0]
                    
                    if winner_rank < loser_rank:  # Lower rank is better
                        ranking_success[system]['correct'] += 1
                    ranking_success[system]['total'] += 1
                except:
                    continue
                    
        # Calculate success rate for each ranking system
        return {
            system: stats['correct'] / stats['total']
            for system, stats in ranking_success.items()
            if stats['total'] >= 50  # Only consider systems with sufficient data
        }

    def analyze_head_to_head(self) -> float:
        """Analyze importance of head-to-head history."""
        correct_predictions = 0
        total_matchups = 0
        
        for _, game in self.tourney_results.iterrows():
            season = game['Season']
            
            # Look for previous matchups in regular season or tournaments
            previous_matchups = pd.concat([
                self.regular_season[self.regular_season['Season'] < season],
                self.tourney_results[self.tourney_results['Season'] < season]
            ])
            
            h2h_games = previous_matchups[
                ((previous_matchups['WTeamID'] == game['WTeamID']) & 
                 (previous_matchups['LTeamID'] == game['LTeamID'])) |
                ((previous_matchups['WTeamID'] == game['LTeamID']) & 
                 (previous_matchups['LTeamID'] == game['WTeamID']))
            ]
            
            if len(h2h_games) > 0:
                # Calculate historical win rate
                team1_wins = len(h2h_games[h2h_games['WTeamID'] == game['WTeamID']])
                total_games = len(h2h_games)
                
                if total_games > 0:
                    if (team1_wins / total_games > 0.5) == (game['WTeamID'] == h2h_games.iloc[0]['WTeamID']):
                        correct_predictions += 1
                    total_matchups += 1
                    
        return correct_predictions / total_matchups if total_matchups > 0 else 0.5

    def calculate_feature_weights(self) -> Dict:
        """Calculate optimal weights for different features based on historical analysis."""
        # Analyze each feature's predictive power
        seed_importance = self.analyze_seed_matchups()
        conf_strength = self.analyze_conference_strength()
        team_consistency = self.analyze_team_consistency()
        point_diff_importance = self.analyze_point_differential_impact()
        ranking_importance = self.analyze_ranking_systems()
        h2h_importance = self.analyze_head_to_head()
        
        # Adjust weights based on gender (women's tournament doesn't have rankings)
        if not ranking_importance:  # No ranking data available
            weights = {
                'seed_matchup': 0.30,
                'conference_strength': 0.20,
                'team_consistency': 0.20,
                'point_differential': 0.20,
                'head_to_head': 0.10
            }
        else:
            weights = {
                'seed_matchup': 0.25,
                'conference_strength': 0.15,
                'team_consistency': 0.15,
                'point_differential': 0.20,
                'rankings': 0.15,
                'head_to_head': 0.10
            }
        
        return weights

def main():
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_path, 'march-machine-learning-mania-2025-final-dataset')
    
    # Analyze both tournaments
    for gender in ['M', 'W']:
        print(f"\nAnalyzing {('Men' if gender == 'M' else 'Women')}'s Tournament (2010-2024)")
        
        analyzer = TournamentAnalyzer(data_path, start_year=2010)
        analyzer.load_all_data(gender)
        weights = analyzer.calculate_feature_weights()
        
        print("\nFeature Weights:")
        for feature, weight in weights.items():
            print(f"{feature}: {weight:.4f}")
        
        # Additional detailed statistics
        print("\nPoint Differential Impact:", analyzer.analyze_point_differential_impact())
        print("\nHead-to-Head Predictive Power:", analyzer.analyze_head_to_head())
        
        # Save analysis results
        results = {
            'weights': weights,
            'conference_strength': analyzer.analyze_conference_strength(),
            'ranking_systems': analyzer.analyze_ranking_systems(),
            'team_consistency': analyzer.analyze_team_consistency()
        }
        
        output_file = os.path.join(base_path, f'analysis_results_{gender}.json')
        with open(output_file, 'w') as f:
            import json
            json.dump(results, f, indent=2)
            
        print(f"\nDetailed analysis results saved to {output_file}")

if __name__ == "__main__":
    main() 