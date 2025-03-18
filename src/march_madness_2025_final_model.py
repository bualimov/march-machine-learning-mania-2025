import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import os
import json
from collections import defaultdict

class FinalMarchMadnessModel:
    def __init__(self, data_path: str, empirical_weights_path: str):
        self.data_path = data_path
        self.empirical_weights_path = empirical_weights_path
        self.team_stats = {}
        self.load_empirical_weights()
        
    def load_empirical_weights(self):
        """Load pre-computed empirical weights."""
        with open(self.empirical_weights_path, 'r') as f:
            self.empirical_weights = json.load(f)
            
    def load_current_season_data(self, season: int, gender: str = 'M'):
        """Load and process current season data."""
        # Load necessary data files
        self.regular_season = pd.read_csv(os.path.join(self.data_path, f'{gender}RegularSeasonCompactResults.csv'))
        self.seeds = pd.read_csv(os.path.join(self.data_path, f'{gender}NCAATourneySeeds.csv'))
        self.conferences = pd.read_csv(os.path.join(self.data_path, f'{gender}TeamConferences.csv'))
        
        # Filter for current season
        self.regular_season = self.regular_season[self.regular_season['Season'] == season]
        self.seeds = self.seeds[self.seeds['Season'] == season]
        self.conferences = self.conferences[self.conferences['Season'] == season]
        
        # Pre-calculate team statistics
        self._calculate_team_stats()
        
    def _calculate_team_stats(self):
        """Calculate comprehensive team statistics for current season."""
        team_stats = defaultdict(dict)
        
        # Process regular season results
        for _, game in self.regular_season.iterrows():
            # Winner stats
            if game['WTeamID'] not in team_stats:
                team_stats[game['WTeamID']] = {
                    'wins': 0, 'losses': 0,
                    'points_for': 0, 'points_against': 0,
                    'home_wins': 0, 'away_wins': 0, 'neutral_wins': 0,
                    'strength_of_schedule': 0,
                    'last_10_games': {'wins': 0, 'losses': 0},
                    'conference': self.conferences[
                        self.conferences['TeamID'] == game['WTeamID']
                    ]['ConfAbbrev'].iloc[0]
                }
            
            # Loser stats
            if game['LTeamID'] not in team_stats:
                team_stats[game['LTeamID']] = {
                    'wins': 0, 'losses': 0,
                    'points_for': 0, 'points_against': 0,
                    'home_wins': 0, 'away_wins': 0, 'neutral_wins': 0,
                    'strength_of_schedule': 0,
                    'last_10_games': {'wins': 0, 'losses': 0},
                    'conference': self.conferences[
                        self.conferences['TeamID'] == game['LTeamID']
                    ]['ConfAbbrev'].iloc[0]
                }
            
            # Update stats
            w_stats = team_stats[game['WTeamID']]
            l_stats = team_stats[game['LTeamID']]
            
            w_stats['wins'] += 1
            l_stats['losses'] += 1
            
            w_stats['points_for'] += game['WScore']
            w_stats['points_against'] += game['LScore']
            l_stats['points_for'] += game['LScore']
            l_stats['points_against'] += game['WScore']
            
            # Track location wins
            if game['WLoc'] == 'H':
                w_stats['home_wins'] += 1
            elif game['WLoc'] == 'A':
                w_stats['away_wins'] += 1
            else:
                w_stats['neutral_wins'] += 1
        
        # Calculate derived statistics
        for team_id, stats in team_stats.items():
            games_played = stats['wins'] + stats['losses']
            stats['win_pct'] = stats['wins'] / games_played
            stats['point_diff'] = (stats['points_for'] - stats['points_against']) / games_played
            stats['home_win_pct'] = stats['home_wins'] / games_played
            stats['away_win_pct'] = stats['away_wins'] / games_played
            stats['neutral_win_pct'] = stats['neutral_wins'] / games_played
            
            # Calculate strength of schedule
            opponent_win_pcts = []
            for _, game in self.regular_season.iterrows():
                if game['WTeamID'] == team_id:
                    opponent_id = game['LTeamID']
                elif game['LTeamID'] == team_id:
                    opponent_id = game['WTeamID']
                else:
                    continue
                    
                opponent_games = self.regular_season[
                    (self.regular_season['WTeamID'] == opponent_id) | 
                    (self.regular_season['LTeamID'] == opponent_id)
                ]
                if len(opponent_games) > 0:
                    opponent_wins = len(opponent_games[opponent_games['WTeamID'] == opponent_id])
                    opponent_win_pcts.append(opponent_wins / len(opponent_games))
            
            stats['strength_of_schedule'] = np.mean(opponent_win_pcts) if opponent_win_pcts else 0.5
            
            # Calculate last 10 games performance
            team_games = self.regular_season[
                (self.regular_season['WTeamID'] == team_id) | 
                (self.regular_season['LTeamID'] == team_id)
            ].tail(10)
            
            stats['last_10_games']['wins'] = len(team_games[team_games['WTeamID'] == team_id])
            stats['last_10_games']['losses'] = len(team_games[team_games['LTeamID'] == team_id])
            stats['last_10_win_pct'] = stats['last_10_games']['wins'] / 10
            
            # Get seed if available
            team_seed = self.seeds[self.seeds['TeamID'] == team_id]
            if not team_seed.empty:
                stats['seed'] = int(team_seed['Seed'].iloc[0][1:3])
            else:
                stats['seed'] = 16  # Default for unseeded teams
        
        self.team_stats = team_stats
        
    def predict_matchup(self, team1_id: int, team2_id: int) -> float:
        """Predict the probability of team1 beating team2 using empirical weights."""
        if team1_id not in self.team_stats or team2_id not in self.team_stats:
            return 0.5
            
        team1 = self.team_stats[team1_id]
        team2 = self.team_stats[team2_id]
        
        # Calculate feature differences
        features = {
            'win_pct': team1['win_pct'] - team2['win_pct'],
            'point_diff': team1['point_diff'] - team2['point_diff'],
            'strength_of_schedule': team1['strength_of_schedule'] - team2['strength_of_schedule'],
            'last_10_win_pct': team1['last_10_win_pct'] - team2['last_10_win_pct'],
            'home_win_pct': team1['home_win_pct'] - team2['home_win_pct'],
            'away_win_pct': team1['away_win_pct'] - team2['away_win_pct'],
            'neutral_win_pct': team1['neutral_win_pct'] - team2['neutral_win_pct'],
            'seed': team2['seed'] - team1['seed']  # Lower seed is better
        }
        
        # Calculate weighted score based on empirical weights
        score = 0
        for feature, diff in features.items():
            if feature in self.empirical_weights['weights']:
                score += self.empirical_weights['weights'][feature] * diff
        
        # Convert to probability using sigmoid function
        prob = 1 / (1 + np.exp(-5 * score))  # Scale factor of 5 for better spread
        
        # Clip probabilities to avoid extreme values
        return np.clip(prob, 0.025, 0.975)
        
    def generate_predictions(self, submission_file: str) -> pd.DataFrame:
        """Generate predictions for all matchups in the submission file."""
        sample_submission = pd.read_csv(submission_file)
        predictions = []
        
        for _, row in sample_submission.iterrows():
            season, team1_id, team2_id = map(int, row['ID'].split('_'))
            
            # Always predict probability for lower ID beating higher ID
            prob = self.predict_matchup(team1_id, team2_id)
            
            predictions.append({
                'ID': row['ID'],
                'Pred': prob
            })
            
        return pd.DataFrame(predictions)

def main():
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_path, 'march-machine-learning-mania-2025-final-dataset')
    
    # Process both tournaments
    for gender in ['M', 'W']:
        print(f"\nProcessing {('Men' if gender == 'M' else 'Women')}'s Tournament")
        
        # Initialize model with empirical weights
        weights_file = os.path.join(base_path, f'empirical_weights_{gender}.json')
        model = FinalMarchMadnessModel(data_path, weights_file)
        
        # Load current season data
        print("Loading 2025 season data...")
        model.load_current_season_data(2025, gender)
        
        # Generate predictions
        print("Generating predictions...")
        submission_file = os.path.join(data_path, 'SampleSubmissionStage1.csv')
        predictions = model.generate_predictions(submission_file)
        
        # Save predictions
        output_path = os.path.join(base_path, f'predictions_{gender}_2025_final.csv')
        predictions.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")
        
        # Print prediction statistics
        print("\nPrediction Statistics:")
        print(predictions['Pred'].describe())
        
        # Analyze prediction distribution
        bins = np.arange(0, 1.1, 0.1)
        hist, _ = np.histogram(predictions['Pred'], bins=bins)
        print("\nPrediction Distribution:")
        for i in range(len(hist)):
            print(f"{bins[i]:.1f}-{bins[i+1]:.1f}: {hist[i]}")

if __name__ == "__main__":
    main() 