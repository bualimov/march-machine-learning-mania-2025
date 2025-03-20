import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import os
import json
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

class ImprovedMarchMadnessModel:
    def __init__(self, data_path: str, analysis_results_path: str):
        self.data_path = data_path
        self.analysis_results_path = analysis_results_path
        self.team_stats = {}
        self.conference_strengths = {}
        self.ranking_weights = {}
        self.team_consistency = {}
        self.load_analysis_results()
        
    def load_analysis_results(self):
        """Load pre-computed analysis results."""
        with open(self.analysis_results_path, 'r') as f:
            self.analysis_results = json.load(f)
            
        self.weights = self.analysis_results['weights']
        self.conference_strengths = self.analysis_results['conference_strength']
        self.ranking_weights = self.analysis_results['ranking_systems']
        self.team_consistency = self.analysis_results['team_consistency']
        
    def load_current_season_data(self, season: int, gender: str = 'M'):
        """Load and process current season data."""
        # Load necessary data files
        self.regular_season = pd.read_csv(os.path.join(self.data_path, f'{gender}RegularSeasonCompactResults.csv'))
        
        # Try to load rankings if available
        ranking_file = os.path.join(self.data_path, f'{gender}MasseyOrdinals.csv')
        self.rankings = pd.read_csv(ranking_file) if os.path.exists(ranking_file) else None
        
        self.conferences = pd.read_csv(os.path.join(self.data_path, f'{gender}TeamConferences.csv'))
        
        # Filter for current season
        self.regular_season = self.regular_season[self.regular_season['Season'] == season]
        if self.rankings is not None:
            self.rankings = self.rankings[
                (self.rankings['Season'] == season) &
                (self.rankings['RankingDayNum'] == 133)  # End of regular season
            ]
        self.conferences = self.conferences[self.conferences['Season'] == season]
        
        # Pre-calculate team statistics
        self._calculate_team_stats()
        
    def _calculate_team_stats(self):
        """Calculate comprehensive team statistics."""
        team_stats = defaultdict(dict)
        
        # Process regular season results
        for _, game in self.regular_season.iterrows():
            # Winner stats
            if game['WTeamID'] not in team_stats:
                team_stats[game['WTeamID']] = {
                    'wins': 0, 'losses': 0,
                    'points_for': 0, 'points_against': 0,
                    'home_wins': 0, 'away_wins': 0, 'neutral_wins': 0,
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
            stats['road_win_pct'] = stats['away_wins'] / games_played
            stats['neutral_win_pct'] = stats['neutral_wins'] / games_played
            
            # Get average ranking across systems if available
            if self.rankings is not None:
                team_rankings = self.rankings[self.rankings['TeamID'] == team_id]
                if not team_rankings.empty:
                    stats['avg_ranking'] = team_rankings['OrdinalRank'].mean()
                else:
                    stats['avg_ranking'] = 400  # Default for unranked teams
            else:
                stats['avg_ranking'] = None  # No rankings available
            
            # Add conference strength
            stats['conf_strength'] = self.conference_strengths.get(stats['conference'], 0.5)
            
            # Add historical consistency if available
            if str(team_id) in self.team_consistency:
                stats['historical_consistency'] = self.team_consistency[str(team_id)]['consistency']
                stats['tournament_experience'] = self.team_consistency[str(team_id)]['experience']
            else:
                stats['historical_consistency'] = 0
                stats['tournament_experience'] = 0
        
        self.team_stats = team_stats
        
    def predict_matchup(self, team1_id: int, team2_id: int) -> float:
        """Predict the probability of team1 beating team2."""
        if team1_id not in self.team_stats or team2_id not in self.team_stats:
            return 0.5
            
        team1 = self.team_stats[team1_id]
        team2 = self.team_stats[team2_id]
        
        # Calculate feature differences
        features = {
            'win_pct': team1['win_pct'] - team2['win_pct'],
            'point_differential': (team1['point_diff'] - team2['point_diff']) / 20,  # Normalize
            'conference_strength': team1['conf_strength'] - team2['conf_strength'],
            'consistency': team1['historical_consistency'] - team2['historical_consistency'],
            'experience': (team1['tournament_experience'] - team2['tournament_experience']) / 10
        }
        
        # Add ranking difference if available
        if team1['avg_ranking'] is not None and team2['avg_ranking'] is not None:
            features['ranking'] = (team2['avg_ranking'] - team1['avg_ranking']) / 400  # Normalize, lower is better
        
        # Calculate weighted score based on available features
        score = 0
        if 'ranking' in features:
            score = (
                self.weights['point_differential'] * features['point_differential'] +
                self.weights['rankings'] * features['ranking'] +
                self.weights['conference_strength'] * features['conference_strength'] +
                self.weights['team_consistency'] * (
                    0.7 * features['consistency'] + 0.3 * features['experience']
                )
            )
        else:
            # Use women's tournament weights (no rankings)
            score = (
                self.weights['point_differential'] * features['point_differential'] +
                self.weights['conference_strength'] * features['conference_strength'] +
                self.weights['team_consistency'] * (
                    0.7 * features['consistency'] + 0.3 * features['experience']
                )
            )
        
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
        
        # Initialize model with analysis results
        analysis_file = os.path.join(base_path, f'analysis_results_{gender}.json')
        model = ImprovedMarchMadnessModel(data_path, analysis_file)
        
        # Load current season data
        print("Loading 2025 season data...")
        model.load_current_season_data(2025, gender)
        
        # Generate predictions
        print("Generating predictions...")
        submission_file = os.path.join(data_path, 'SampleSubmissionStage1.csv')
        predictions = model.generate_predictions(submission_file)
        
        # Save predictions
        output_path = os.path.join(base_path, f'predictions_{gender}_2025_improved.csv')
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