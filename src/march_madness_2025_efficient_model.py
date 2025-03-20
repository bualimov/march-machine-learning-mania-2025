import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import os
from sklearn.preprocessing import StandardScaler

class EfficientMarchMadnessModel:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.scaler = StandardScaler()
        self.team_stats = {}
        self.empirical_weights = None
        
    def load_data(self, start_year: int = 2019, gender: str = 'M'):
        """Load and preprocess only the necessary data."""
        print(f"Loading {gender} data from {start_year}-present...")
        
        # Load only recent data
        regular_season = pd.read_csv(
            os.path.join(self.data_path, f'{gender}RegularSeasonCompactResults.csv')
        )
        regular_season = regular_season[regular_season['Season'] >= start_year]
        
        # Pre-calculate team stats for each season
        for season in regular_season['Season'].unique():
            season_games = regular_season[regular_season['Season'] == season]
            
            # Calculate stats for all teams in one pass
            team_stats = {}
            
            # Process winning team stats
            for _, game in season_games.iterrows():
                # Update winner stats
                if game['WTeamID'] not in team_stats:
                    team_stats[game['WTeamID']] = {
                        'Wins': 0, 'Losses': 0,
                        'PointsFor': 0, 'PointsAgainst': 0
                    }
                if game['LTeamID'] not in team_stats:
                    team_stats[game['LTeamID']] = {
                        'Wins': 0, 'Losses': 0,
                        'PointsFor': 0, 'PointsAgainst': 0
                    }
                    
                # Update stats
                team_stats[game['WTeamID']]['Wins'] += 1
                team_stats[game['WTeamID']]['PointsFor'] += game['WScore']
                team_stats[game['WTeamID']]['PointsAgainst'] += game['LScore']
                
                team_stats[game['LTeamID']]['Losses'] += 1
                team_stats[game['LTeamID']]['PointsFor'] += game['LScore']
                team_stats[game['LTeamID']]['PointsAgainst'] += game['WScore']
            
            # Calculate derived stats
            for team_id in team_stats:
                games_played = team_stats[team_id]['Wins'] + team_stats[team_id]['Losses']
                team_stats[team_id]['WinPct'] = team_stats[team_id]['Wins'] / games_played
                team_stats[team_id]['PointDiff'] = (
                    team_stats[team_id]['PointsFor'] - team_stats[team_id]['PointsAgainst']
                ) / games_played
            
            self.team_stats[(season, gender)] = team_stats
            
        # Load rankings if available
        if gender == 'M':
            rankings = pd.read_csv(os.path.join(self.data_path, 'MMasseyOrdinals.csv'))
            rankings = rankings[
                (rankings['Season'] >= start_year) & 
                (rankings['RankingDayNum'] == 133)
            ]
            self.rankings = rankings.groupby(['Season', 'TeamID'])['OrdinalRank'].mean()
        else:
            self.rankings = None
            
    def predict_matchup(self, team1_id: int, team2_id: int, 
                       season: int, gender: str) -> float:
        """Efficiently predict matchup probability using pre-calculated stats."""
        # Get team stats
        team1_stats = self.team_stats.get((season, gender), {}).get(team1_id, {})
        team2_stats = self.team_stats.get((season, gender), {}).get(team2_id, {})
        
        if not team1_stats or not team2_stats:
            return 0.5  # Default to 50% if we don't have data
        
        # Calculate feature differences
        win_pct_diff = team1_stats['WinPct'] - team2_stats['WinPct']
        point_diff_diff = team1_stats['PointDiff'] - team2_stats['PointDiff']
        
        # Get ranking difference if available
        if self.rankings is not None and gender == 'M':
            try:
                rank1 = self.rankings.get((season, team1_id), 400)
                rank2 = self.rankings.get((season, team2_id), 400)
                rank_diff = rank2 - rank1  # Positive means team1 is better ranked
            except:
                rank_diff = 0
        else:
            rank_diff = 0
            
        # Apply empirical weights (these would come from feature_importance analysis)
        win_weight = 0.3
        point_diff_weight = 0.3
        rank_weight = 0.4
        
        # Calculate probability
        score = (
            win_weight * win_pct_diff +
            point_diff_weight * (point_diff_diff / 20) +  # Normalize point diff
            rank_weight * (rank_diff / 400)  # Normalize ranking
        )
        
        # Convert to probability using sigmoid function
        prob = 1 / (1 + np.exp(-score))
        
        # Clip probability to avoid extreme values
        return np.clip(prob, 0.025, 0.975)
    
    def generate_predictions(self, season: int, gender: str) -> pd.DataFrame:
        """Generate predictions for all possible matchups."""
        # Load sample submission to get required matchups
        sample_submission = pd.read_csv(
            os.path.join(self.data_path, 'SampleSubmissionStage1.csv')
        )
        
        predictions = []
        for _, row in sample_submission.iterrows():
            season, team1_id, team2_id = map(int, row['ID'].split('_'))
            
            # Always predict probability for lower ID beating higher ID
            prob = self.predict_matchup(team1_id, team2_id, season, gender)
            
            predictions.append({
                'ID': row['ID'],
                'Pred': prob
            })
            
        return pd.DataFrame(predictions)

def main():
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_path, 'march-machine-learning-mania-2025-final-dataset')
    
    # First, analyze feature importance
    print("Analyzing historical tournament outcomes...")
    os.system(f'python3 {os.path.join(base_path, "src", "march_madness_2025_feature_importance.py")}')
    
    # Create and run efficient model
    model = EfficientMarchMadnessModel(data_path)
    
    for gender in ['M', 'W']:
        print(f"\nProcessing {('Men' if gender == 'M' else 'Women')}'s Tournament")
        
        # Load only recent data (2019-present)
        model.load_data(start_year=2019, gender=gender)
        
        # Generate predictions
        print("Generating predictions...")
        predictions = model.generate_predictions(2025, gender)
        
        # Save predictions
        output_path = os.path.join(base_path, f'predictions_{gender}_2025.csv')
        predictions.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")
        
        # Print prediction stats
        print("\nPrediction Statistics:")
        print(predictions['Pred'].describe())

if __name__ == "__main__":
    main() 