import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

class MarchMadnessFeatureEngineer:
    def __init__(self, data_processor):
        self.data_processor = data_processor
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        self.feature_columns = [
            'WinPctDiff',
            'PointDiffDiff',
            'RankDiff',
            'H2H_Ratio',
            'ConfStrengthDiff'
        ]
        
    def create_team_features(self, season: int, gender: str) -> pd.DataFrame:
        """Create comprehensive feature set for each team."""
        # Get basic season stats
        season_stats = self.data_processor.calculate_team_season_stats(season, gender)
        
        # Get point differential
        point_diff = self.data_processor.calculate_point_differential(season, gender)
        
        # Get rankings if available (men's only)
        if gender == 'M' and self.data_processor.rankings is not None:
            season_rankings = self.data_processor.rankings[
                (self.data_processor.rankings['Season'] == season) &
                (self.data_processor.rankings['RankingDayNum'] == 133)  # Final pre-tournament rankings
            ]
            
            # Average ranking across all systems
            avg_rankings = season_rankings.groupby('TeamID')['OrdinalRank'].mean().reset_index()
            avg_rankings.columns = ['TeamID', 'AvgRank']
        else:
            avg_rankings = pd.DataFrame(columns=['TeamID', 'AvgRank'])
            
        # Merge all features
        features = pd.merge(season_stats, point_diff, on='TeamID', how='left')
        features = pd.merge(features, avg_rankings, on='TeamID', how='left')
        
        # Fill missing values for rankings with a high number (worse than last place)
        features['AvgRank'] = features['AvgRank'].fillna(400)  # Assuming no team is ranked worse than 400
        
        # Add conference strength (if available)
        if self.data_processor.conferences is not None:
            conf_features = self._calculate_conference_strength(season, gender)
            features = pd.merge(features, conf_features, on='TeamID', how='left')
            features['ConfStrength'] = features['ConfStrength'].fillna(0)  # Fill missing conference strength with 0
            
        # Fill any remaining missing values with 0
        features = features.fillna(0)
            
        return features
    
    def _calculate_conference_strength(self, season: int, gender: str) -> pd.DataFrame:
        """Calculate conference strength metrics."""
        if gender == 'M':
            conf_file = 'MTeamConferences.csv'
        else:
            conf_file = 'WTeamConferences.csv'
            
        team_conferences = pd.read_csv(f"{self.data_processor.data_path}/{conf_file}")
        season_confs = team_conferences[team_conferences['Season'] == season]
        
        # Get regular season results for conference games
        reg_season = self.data_processor.regular_season_results[gender]
        season_games = reg_season[reg_season['Season'] == season]
        
        # Merge conference info
        games_w_conf = pd.merge(season_games, 
                              season_confs[['TeamID', 'ConfAbbrev']], 
                              left_on='WTeamID', 
                              right_on='TeamID')
        games_w_conf = pd.merge(games_w_conf,
                               season_confs[['TeamID', 'ConfAbbrev']],
                               left_on='LTeamID',
                               right_on='TeamID',
                               suffixes=('_winner', '_loser'))
        
        # Calculate conference win rates
        conf_stats = games_w_conf.groupby('ConfAbbrev_winner').agg({
            'WTeamID': 'count',  # Number of wins
            'WScore': 'mean',    # Average points scored
            'LScore': 'mean'     # Average points allowed
        }).reset_index()
        
        conf_stats['PointDiff'] = conf_stats['WScore'] - conf_stats['LScore']
        
        # Map back to teams
        team_conf_strength = pd.merge(season_confs,
                                    conf_stats,
                                    left_on='ConfAbbrev',
                                    right_on='ConfAbbrev_winner',
                                    how='left')
        
        return team_conf_strength[['TeamID', 'PointDiff']].rename(
            columns={'PointDiff': 'ConfStrength'}
        )
    
    def create_matchup_features(self, team1_id: int, team2_id: int, 
                              season: int, gender: str) -> pd.DataFrame:
        """Create features for a specific matchup between two teams."""
        # Get team features
        team_features = self.create_team_features(season, gender)
        
        # Get head-to-head history
        h2h = self.data_processor.get_head_to_head(team1_id, team2_id, gender=gender)
        
        # Create matchup features
        team1_stats = team_features[team_features['TeamID'] == team1_id]
        team2_stats = team_features[team_features['TeamID'] == team2_id]
        
        # Handle cases where team data is missing
        if len(team1_stats) == 0 or len(team2_stats) == 0:
            # Create default features for unknown teams
            matchup_features = {
                'WinPctDiff': 0,
                'PointDiffDiff': 0,
                'RankDiff': 0,
                'H2H_Ratio': 0.5,
                'ConfStrengthDiff': 0
            }
        else:
            team1_stats = team1_stats.iloc[0]
            team2_stats = team2_stats.iloc[0]
            
            matchup_features = {
                'WinPctDiff': team1_stats['WinPct'] - team2_stats['WinPct'],
                'PointDiffDiff': team1_stats['PointDiff'] - team2_stats['PointDiff'],
                'RankDiff': team1_stats.get('AvgRank', 400) - team2_stats.get('AvgRank', 400),
                'H2H_Ratio': h2h['team1_wins'] / (h2h['total_games'] + 1),  # Add 1 to avoid division by zero
                'ConfStrengthDiff': team1_stats.get('ConfStrength', 0) - team2_stats.get('ConfStrength', 0)
            }
            
        # Ensure we only return the features we want to use
        features_df = pd.DataFrame([matchup_features])
        return features_df[self.feature_columns]
    
    def prepare_training_data(self, start_year: int, end_year: int, 
                            gender: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare historical matchup data for model training."""
        X = []
        y = []
        
        # Use tournament games for training
        tourney_games = self.data_processor.tournament_results[gender]
        
        for season in range(start_year, end_year + 1):
            season_games = tourney_games[tourney_games['Season'] == season]
            
            for _, game in season_games.iterrows():
                # Create features for this matchup
                features = self.create_matchup_features(
                    game['WTeamID'], game['LTeamID'], season, gender
                )
                
                X.append(features)
                y.append(1)  # Winner
                
                # Also add the reverse matchup with opposite outcome
                features_reverse = self.create_matchup_features(
                    game['LTeamID'], game['WTeamID'], season, gender
                )
                
                X.append(features_reverse)
                y.append(0)  # Loser
                
        X = pd.concat(X, ignore_index=True)
        y = pd.Series(y)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return pd.DataFrame(X_scaled, columns=self.feature_columns), y 