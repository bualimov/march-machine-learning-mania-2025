import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import os

class MarchMadnessDataProcessor:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.regular_season_results = {}
        self.tournament_results = {}
        self.team_stats = {}
        self.rankings = {}
        self.coaches = None
        self.conferences = None
        
    def load_regular_season_data(self, start_year: int = 2003) -> None:
        """Load regular season data for both men's and women's tournaments."""
        # Men's regular season
        men_reg = pd.read_csv(os.path.join(self.data_path, 'MRegularSeasonDetailedResults.csv'))
        women_reg = pd.read_csv(os.path.join(self.data_path, 'WRegularSeasonDetailedResults.csv'))
        
        self.regular_season_results['M'] = men_reg[men_reg['Season'] >= start_year]
        self.regular_season_results['W'] = women_reg[women_reg['Season'] >= start_year]
        
    def load_tournament_data(self, start_year: int = 2003) -> None:
        """Load NCAA tournament data for both men's and women's tournaments."""
        men_tourney = pd.read_csv(os.path.join(self.data_path, 'MNCAATourneyDetailedResults.csv'))
        women_tourney = pd.read_csv(os.path.join(self.data_path, 'WNCAATourneyDetailedResults.csv'))
        
        self.tournament_results['M'] = men_tourney[men_tourney['Season'] >= start_year]
        self.tournament_results['W'] = women_tourney[women_tourney['Season'] >= start_year]
        
    def load_rankings(self) -> None:
        """Load Massey Ordinals rankings data."""
        self.rankings = pd.read_csv(os.path.join(self.data_path, 'MMasseyOrdinals.csv'))
        
    def load_coaches(self) -> None:
        """Load coaching data."""
        self.coaches = pd.read_csv(os.path.join(self.data_path, 'MTeamCoaches.csv'))
        
    def load_conferences(self) -> None:
        """Load conference data."""
        self.conferences = pd.read_csv(os.path.join(self.data_path, 'Conferences.csv'))
        
    def calculate_team_season_stats(self, season: int, gender: str) -> pd.DataFrame:
        """Calculate season-level statistics for each team."""
        data = self.regular_season_results[gender]
        season_data = data[data['Season'] == season]
        
        team_stats = []
        
        # Process winning team stats
        w_stats = season_data.groupby('WTeamID').agg({
            'WScore': ['mean', 'std'],
            'WFGM': 'mean',
            'WFGA': 'mean',
            'WFGM3': 'mean',
            'WFGA3': 'mean',
            'WFTM': 'mean',
            'WFTA': 'mean',
            'WOR': 'mean',
            'WDR': 'mean',
            'WAst': 'mean',
            'WTO': 'mean',
            'WStl': 'mean',
            'WBlk': 'mean'
        }).reset_index()
        
        # Process losing team stats
        l_stats = season_data.groupby('LTeamID').agg({
            'LScore': ['mean', 'std'],
            'LFGM': 'mean',
            'LFGA': 'mean',
            'LFGM3': 'mean',
            'LFGA3': 'mean',
            'LFTM': 'mean',
            'LFTA': 'mean',
            'LOR': 'mean',
            'LDR': 'mean',
            'LAst': 'mean',
            'LTO': 'mean',
            'LStl': 'mean',
            'LBlk': 'mean'
        }).reset_index()
        
        # Calculate win percentage
        wins = season_data['WTeamID'].value_counts().reset_index()
        wins.columns = ['TeamID', 'Wins']
        losses = season_data['LTeamID'].value_counts().reset_index()
        losses.columns = ['TeamID', 'Losses']
        
        # Merge all stats
        team_stats = pd.merge(wins, losses, on='TeamID', how='outer').fillna(0)
        team_stats['WinPct'] = team_stats['Wins'] / (team_stats['Wins'] + team_stats['Losses'])
        
        return team_stats
    
    def calculate_point_differential(self, season: int, gender: str) -> pd.DataFrame:
        """Calculate season point differential for each team."""
        data = self.regular_season_results[gender]
        season_data = data[data['Season'] == season]
        
        # Calculate point differential for winning teams
        w_diff = season_data.groupby('WTeamID')['WScore'].sum() - season_data.groupby('WTeamID')['LScore'].sum()
        l_diff = season_data.groupby('LTeamID')['LScore'].sum() - season_data.groupby('LTeamID')['WScore'].sum()
        
        # Combine and calculate total point differential
        point_diff = pd.DataFrame({
            'TeamID': pd.concat([w_diff, l_diff]).index.unique()
        })
        point_diff['PointDiff'] = 0
        
        for team in point_diff['TeamID']:
            w_points = w_diff[team] if team in w_diff else 0
            l_points = l_diff[team] if team in l_diff else 0
            point_diff.loc[point_diff['TeamID'] == team, 'PointDiff'] = w_points + l_points
            
        return point_diff
    
    def get_head_to_head(self, team1_id: int, team2_id: int, 
                        lookback_years: int = 5, gender: str = 'M') -> Dict:
        """Get head-to-head record between two teams over the past N years."""
        current_season = self.regular_season_results[gender]['Season'].max()
        data = self.regular_season_results[gender]
        recent_data = data[data['Season'] > current_season - lookback_years]
        
        # Games where team1 won
        team1_wins = len(recent_data[
            ((recent_data['WTeamID'] == team1_id) & (recent_data['LTeamID'] == team2_id))
        ])
        
        # Games where team2 won
        team2_wins = len(recent_data[
            ((recent_data['WTeamID'] == team2_id) & (recent_data['LTeamID'] == team1_id))
        ])
        
        return {
            'team1_wins': team1_wins,
            'team2_wins': team2_wins,
            'total_games': team1_wins + team2_wins
        } 