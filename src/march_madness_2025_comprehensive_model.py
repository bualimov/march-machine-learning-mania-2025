import pandas as pd
import numpy as np
import os
import json
from collections import defaultdict
from typing import Dict, List, Tuple
from sklearn.preprocessing import MinMaxScaler

class ComprehensiveMarchMadnessModel:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.team_stats = {}
        self.team_history = {}
        self.coach_records = {}
        self.conference_strength = {}
        self.current_season = 2025
        self.time_decay_factor = 0.9  # For historical weighting (recency)
        
    def load_all_data(self, gender: str = 'M'):
        """Load all necessary data files."""
        print(f"Loading data for {gender} tournament...")
        
        # Current season data
        self.regular_season = pd.read_csv(os.path.join(self.data_path, f'{gender}RegularSeasonCompactResults.csv'))
        self.seeds = pd.read_csv(os.path.join(self.data_path, f'{gender}NCAATourneySeeds.csv'))
        self.teams = pd.read_csv(os.path.join(self.data_path, f'{gender}Teams.csv'))
        self.conferences = pd.read_csv(os.path.join(self.data_path, f'{gender}TeamConferences.csv'))
        
        # Historical data
        self.tourney_results = pd.read_csv(os.path.join(self.data_path, f'{gender}NCAATourneyCompactResults.csv'))
        
        # Try to load coaches data if available
        try:
            self.coaches = pd.read_csv(os.path.join(self.data_path, f'{gender}TeamCoaches.csv'))
            self.has_coaches_data = True
        except FileNotFoundError:
            print(f"Coaches data not found for {gender} tournament. Proceeding without coach success rates.")
            self.has_coaches_data = False
            
        # Try to load rankings data if available
        try:
            self.rankings = pd.read_csv(os.path.join(self.data_path, f'{gender}MasseyOrdinals.csv'))
            self.has_rankings_data = True
        except FileNotFoundError:
            print(f"Rankings data not found for {gender} tournament. Proceeding without ranking information.")
            self.has_rankings_data = False
        
        self.gender = gender
        
        # Process data
        self.analyze_historical_performance()
        self.calculate_conference_strength()
        if self.has_coaches_data:
            self.analyze_coach_performance()
        self.calculate_current_season_stats()
        
    def analyze_historical_performance(self):
        """Analyze historical performance of teams, with recency weighting."""
        print("Analyzing historical team performance...")
        team_history = defaultdict(lambda: {
            'tournament_appearances': [],
            'tournament_wins': defaultdict(int),
            'tournament_results': defaultdict(list),
            'consistency_score': 0,
            'recent_success': 0
        })
        
        # Get all possible seasons from the data
        all_seasons = sorted(self.tourney_results['Season'].unique())
        min_season = min(all_seasons)
        max_season = max(all_seasons)
        
        # Calculate tournament appearances and wins for each team
        for _, row in self.tourney_results.iterrows():
            season = row['Season']
            # Add winner stats
            team_history[row['WTeamID']]['tournament_appearances'].append(season)
            team_history[row['WTeamID']]['tournament_wins'][season] += 1
            team_history[row['WTeamID']]['tournament_results'][season].append(1)  # 1 for win
            
            # Add loser appearance
            team_history[row['LTeamID']]['tournament_appearances'].append(season)
            team_history[row['LTeamID']]['tournament_results'][season].append(0)  # 0 for loss
        
        # Calculate team consistency and recency scores
        for team_id, history in team_history.items():
            # Sort appearances
            history['tournament_appearances'] = sorted(set(history['tournament_appearances']))
            
            # Calculate consistency (frequency of appearances)
            total_possible_seasons = max_season - min_season + 1
            appearance_rate = len(history['tournament_appearances']) / total_possible_seasons
            
            # Calculate recency-weighted success
            recent_success = 0
            total_weight = 0
            for season in range(min_season, max_season + 1):
                # Exponential decay weight based on recency
                season_weight = self.time_decay_factor ** (max_season - season)
                total_weight += season_weight
                
                if season in history['tournament_wins']:
                    # More weight for wins in later rounds (approximated by total wins in season)
                    recent_success += history['tournament_wins'][season] * season_weight
            
            if total_weight > 0:
                history['recent_success'] = recent_success / total_weight
                
            # Overall consistency score combines appearance rate and recent success
            history['consistency_score'] = 0.4 * appearance_rate + 0.6 * (history['recent_success'] if total_weight > 0 else 0)
        
        self.team_history = team_history
    
    def calculate_conference_strength(self):
        """Calculate conference strength based on tournament performance."""
        print("Calculating conference strength...")
        conference_performance = defaultdict(lambda: {
            'tournament_wins': 0,
            'tournament_appearances': 0,
            'total_games': 0,
            'win_rate': 0,
            'weighted_wins': 0
        })
        
        # Get all possible seasons from the data
        max_season = max(self.tourney_results['Season'].unique())
        
        # Calculate conference performance in tournaments
        for _, game in self.tourney_results.iterrows():
            season = game['Season']
            season_weight = self.time_decay_factor ** (max_season - season)
            
            # Get conferences for winner and loser
            winner_conf = self.conferences[
                (self.conferences['Season'] == season) & 
                (self.conferences['TeamID'] == game['WTeamID'])
            ]
            
            loser_conf = self.conferences[
                (self.conferences['Season'] == season) & 
                (self.conferences['TeamID'] == game['LTeamID'])
            ]
            
            if not winner_conf.empty and not loser_conf.empty:
                winner_conf_name = winner_conf['ConfAbbrev'].iloc[0]
                loser_conf_name = loser_conf['ConfAbbrev'].iloc[0]
                
                # Add win for winner's conference
                conference_performance[winner_conf_name]['tournament_wins'] += 1
                conference_performance[winner_conf_name]['weighted_wins'] += season_weight
                conference_performance[winner_conf_name]['total_games'] += 1
                
                # Add game for loser's conference
                conference_performance[loser_conf_name]['total_games'] += 1
        
        # Calculate win rates and normalize strengths
        for conf, stats in conference_performance.items():
            if stats['total_games'] > 0:
                stats['win_rate'] = stats['tournament_wins'] / stats['total_games']
        
        # Calculate tournament appearance rates by conference
        conf_appearances = defaultdict(set)
        for _, row in self.seeds.iterrows():
            season = row['Season']
            team_id = row['TeamID']
            
            # Get conference for this team and season
            conf_row = self.conferences[
                (self.conferences['Season'] == season) & 
                (self.conferences['TeamID'] == team_id)
            ]
            
            if not conf_row.empty:
                conf_name = conf_row['ConfAbbrev'].iloc[0]
                conf_appearances[conf_name].add(season)
        
        # Add appearance counts to conference stats
        for conf, seasons in conf_appearances.items():
            if conf in conference_performance:
                conference_performance[conf]['tournament_appearances'] = len(seasons)
        
        # Normalize conference strength to 0-1 scale
        strengths = [stats['win_rate'] for _, stats in conference_performance.items() if stats['total_games'] > 0]
        if strengths:
            min_strength = min(strengths)
            max_strength = max(strengths)
            strength_range = max_strength - min_strength
            
            for conf, stats in conference_performance.items():
                if stats['total_games'] > 0 and strength_range > 0:
                    stats['normalized_strength'] = (stats['win_rate'] - min_strength) / strength_range
                else:
                    stats['normalized_strength'] = 0.5
        
        self.conference_strength = conference_performance
        
        # Get current conferences for all teams
        current_conferences = self.conferences[self.conferences['Season'] == self.current_season]
        self.team_conferences = {row['TeamID']: row['ConfAbbrev'] for _, row in current_conferences.iterrows()}
    
    def analyze_coach_performance(self):
        """Analyze coach performance and success rates."""
        print("Analyzing coach performance...")
        coach_records = defaultdict(lambda: {
            'tournament_appearances': 0,
            'tournament_wins': 0,
            'win_rate': 0,
            'experience_years': 0
        })
        
        if not self.has_coaches_data:
            return
            
        # Get current coaches for teams in 2025
        current_coaches = self.coaches[self.coaches['Season'] == self.current_season]
        self.team_coaches = {row['TeamID']: row['CoachName'] for _, row in current_coaches.iterrows()}
        
        # Calculate tournament success for each coach
        for _, game in self.tourney_results.iterrows():
            season = game['Season']
            
            # Get coaches for this game
            winner_coach = self.coaches[
                (self.coaches['Season'] == season) & 
                (self.coaches['TeamID'] == game['WTeamID'])
            ]
            
            loser_coach = self.coaches[
                (self.coaches['Season'] == season) & 
                (self.coaches['TeamID'] == game['LTeamID'])
            ]
            
            if not winner_coach.empty:
                coach_name = winner_coach['CoachName'].iloc[0]
                coach_records[coach_name]['tournament_wins'] += 1
            
            # Count appearances for both coaches
            for coach_df in [winner_coach, loser_coach]:
                if not coach_df.empty:
                    coach_name = coach_df['CoachName'].iloc[0]
                    coach_records[coach_name]['tournament_appearances'] += 1
        
        # Calculate unique tournament appearances and win rates
        for coach_name, stats in coach_records.items():
            # Calculate experience (years coaching)
            coach_seasons = self.coaches[self.coaches['CoachName'] == coach_name]['Season'].unique()
            stats['experience_years'] = len(coach_seasons)
            
            # Calculate win rate
            if stats['tournament_appearances'] > 0:
                stats['win_rate'] = stats['tournament_wins'] / stats['tournament_appearances']
        
        self.coach_records = coach_records
    
    def calculate_current_season_stats(self):
        """Calculate comprehensive statistics for the current season."""
        print("Calculating current season statistics...")
        team_stats = defaultdict(dict)
        
        # Filter for current season data
        current_season_games = self.regular_season[self.regular_season['Season'] == self.current_season]
        
        # Initialize stats for all teams that appear in the current season
        all_teams = set(current_season_games['WTeamID']).union(set(current_season_games['LTeamID']))
        
        for team_id in all_teams:
            team_stats[team_id] = {
                'wins': 0, 'losses': 0,
                'points_for': 0, 'points_against': 0,
                'home_wins': 0, 'away_wins': 0, 'neutral_wins': 0,
                'opponents_wins': [],
                'last_10_games': {'wins': 0, 'losses': 0},
                'seed': 16  # Default seed for unseeded teams
            }
            
            # Add conference information if available
            if team_id in self.team_conferences:
                team_stats[team_id]['conference'] = self.team_conferences[team_id]
            
            # Add coach information if available
            if self.has_coaches_data and team_id in self.team_coaches:
                team_stats[team_id]['coach'] = self.team_coaches[team_id]
            
            # Add historical performance if available
            if team_id in self.team_history:
                team_stats[team_id]['consistency_score'] = self.team_history[team_id]['consistency_score']
                team_stats[team_id]['recent_success'] = self.team_history[team_id]['recent_success']
        
        # Process current season games
        for _, game in current_season_games.iterrows():
            # Update winner stats
            w_stats = team_stats[game['WTeamID']]
            w_stats['wins'] += 1
            w_stats['points_for'] += game['WScore']
            w_stats['points_against'] += game['LScore']
            
            # Update loser stats
            l_stats = team_stats[game['LTeamID']]
            l_stats['losses'] += 1
            l_stats['points_for'] += game['LScore']
            l_stats['points_against'] += game['WScore']
            
            # Track location wins
            if game['WLoc'] == 'H':
                w_stats['home_wins'] += 1
            elif game['WLoc'] == 'A':
                w_stats['away_wins'] += 1
            else:
                w_stats['neutral_wins'] += 1
            
            # Track opponent wins for strength of schedule
            w_stats['opponents_wins'].append(game['LTeamID'])
            l_stats['opponents_wins'].append(game['WTeamID'])
        
        # Calculate derived statistics for all teams
        for team_id, stats in team_stats.items():
            games_played = stats['wins'] + stats['losses']
            if games_played > 0:
                stats['win_pct'] = stats['wins'] / games_played
                stats['point_diff'] = (stats['points_for'] - stats['points_against']) / games_played
                stats['points_per_game'] = stats['points_for'] / games_played
                stats['points_allowed_per_game'] = stats['points_against'] / games_played
                
                if 'conference' in stats and stats['conference'] in self.conference_strength:
                    stats['conference_strength'] = self.conference_strength[stats['conference']].get('normalized_strength', 0.5)
                else:
                    stats['conference_strength'] = 0.5
                
                if 'coach' in stats and stats['coach'] in self.coach_records:
                    coach_stats = self.coach_records[stats['coach']]
                    stats['coach_win_rate'] = coach_stats['win_rate']
                    stats['coach_experience'] = coach_stats['experience_years']
                else:
                    stats['coach_win_rate'] = 0.5
                    stats['coach_experience'] = 0
                
                # Set defaults for missing historical data
                if 'consistency_score' not in stats:
                    stats['consistency_score'] = 0
                if 'recent_success' not in stats:
                    stats['recent_success'] = 0
            
            # Calculate last 10 games performance
            team_games = current_season_games[
                (current_season_games['WTeamID'] == team_id) | 
                (current_season_games['LTeamID'] == team_id)
            ].sort_values('DayNum').tail(10)
            
            if not team_games.empty:
                stats['last_10_games']['wins'] = len(team_games[team_games['WTeamID'] == team_id])
                stats['last_10_games']['losses'] = len(team_games[team_games['LTeamID'] == team_id])
                stats['last_10_win_pct'] = stats['last_10_games']['wins'] / len(team_games)
            else:
                stats['last_10_win_pct'] = 0
            
            # Get tournament seed for current season
            team_seed = self.seeds[
                (self.seeds['Season'] == self.current_season) & 
                (self.seeds['TeamID'] == team_id)
            ]
            
            if not team_seed.empty:
                seed_str = team_seed['Seed'].iloc[0]
                stats['seed'] = int(seed_str[1:3])
            
            # Get ranking if available
            if self.has_rankings_data:
                # Get the most recent Massey ranking before the tournament
                system_ranking = 'MAS'  # Use MAS (Massey) ranking system
                current_rankings = self.rankings[
                    (self.rankings['Season'] == self.current_season) & 
                    (self.rankings['TeamID'] == team_id) & 
                    (self.rankings['SystemName'] == system_ranking)
                ].sort_values('RankingDayNum', ascending=False)
                
                if not current_rankings.empty:
                    stats['ranking'] = current_rankings['OrdinalRank'].iloc[0]
                else:
                    # If Massey not available, try another ranking system
                    any_ranking = self.rankings[
                        (self.rankings['Season'] == self.current_season) & 
                        (self.rankings['TeamID'] == team_id)
                    ].sort_values(['RankingDayNum', 'SystemName'], ascending=[False, True])
                    
                    if not any_ranking.empty:
                        stats['ranking'] = any_ranking['OrdinalRank'].iloc[0]
                    else:
                        # Use seed as fallback if available, otherwise high rank
                        stats['ranking'] = stats['seed'] * 4 if 'seed' in stats else 100
            else:
                # Use seed as fallback
                stats['ranking'] = stats['seed'] * 4
        
        # Calculate strength of schedule for each team
        for team_id, stats in team_stats.items():
            # Average win percentage of opponents
            opponent_win_pcts = []
            for opp_id in stats['opponents_wins']:
                if opp_id in team_stats and team_stats[opp_id]['wins'] + team_stats[opp_id]['losses'] > 0:
                    opp_win_pct = team_stats[opp_id]['wins'] / (team_stats[opp_id]['wins'] + team_stats[opp_id]['losses'])
                    opponent_win_pcts.append(opp_win_pct)
            
            stats['strength_of_schedule'] = np.mean(opponent_win_pcts) if opponent_win_pcts else 0.5
            
            # Remove the temporary list of opponent IDs
            del stats['opponents_wins']
        
        self.team_stats = team_stats
    
    def normalize_features(self):
        """Normalize all numerical features for fair comparison."""
        print("Normalizing features...")
        
        # Extract all teams' stat values for each feature
        features = {}
        numerical_features = [
            'win_pct', 'point_diff', 'points_per_game', 'points_allowed_per_game',
            'strength_of_schedule', 'last_10_win_pct', 'consistency_score',
            'recent_success', 'conference_strength', 'coach_win_rate',
            'coach_experience', 'ranking', 'seed'
        ]
        
        for feature in numerical_features:
            features[feature] = []
            for team_id, stats in self.team_stats.items():
                if feature in stats:
                    features[feature].append(stats[feature])
        
        # Normalize each feature
        for feature in numerical_features:
            if features[feature]:
                if feature == 'ranking' or feature == 'seed' or feature == 'points_allowed_per_game':
                    # Lower is better for these features
                    feature_min = min(features[feature])
                    feature_max = max(features[feature])
                    if feature_max > feature_min:
                        for team_id, stats in self.team_stats.items():
                            if feature in stats:
                                stats[f'norm_{feature}'] = 1 - ((stats[feature] - feature_min) / (feature_max - feature_min))
                else:
                    # Higher is better
                    feature_min = min(features[feature])
                    feature_max = max(features[feature])
                    if feature_max > feature_min:
                        for team_id, stats in self.team_stats.items():
                            if feature in stats:
                                stats[f'norm_{feature}'] = (stats[feature] - feature_min) / (feature_max - feature_min)
    
    def predict_matchup(self, team1_id: int, team2_id: int) -> float:
        """Predict probability of team1 beating team2."""
        if team1_id not in self.team_stats or team2_id not in self.team_stats:
            return 0.5
        
        team1 = self.team_stats[team1_id]
        team2 = self.team_stats[team2_id]
        
        # Feature weights based on empirical analysis
        feature_weights = {
            'norm_seed': 0.22,                    # Tournament seeding
            'norm_ranking': 0.13,                 # End of season ranking
            'norm_point_diff': 0.14,              # Point differential
            'norm_win_pct': 0.11,                 # Win percentage
            'norm_conference_strength': 0.09,     # Conference strength
            'norm_strength_of_schedule': 0.08,    # Strength of schedule
            'norm_last_10_win_pct': 0.06,         # Recent form
            'norm_recent_success': 0.05,          # Historical tournament success
            'norm_consistency_score': 0.04,       # Tournament appearance consistency
            'norm_coach_win_rate': 0.05,          # Coach's tournament success
            'norm_coach_experience': 0.03         # Coach's experience
        }
        
        score = 0
        
        # Calculate weighted score based on feature differences
        for feature, weight in feature_weights.items():
            if feature in team1 and feature in team2:
                diff = team1[feature] - team2[feature]
                score += weight * diff
        
        # Convert score to probability using sigmoid function
        prob = 1 / (1 + np.exp(-5 * score))  # Scale factor adjusts steepness
        
        # Clip probabilities to avoid extreme values but allow reasonable range
        return np.clip(prob, 0.15, 0.85)
    
    def generate_predictions(self, gender: str) -> pd.DataFrame:
        """Generate predictions for all possible matchups in the 2025 tournament."""
        print(f"Generating predictions for {gender} tournament...")
        
        # Get all teams with seeds in 2025
        tournament_teams = self.seeds[self.seeds['Season'] == self.current_season]['TeamID'].unique()
        
        predictions = []
        for i, team1_id in enumerate(tournament_teams):
            for team2_id in tournament_teams[i+1:]:
                if team1_id < team2_id:
                    prob = self.predict_matchup(team1_id, team2_id)
                else:
                    prob = 1 - self.predict_matchup(team2_id, team1_id)
                
                predictions.append({
                    'ID': f"2025_{team1_id}_{team2_id}",
                    'Pred': prob
                })
        
        return pd.DataFrame(predictions)

def main():
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_path, 'march-machine-learning-mania-2025-final-dataset')
    
    # Process both tournaments
    for gender in ['M', 'W']:
        gender_name = 'Men' if gender == 'M' else 'Women'
        print(f"\nProcessing {gender_name}'s Tournament for 2025")
        
        # Initialize and train model
        model = ComprehensiveMarchMadnessModel(data_path)
        model.load_all_data(gender)
        model.normalize_features()
        
        # Generate predictions
        predictions = model.generate_predictions(gender)
        
        # Save predictions
        output_path = os.path.join(base_path, f'predictions_{gender}_2025_only.csv')
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