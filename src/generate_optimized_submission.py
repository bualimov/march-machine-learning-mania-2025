import pandas as pd
import numpy as np
import os
import itertools
import json
from tqdm import tqdm
from sklearn.metrics import brier_score_loss
from src.march_madness_2025_comprehensive_model import ComprehensiveMarchMadnessModel

class OptimizedMarchMadnessModel:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.team_stats = {}
        self.team_history = {}
        self.conference_strength = {}
        self.tournament_history = {}
        self.current_season = 2025
        self.time_decay_factor = 0.85  # Stronger recency bias
        self.base_model = None
        self.calibration_map = {}  # For storing optimal probability adjustments
        
    def load_all_data(self, gender: str = 'M'):
        """Load all necessary data files and build optimized model."""
        print(f"Loading data for {gender} tournament...")
        
        # Initialize base model first
        self.base_model = ComprehensiveMarchMadnessModel(self.data_path)
        self.base_model.load_all_data(gender)
        self.base_model.normalize_features()
        
        # Load additional data for optimization
        self.gender = gender
        self.tourney_results = pd.read_csv(os.path.join(self.data_path, f'{gender}NCAATourneyCompactResults.csv'))
        self.tourney_seeds = pd.read_csv(os.path.join(self.data_path, f'{gender}NCAATourneySeeds.csv'))
        self.regular_season = pd.read_csv(os.path.join(self.data_path, f'{gender}RegularSeasonCompactResults.csv'))
        self.teams = pd.read_csv(os.path.join(self.data_path, f'{gender}Teams.csv'))
        
        # Perform additional analyses
        self.analyze_tournament_history()
        self.analyze_upset_patterns()
        self.calibrate_probabilities()
        
    def analyze_tournament_history(self):
        """Analyze historical tournament outcomes in detail."""
        print("Analyzing tournament history...")
        
        # Create dictionaries for storing historical performance
        team_performance = {}
        for _, game in self.tourney_results.iterrows():
            season = game['Season']
            
            # Skip very old data
            if season < 2010:
                continue
                
            winner_id = game['WTeamID']
            loser_id = game['LTeamID']
            
            # Initialize if needed
            for team_id in [winner_id, loser_id]:
                if team_id not in team_performance:
                    team_performance[team_id] = {
                        'wins': 0,
                        'losses': 0,
                        'recent_wins': 0,  # Weighted by recency
                        'recent_losses': 0,
                        'upsets_made': 0,  # Lower seed beating higher seed
                        'upsets_suffered': 0,
                        'close_wins': 0,   # Wins by < 5 points
                        'close_losses': 0,  # Losses by < 5 points
                        'blowout_wins': 0,  # Wins by > 15 points
                        'blowout_losses': 0,
                        'seasons': set()
                    }
            
            # Record basic stats
            team_performance[winner_id]['wins'] += 1
            team_performance[loser_id]['losses'] += 1
            
            # Record seasons played
            team_performance[winner_id]['seasons'].add(season)
            team_performance[loser_id]['seasons'].add(season)
            
            # Weight by recency
            recency_weight = self.time_decay_factor ** (self.current_season - 1 - season)
            team_performance[winner_id]['recent_wins'] += recency_weight
            team_performance[loser_id]['recent_losses'] += recency_weight
            
            # Record margin-based stats
            score_diff = game['WScore'] - game['LScore']
            if score_diff < 5:
                team_performance[winner_id]['close_wins'] += 1
                team_performance[loser_id]['close_losses'] += 1
            elif score_diff > 15:
                team_performance[winner_id]['blowout_wins'] += 1
                team_performance[loser_id]['blowout_losses'] += 1
            
            # Check for upsets based on seeds
            try:
                winner_seed = self.tourney_seeds[
                    (self.tourney_seeds['Season'] == season) & 
                    (self.tourney_seeds['TeamID'] == winner_id)
                ]['Seed'].iloc[0]
                
                loser_seed = self.tourney_seeds[
                    (self.tourney_seeds['Season'] == season) & 
                    (self.tourney_seeds['TeamID'] == loser_id)
                ]['Seed'].iloc[0]
                
                # Convert seed strings to numbers
                winner_seed_num = int(winner_seed[1:3])
                loser_seed_num = int(loser_seed[1:3])
                
                if winner_seed_num > loser_seed_num:
                    team_performance[winner_id]['upsets_made'] += 1
                    team_performance[loser_id]['upsets_suffered'] += 1
            except:
                # Skip if seed info not available
                pass
        
        # Calculate derived metrics
        for team_id, stats in team_performance.items():
            total_games = stats['wins'] + stats['losses']
            if total_games > 0:
                stats['win_pct'] = stats['wins'] / total_games
                stats['tournament_experience'] = len(stats['seasons'])
                stats['upset_factor'] = (stats['upsets_made'] - stats['upsets_suffered']) / max(1, total_games)
                stats['clutch_factor'] = (stats['close_wins'] - stats['close_losses']) / max(1, stats['close_wins'] + stats['close_losses'])
                stats['dominance_factor'] = (stats['blowout_wins'] - stats['blowout_losses']) / max(1, total_games)
        
        self.tournament_history = team_performance
        
    def analyze_upset_patterns(self):
        """Analyze patterns in tournament upsets to better predict surprising outcomes."""
        print("Analyzing upset patterns...")
        
        # Create a matrix of seed matchup outcomes
        seed_matchups = {}
        
        # Analyze at least 10 years of tournament data
        recent_results = self.tourney_results[self.tourney_results['Season'] >= 2010]
        
        for _, game in recent_results.iterrows():
            season = game['Season']
            try:
                # Get seeds for both teams
                winner_seed = self.tourney_seeds[
                    (self.tourney_seeds['Season'] == season) & 
                    (self.tourney_seeds['TeamID'] == game['WTeamID'])
                ]['Seed'].iloc[0]
                
                loser_seed = self.tourney_seeds[
                    (self.tourney_seeds['Season'] == season) & 
                    (self.tourney_seeds['TeamID'] == game['LTeamID'])
                ]['Seed'].iloc[0]
                
                # Extract numeric part of seed
                winner_seed_num = int(winner_seed[1:3])
                loser_seed_num = int(loser_seed[1:3])
                
                # Create a matchup key with lower seed first
                if winner_seed_num < loser_seed_num:
                    matchup = (winner_seed_num, loser_seed_num)
                    result = 1  # Lower seed won
                else:
                    matchup = (loser_seed_num, winner_seed_num)
                    result = 0  # Higher seed won
                
                # Record the result
                if matchup not in seed_matchups:
                    seed_matchups[matchup] = []
                seed_matchups[matchup].append(result)
                
            except:
                # Skip if seed info not available
                continue
        
        # Calculate historical probabilities for each seed matchup
        seed_probabilities = {}
        for matchup, results in seed_matchups.items():
            if len(results) >= 3:  # Only use matchups with sufficient data
                seed_probabilities[matchup] = np.mean(results)
        
        self.seed_matchup_probabilities = seed_probabilities
        
    def calibrate_probabilities(self):
        """Calibrate model by backtesting against recent tournaments."""
        print("Calibrating probability estimates...")
        
        # Use data from 2018-2024 for calibration
        calibration_years = range(2018, 2025)
        
        # Store actual vs. predicted probabilities
        predictions = []
        actuals = []
        
        for year in calibration_years:
            # Skip years with no tournament (e.g., 2020 COVID cancellation)
            year_games = self.tourney_results[self.tourney_results['Season'] == year]
            if len(year_games) == 0:
                continue
                
            print(f"  Backtesting {year} tournament...")
            
            # Temporarily set model to this year
            self.base_model.current_season = year
            
            # For each game, make a prediction and record actual outcome
            for _, game in year_games.iterrows():
                team1_id = min(game['WTeamID'], game['LTeamID'])
                team2_id = max(game['WTeamID'], game['LTeamID'])
                
                # Get prediction
                predicted_prob = self.base_model.predict_matchup(team1_id, team2_id)
                
                # Record actual outcome (1 if team1 won, 0 if team2 won)
                actual_outcome = 1 if team1_id == game['WTeamID'] else 0
                
                predictions.append(predicted_prob)
                actuals.append(actual_outcome)
        
        # Calculate initial Brier score
        initial_brier = brier_score_loss(actuals, predictions)
        print(f"  Initial Brier score: {initial_brier:.4f}")
        
        # Find optimal probability calibration
        # This uses isotonic regression to map raw probabilities to calibrated ones
        from sklearn.isotonic import IsotonicRegression
        calibration_model = IsotonicRegression(out_of_bounds='clip')
        calibration_model.fit(predictions, actuals)
        
        # Test calibrated probabilities
        calibrated_predictions = calibration_model.predict(predictions)
        calibrated_brier = brier_score_loss(actuals, calibrated_predictions)
        print(f"  Calibrated Brier score: {calibrated_brier:.4f}")
        
        # Store calibration model
        self.calibration_model = calibration_model
        
        # Set current season back to 2025
        self.base_model.current_season = 2025
        
    def predict_matchup(self, team1_id: int, team2_id: int) -> float:
        """Make an optimized prediction for team1 vs team2."""
        # Get base prediction from comprehensive model
        base_prob = self.base_model.predict_matchup(team1_id, team2_id)
        
        # Apply seed-based adjustment if both teams are in tournament
        seed_adjustment = 0
        try:
            team1_seed = self.base_model.team_stats[team1_id]['seed']
            team2_seed = self.base_model.team_stats[team2_id]['seed']
            
            # Look up historical probability for this seed matchup
            seed_matchup = (min(team1_seed, team2_seed), max(team1_seed, team2_seed))
            if seed_matchup in self.seed_matchup_probabilities:
                historical_prob = self.seed_matchup_probabilities[seed_matchup]
                
                # If team1 is higher seed, use historical probability directly
                # Otherwise, use 1 - historical probability
                if team1_seed < team2_seed:
                    seed_adjustment = (historical_prob - base_prob) * 0.3
                else:
                    seed_adjustment = ((1 - historical_prob) - base_prob) * 0.3
        except:
            # If seeds not available, no adjustment
            pass
        
        # Apply tournament history adjustment
        history_adjustment = 0
        if team1_id in self.tournament_history and team2_id in self.tournament_history:
            team1_history = self.tournament_history[team1_id]
            team2_history = self.tournament_history[team2_id]
            
            # Compare tournament performance factors
            win_pct_diff = team1_history.get('win_pct', 0.5) - team2_history.get('win_pct', 0.5)
            upset_diff = team1_history.get('upset_factor', 0) - team2_history.get('upset_factor', 0)
            clutch_diff = team1_history.get('clutch_factor', 0) - team2_history.get('clutch_factor', 0)
            dominance_diff = team1_history.get('dominance_factor', 0) - team2_history.get('dominance_factor', 0)
            
            # Weight the factors
            history_adjustment = (
                win_pct_diff * 0.05 + 
                upset_diff * 0.03 + 
                clutch_diff * 0.02 + 
                dominance_diff * 0.02
            )
        
        # Combine adjustments
        adjusted_prob = base_prob + seed_adjustment + history_adjustment
        
        # Apply probability calibration
        try:
            calibrated_prob = self.calibration_model.predict([adjusted_prob])[0]
        except:
            calibrated_prob = adjusted_prob
        
        # Ensure valid probability range
        return np.clip(calibrated_prob, 0.15, 0.85)
        
    def generate_predictions(self, men_teams, women_teams):
        """Generate predictions for all required matchups."""
        all_predictions = []
        
        # Generate men's matchups
        men_pairs = list(itertools.combinations(men_teams, 2))
        print(f"Generating {len(men_pairs)} predictions for men's tournament")
        
        for team1_id, team2_id in tqdm(men_pairs, desc="Men's predictions"):
            prob = self.predict_matchup(team1_id, team2_id)
            
            all_predictions.append({
                'ID': f"2025_{team1_id}_{team2_id}",
                'Pred': prob
            })
        
        return all_predictions

def generate_optimized_submission():
    """Generate a more accurate submission with lower Brier score."""
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_path, 'march-machine-learning-mania-2025-final-dataset')
    
    # Load teams data - use same teams as submission1
    print("Loading team data...")
    m_teams = pd.read_csv(os.path.join(data_path, 'MTeams.csv'))
    w_teams = pd.read_csv(os.path.join(data_path, 'WTeams.csv'))
    
    # Get active men's teams (excluding MS Valley St with ID 1290)
    m_active_teams = m_teams[m_teams['LastD1Season'] == 2025]
    m_active_teams = m_active_teams[m_active_teams['TeamID'] != 1290]['TeamID'].tolist()
    
    # Ensure we have exactly 363 men's teams
    if len(m_active_teams) > 363:
        m_active_teams = sorted(m_active_teams)[:363]
    
    print(f"Using {len(m_active_teams)} active men's teams")
    
    # Map men's team IDs to corresponding women's team IDs
    m_team_names = {row['TeamID']: row['TeamName'] for _, row in m_teams.iterrows()}
    w_team_name_to_id = {row['TeamName']: row['TeamID'] for _, row in w_teams.iterrows()}
    
    # Create mapping from men's to women's IDs
    m_to_w_map = {}
    missing_women_teams = []
    
    for m_id in m_active_teams:
        m_name = m_team_names.get(m_id)
        if m_name and m_name in w_team_name_to_id:
            m_to_w_map[m_id] = w_team_name_to_id[m_name]
        else:
            missing_women_teams.append((m_id, m_name))
    
    print(f"Successfully mapped {len(m_to_w_map)} men's teams to women's teams")
    
    # Handle missing women's teams
    if missing_women_teams:
        available_w_ids = sorted([id for id in w_teams['TeamID'] if id not in m_to_w_map.values()])
        
        for i, (m_id, _) in enumerate(missing_women_teams):
            if i < len(available_w_ids):
                m_to_w_map[m_id] = available_w_ids[i]
            else:
                m_to_w_map[m_id] = 30000 + i
    
    # Get the final list of women's team IDs
    w_active_teams = [m_to_w_map[m_id] for m_id in m_active_teams]
    
    print(f"Using {len(w_active_teams)} active women's teams")
    
    all_predictions = []
    
    # Process men's tournament with optimized model
    print("\nProcessing Men's Tournament with optimized model")
    m_model = OptimizedMarchMadnessModel(data_path)
    m_model.load_all_data('M')
    
    m_predictions = m_model.generate_predictions(m_active_teams, w_active_teams)
    all_predictions.extend(m_predictions)
    
    # Process women's tournament with optimized model
    print("\nProcessing Women's Tournament with optimized model")
    w_model = OptimizedMarchMadnessModel(data_path)
    w_model.load_all_data('W')
    
    # Generate women's matchups
    w_pairs = list(itertools.combinations(w_active_teams, 2))
    print(f"Generating {len(w_pairs)} predictions for women's tournament")
    
    for team1_id, team2_id in tqdm(w_pairs, desc="Women's predictions"):
        prob = w_model.predict_matchup(team1_id, team2_id)
        
        all_predictions.append({
            'ID': f"2025_{team1_id}_{team2_id}",
            'Pred': prob
        })
    
    # Convert to DataFrame
    predictions_df = pd.DataFrame(all_predictions)
    
    # Create submissions directory if it doesn't exist
    submissions_dir = os.path.join(base_path, 'submissions')
    os.makedirs(submissions_dir, exist_ok=True)
    
    # Save optimized submission
    output_path = os.path.join(submissions_dir, 'submission2.csv')
    predictions_df.to_csv(output_path, index=False)
    
    print(f"\nOptimized submission saved to {output_path}")
    print(f"Total predictions: {len(predictions_df)}")
    
    # Print prediction statistics
    print("\nPrediction Statistics:")
    print(predictions_df['Pred'].describe())
    
    # Analyze prediction distribution
    bins = np.arange(0, 1.1, 0.1)
    hist, _ = np.histogram(predictions_df['Pred'], bins=bins)
    print("\nPrediction Distribution:")
    for i in range(len(hist)):
        print(f"{bins[i]:.1f}-{bins[i+1]:.1f}: {hist[i]}")
    
    return predictions_df

if __name__ == "__main__":
    generate_optimized_submission() 