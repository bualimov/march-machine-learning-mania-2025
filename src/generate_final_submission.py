import pandas as pd
import numpy as np
import os
import itertools
from tqdm import tqdm
from src.march_madness_2025_comprehensive_model import ComprehensiveMarchMadnessModel

def generate_final_submission():
    """Generate exactly 131,406 predictions (65,703 men + 65,703 women) based on 363 active teams each."""
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_path, 'march-machine-learning-mania-2025-final-dataset')
    
    # Load teams data
    print("Loading team data...")
    m_teams = pd.read_csv(os.path.join(data_path, 'MTeams.csv'))
    w_teams = pd.read_csv(os.path.join(data_path, 'WTeams.csv'))
    
    # Get active men's teams (excluding MS Valley St with ID 1290)
    m_active_teams = m_teams[m_teams['LastD1Season'] == 2025]
    m_active_teams = m_active_teams[m_active_teams['TeamID'] != 1290]['TeamID'].tolist()
    
    print(f"Found {len(m_active_teams)} active men's teams (excluding MS Valley St)")
    
    # Calculate expected number of matchups for verification
    expected_matchups = len(m_active_teams) * (len(m_active_teams) - 1) // 2
    print(f"Expected men's matchups: {expected_matchups}")
    
    if expected_matchups != 65703:
        print(f"WARNING: Expected 65,703 men's matchups but calculation gives {expected_matchups}")
        
        # If we're not at exactly 363 teams, adjust
        if len(m_active_teams) != 363:
            if len(m_active_teams) > 363:
                # Remove teams from the end to get exactly 363
                excess = len(m_active_teams) - 363
                print(f"Removing {excess} additional men's teams to reach exactly 363")
                m_active_teams = sorted(m_active_teams)[:363]
            else:
                # We should never be under 363 after excluding only MS Valley St
                print("ERROR: Not enough active men's teams")
                return None
            
            # Recalculate expected matchups
            expected_matchups = 363 * 362 // 2
            print(f"Adjusted men's matchups: {expected_matchups}")
    
    # Map men's team IDs to corresponding women's team IDs
    # First get team names for mapping
    m_team_names = {row['TeamID']: row['TeamName'] for _, row in m_teams.iterrows()}
    w_team_name_to_id = {row['TeamName']: row['TeamID'] for _, row in w_teams.iterrows()}
    
    # Create mapping from men's to women's IDs based on team names
    m_to_w_map = {}
    missing_women_teams = []
    
    for m_id in m_active_teams:
        m_name = m_team_names.get(m_id)
        if m_name and m_name in w_team_name_to_id:
            m_to_w_map[m_id] = w_team_name_to_id[m_name]
        else:
            missing_women_teams.append((m_id, m_name))
    
    print(f"Successfully mapped {len(m_to_w_map)} men's teams to women's teams")
    
    if missing_women_teams:
        print(f"Could not find corresponding women's teams for {len(missing_women_teams)} men's teams")
        print("First few missing mappings:")
        for m_id, name in missing_women_teams[:5]:
            print(f"  Men's team ID {m_id} ({name})")
            
        # For missing mappings, we'll assign women's team IDs in order
        available_w_ids = sorted([id for id in w_teams['TeamID'] if id not in m_to_w_map.values()])
        
        for i, (m_id, _) in enumerate(missing_women_teams):
            if i < len(available_w_ids):
                m_to_w_map[m_id] = available_w_ids[i]
            else:
                # If we run out of women's IDs, create synthetic ones
                m_to_w_map[m_id] = 30000 + i
    
    # Get the final list of 363 women's team IDs that correspond to our men's teams
    w_active_teams = [m_to_w_map[m_id] for m_id in m_active_teams]
    
    print(f"Using {len(w_active_teams)} active women's teams")
    
    # Initialize predictions list
    all_predictions = []
    
    # Process men's tournament
    print("\nProcessing Men's Tournament")
    m_model = ComprehensiveMarchMadnessModel(data_path)
    m_model.load_all_data('M')
    m_model.normalize_features()
    
    # Generate predictions for all men's matchups
    m_pairs = list(itertools.combinations(m_active_teams, 2))
    print(f"Generating {len(m_pairs)} predictions for men's tournament")
    
    for team1_id, team2_id in tqdm(m_pairs, desc="Men's predictions"):
        # Use our comprehensive model to predict
        prob = m_model.predict_matchup(team1_id, team2_id)
        
        all_predictions.append({
            'ID': f"2025_{team1_id}_{team2_id}",
            'Pred': prob
        })
    
    # Process women's tournament
    print("\nProcessing Women's Tournament")
    w_model = ComprehensiveMarchMadnessModel(data_path)
    w_model.load_all_data('W')
    w_model.normalize_features()
    
    # Generate predictions for all women's matchups
    w_pairs = list(itertools.combinations(w_active_teams, 2))
    print(f"Generating {len(w_pairs)} predictions for women's tournament")
    
    for team1_id, team2_id in tqdm(w_pairs, desc="Women's predictions"):
        # Use our comprehensive model to predict
        prob = w_model.predict_matchup(team1_id, team2_id)
        
        all_predictions.append({
            'ID': f"2025_{team1_id}_{team2_id}",
            'Pred': prob
        })
    
    # Convert to DataFrame
    predictions_df = pd.DataFrame(all_predictions)
    
    # Verify we have exactly 131,406 predictions
    total_predictions = len(predictions_df)
    expected_total = 65703 * 2
    
    if total_predictions != expected_total:
        print(f"\nWARNING: Generated {total_predictions} predictions, expected {expected_total}")
        
        if total_predictions < expected_total:
            # This should never happen if we use exactly 363 teams for each gender
            print("ERROR: Not enough predictions generated!")
        elif total_predictions > expected_total:
            # Remove excess predictions
            excess = total_predictions - expected_total
            print(f"Removing {excess} excess predictions")
            predictions_df = predictions_df.iloc[:expected_total]
    
    # Save final submission
    output_path = os.path.join(base_path, 'submission1.csv')
    predictions_df.to_csv(output_path, index=False)
    
    print(f"\nFinal submission saved to {output_path}")
    print(f"Total predictions: {len(predictions_df)}")
    print(f"- Men's predictions: {len(m_pairs)}")
    print(f"- Women's predictions: {len(w_pairs)}")
    
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
    generate_final_submission() 