import pandas as pd
import numpy as np
import os
import itertools
from tqdm import tqdm
from src.march_madness_2025_comprehensive_model import ComprehensiveMarchMadnessModel

def generate_full_submission():
    """Generate a complete submission file with all possible matchups."""
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_path, 'march-machine-learning-mania-2025-final-dataset')
    
    print("Loading sample submission to understand format...")
    try:
        # Try to load the sample submission to understand the required format
        sample_submission = pd.read_csv(os.path.join(data_path, 'SampleSubmissionStage1.csv'))
        print(f"Sample submission has {len(sample_submission)} rows")
    except FileNotFoundError:
        print("Sample submission file not found. Will generate based on all team combinations.")
        sample_submission = None
    
    all_predictions = []
    
    # Process both tournaments
    for gender in ['M', 'W']:
        gender_name = 'Men' if gender == 'M' else 'Women'
        print(f"\nProcessing {gender_name}'s Tournament")
        
        # Initialize model
        model = ComprehensiveMarchMadnessModel(data_path)
        model.load_all_data(gender)
        model.normalize_features()
        
        # Load all teams to consider
        teams = pd.read_csv(os.path.join(data_path, f'{gender}Teams.csv'))
        team_ids = sorted(teams['TeamID'].unique())
        
        print(f"Generating predictions for all possible matchups between {len(team_ids)} teams...")
        
        # Generate all possible team combinations
        matchups = list(itertools.combinations(team_ids, 2))
        print(f"Total possible matchups: {len(matchups)}")
        
        # Generate predictions for all matchups
        for team1_id, team2_id in tqdm(matchups, desc=f"Generating {gender_name}'s predictions"):
            # Always predict probability for lower ID beating higher ID
            if team1_id < team2_id:
                prob = model.predict_matchup(team1_id, team2_id)
                prediction_id = f"2025_{team1_id}_{team2_id}"
            else:
                prob = 1 - model.predict_matchup(team2_id, team1_id)
                prediction_id = f"2025_{team2_id}_{team1_id}"
            
            all_predictions.append({
                'ID': prediction_id,
                'Pred': prob
            })
    
    # Convert to DataFrame
    predictions_df = pd.DataFrame(all_predictions)
    
    # If we have a sample submission, validate against it
    if sample_submission is not None:
        print("\nValidating against sample submission...")
        
        # Check if we have all required IDs
        sample_ids = set(sample_submission['ID'])
        prediction_ids = set(predictions_df['ID'])
        
        missing_ids = sample_ids - prediction_ids
        extra_ids = prediction_ids - sample_ids
        
        if missing_ids:
            print(f"WARNING: Missing {len(missing_ids)} IDs required by sample submission")
            print(f"First few missing IDs: {list(missing_ids)[:5]}")
            
            # Add missing IDs with default prediction of 0.5
            missing_df = pd.DataFrame([{'ID': id, 'Pred': 0.5} for id in missing_ids])
            predictions_df = pd.concat([predictions_df, missing_df], ignore_index=True)
        
        if extra_ids:
            print(f"WARNING: Generated {len(extra_ids)} extra IDs not in sample submission")
            print(f"First few extra IDs: {list(extra_ids)[:5]}")
            
            # Remove extra IDs
            predictions_df = predictions_df[predictions_df['ID'].isin(sample_ids)]
    
    # Save final submission
    output_path = os.path.join(base_path, 'predictions_2025_full_submission.csv')
    predictions_df.to_csv(output_path, index=False)
    
    print(f"\nFinal submission saved to {output_path}")
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
    generate_full_submission() 