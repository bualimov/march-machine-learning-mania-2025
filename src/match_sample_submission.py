import pandas as pd
import os
import numpy as np

def match_sample_submission():
    """Modify our submission files to match exactly the IDs in the sample submission file"""
    print("Modifying submission files to match exactly the sample submission file format")
    
    # Define paths
    submissions_dir = "submissions"
    os.makedirs(submissions_dir, exist_ok=True)
    
    data_path = os.path.join(os.getcwd(), "march-machine-learning-mania-2025-final-dataset")
    sample_submission_path = os.path.join(data_path, "SampleSubmissionStage2.csv")
    
    submission1_path = os.path.join(submissions_dir, "submission1.csv")
    submission2_path = os.path.join(submissions_dir, "submission2.csv")
    
    # Load the sample submission file
    print(f"Loading sample submission from: {sample_submission_path}")
    sample_df = pd.read_csv(sample_submission_path)
    print(f"Sample submission has {len(sample_df)} rows with columns {sample_df.columns.tolist()}")
    
    # Load our current submissions
    print(f"Loading our current submissions from {submission1_path} and {submission2_path}")
    df1 = pd.read_csv(submission1_path)
    df2 = pd.read_csv(submission2_path)
    
    print(f"Our submission1 has {len(df1)} rows with columns {df1.columns.tolist()}")
    print(f"Our submission2 has {len(df2)} rows with columns {df2.columns.tolist()}")
    
    # Create dictionaries from our current predictions
    pred_map1 = dict(zip(df1['ID'], df1['Pred']))
    pred_map2 = dict(zip(df2['ID'], df2['Pred']))
    
    # Check which IDs are missing from our submissions
    sample_ids = set(sample_df['ID'])
    our_ids = set(df1['ID'])
    
    missing_ids = sample_ids - our_ids
    extra_ids = our_ids - sample_ids
    
    print(f"Found {len(missing_ids)} IDs in sample that are missing from our submissions")
    print(f"Found {len(extra_ids)} extra IDs in our submissions that aren't in the sample")
    
    # Extract team IDs from sample and our submissions
    sample_team_ids = set()
    for id_str in sample_df['ID']:
        parts = id_str.split('_')
        if len(parts) >= 3:  # Ensure format is like '2025_1101_1102'
            sample_team_ids.add(int(parts[1]))
            sample_team_ids.add(int(parts[2]))
    
    our_team_ids = set()
    for id_str in df1['ID']:
        parts = id_str.split('_')
        if len(parts) >= 3:  # Ensure format is like '2025_1101_1102'
            our_team_ids.add(int(parts[1]))
            our_team_ids.add(int(parts[2]))
    
    print(f"Sample has {len(sample_team_ids)} unique team IDs")
    print(f"Our submissions have {len(our_team_ids)} unique team IDs")
    
    # Check if 1290 is in the sample
    if 1290 in sample_team_ids:
        print("Team ID 1290 IS in the sample submission")
    else:
        print("Team ID 1290 IS NOT in the sample submission")
    
    # Check if 1290 is in our submissions
    if 1290 in our_team_ids:
        print("Team ID 1290 IS in our submissions")
    else:
        print("Team ID 1290 IS NOT in our submissions")
    
    # Function to generate predictions for missing matchups
    def generate_prediction(id_str, model_num):
        parts = id_str.split('_')
        team1 = int(parts[1])
        team2 = int(parts[2])
        
        # If team1 is MS Valley State, it's weak
        if team1 == 1290:
            return 0.25 if model_num == 1 else 0.15
        # If team2 is MS Valley State, it's weak
        if team2 == 1290:
            return 0.75 if model_num == 1 else 0.85
        # Otherwise, use a default value
        return 0.5
    
    # Create new dataframes with exactly the same IDs as the sample
    new_rows1 = []
    new_rows2 = []
    
    missing_count = 0
    for idx, row in sample_df.iterrows():
        id_str = row['ID']
        
        if id_str in pred_map1:
            # Use existing predictions
            pred1 = pred_map1[id_str]
            pred2 = pred_map2[id_str]
        else:
            # Generate new predictions
            missing_count += 1
            pred1 = generate_prediction(id_str, 1)
            pred2 = generate_prediction(id_str, 2)
        
        new_rows1.append((id_str, pred1))
        new_rows2.append((id_str, pred2))
    
    print(f"Generated predictions for {missing_count} missing matchups")
    
    # Create new dataframes
    new_df1 = pd.DataFrame(new_rows1, columns=['ID', 'Pred'])
    new_df2 = pd.DataFrame(new_rows2, columns=['ID', 'Pred'])
    
    print(f"New submission1 has {len(new_df1)} rows")
    print(f"New submission2 has {len(new_df2)} rows")
    
    # Ensure row count matches sample exactly
    if len(new_df1) != len(sample_df):
        print(f"ERROR: New submission has {len(new_df1)} rows, but sample has {len(sample_df)} rows")
    else:
        print(f"SUCCESS: New submissions have exactly {len(new_df1)} rows, matching the sample")
    
    # Save to new files
    output1_path = os.path.join(submissions_dir, "submission1_final.csv")
    output2_path = os.path.join(submissions_dir, "submission2_final.csv")
    
    new_df1.to_csv(output1_path, index=False)
    new_df2.to_csv(output2_path, index=False)
    
    # Verify row counts in saved files
    print(f"Saved file {output1_path} has {sum(1 for _ in open(output1_path))} rows (including header)")
    print(f"Saved file {output2_path} has {sum(1 for _ in open(output2_path))} rows (including header)")
    
    print("\nFinal submission files created. Please upload these files.")

if __name__ == "__main__":
    match_sample_submission() 