import pandas as pd
import numpy as np
import os
import json
from sklearn.metrics import brier_score_loss, log_loss
from src.march_madness_2025_comprehensive_model import ComprehensiveMarchMadnessModel

def evaluate_historical_model(years_to_evaluate=[2019, 2020, 2021, 2022, 2023, 2024]):
    """Evaluate model accuracy on historical tournament data."""
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_path, 'march-machine-learning-mania-2025-final-dataset')
    
    results = {}
    
    for gender in ['M', 'W']:
        gender_name = 'Men' if gender == 'M' else 'Women'
        print(f"\nEvaluating {gender_name}'s Tournament Model on Historical Data")
        
        # Load actual tournament results
        tourney_results = pd.read_csv(os.path.join(data_path, f'{gender}NCAATourneyCompactResults.csv'))
        
        gender_results = {}
        for year in years_to_evaluate:
            # Skip if year not in the data
            if year not in tourney_results['Season'].unique():
                print(f"Skipping {year} (no data available)")
                continue
                
            print(f"\nEvaluating {year} tournament predictions:")
            
            # Initialize model
            model = ComprehensiveMarchMadnessModel(data_path)
            model.current_season = year  # Set the year to evaluate
            model.load_all_data(gender)
            model.normalize_features()
            
            # Get actual tournament games for this year
            year_results = tourney_results[tourney_results['Season'] == year].copy()
            
            # Prepare for evaluation
            all_predictions = []
            actual_outcomes = []
            correct_predictions = 0
            
            # For each game, predict and compare to actual
            for _, game in year_results.iterrows():
                team1_id = min(game['WTeamID'], game['LTeamID'])
                team2_id = max(game['WTeamID'], game['LTeamID'])
                
                # Predict probability of team1 beating team2
                if team1_id == game['WTeamID']:
                    predicted_prob = model.predict_matchup(team1_id, team2_id)
                    actual_outcome = 1  # Team1 won
                else:
                    predicted_prob = 1 - model.predict_matchup(team2_id, team1_id)
                    actual_outcome = 0  # Team1 lost
                
                all_predictions.append(predicted_prob)
                actual_outcomes.append(actual_outcome)
                
                # Check if prediction was correct (>0.5 for win, <0.5 for loss)
                prediction_correct = (predicted_prob > 0.5 and actual_outcome == 1) or \
                                    (predicted_prob < 0.5 and actual_outcome == 0)
                if prediction_correct:
                    correct_predictions += 1
            
            # Calculate metrics
            if all_predictions:
                brier = brier_score_loss(actual_outcomes, all_predictions)
                logloss = log_loss(actual_outcomes, all_predictions)
                accuracy = correct_predictions / len(all_predictions)
                
                print(f"  Total games: {len(all_predictions)}")
                print(f"  Accuracy: {accuracy:.4f} ({correct_predictions}/{len(all_predictions)} correct)")
                print(f"  Brier score: {brier:.4f} (lower is better)")
                print(f"  Log Loss: {logloss:.4f} (lower is better)")
                
                gender_results[year] = {
                    'games': len(all_predictions),
                    'accuracy': accuracy,
                    'brier_score': brier,
                    'log_loss': logloss
                }
            else:
                print("  No games to evaluate")
        
        # Calculate average metrics across years
        if gender_results:
            avg_accuracy = np.mean([yr['accuracy'] for yr in gender_results.values()])
            avg_brier = np.mean([yr['brier_score'] for yr in gender_results.values()])
            avg_logloss = np.mean([yr['log_loss'] for yr in gender_results.values()])
            
            print(f"\nAverage metrics for {gender_name}'s tournaments:")
            print(f"  Accuracy: {avg_accuracy:.4f}")
            print(f"  Brier score: {avg_brier:.4f}")
            print(f"  Log Loss: {avg_logloss:.4f}")
            
            gender_results['average'] = {
                'accuracy': avg_accuracy,
                'brier_score': avg_brier,
                'log_loss': avg_logloss
            }
        
        results[gender] = gender_results
    
    # Save evaluation results
    with open(os.path.join(base_path, 'model_evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

if __name__ == "__main__":
    # Evaluate model on recent tournaments
    evaluate_historical_model() 