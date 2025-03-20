import os
import pandas as pd
from march_madness_2025_data_processor import MarchMadnessDataProcessor
from march_madness_2025_feature_engineer import MarchMadnessFeatureEngineer
from march_madness_2025_model import MarchMadnessModel

def main():
    # Setup paths
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_path, 'march-machine-learning-mania-2025-final-dataset')
    models_path = os.path.join(base_path, 'models')
    
    # Create models directory if it doesn't exist
    os.makedirs(models_path, exist_ok=True)
    
    # Initialize components
    data_processor = MarchMadnessDataProcessor(data_path)
    feature_engineer = MarchMadnessFeatureEngineer(data_processor)
    model = MarchMadnessModel(feature_engineer)
    
    # Load all necessary data
    print("Loading data...")
    data_processor.load_regular_season_data()
    data_processor.load_tournament_data()
    data_processor.load_rankings()
    data_processor.load_coaches()
    data_processor.load_conferences()
    
    # Train models for both men's and women's tournaments
    for gender in ['M', 'W']:
        print(f"\nTraining {('Men' if gender == 'M' else 'Women')}'s Tournament Model")
        
        # Train model using data from 2003-2024
        metrics = model.train(start_year=2003, end_year=2024, gender=gender)
        print(f"Cross-validation Brier score: {metrics['brier_score']:.4f} (±{metrics['brier_score_std']:.4f})")
        print("Note: Lower Brier score is better. Perfect predictions would score 0.0")
        
        # Save model
        model_path = os.path.join(models_path, f'model_{gender}_2025.joblib')
        model.save_model(model_path)
        print(f"Model saved to {model_path}")
        
        # Print feature importance
        print("\nFeature Importance:")
        for feature, importance in sorted(model.feature_importance.items(), 
                                       key=lambda x: x[1], reverse=True):
            print(f"{feature}: {importance:.4f}")
        
        # Generate predictions for 2025 tournament
        sample_submission = pd.read_csv(os.path.join(data_path, 'SampleSubmissionStage1.csv'))
        
        # Extract unique team IDs from the sample submission
        team_ids = set()
        for id_str in sample_submission['ID']:
            _, team1, team2 = id_str.split('_')
            team_ids.add(int(team1))
            team_ids.add(int(team2))
        
        team_ids = sorted(list(team_ids))
        
        # Generate predictions for all possible matchups
        print("\nGenerating predictions for 2025 tournament...")
        predictions = model.generate_tournament_predictions(2025, gender, team_ids)
        
        # Verify predictions match sample submission format
        assert len(predictions) == len(sample_submission), "Prediction count mismatch"
        assert set(predictions['ID']) == set(sample_submission['ID']), "Prediction IDs mismatch"
        
        # Save predictions
        output_path = os.path.join(base_path, f'predictions_{gender}_2025.csv')
        predictions.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")
        
        # Calculate average predicted probability
        avg_prob = predictions['Pred'].mean()
        print(f"Average predicted win probability: {avg_prob:.4f}")
        print(f"Prediction distribution:")
        print(predictions['Pred'].describe())

if __name__ == "__main__":
    main() 