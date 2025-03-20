import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import brier_score_loss
import joblib

class MarchMadnessModel:
    def __init__(self, feature_engineer):
        self.feature_engineer = feature_engineer
        # Modify hyperparameters to optimize for Brier score
        self.model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            random_state=42,
            # Add loss function that's more suitable for probability calibration
            loss='log_loss'
        )
        self.feature_importance = None
        
    def train(self, start_year: int = 2003, end_year: int = 2024, 
              gender: str = 'M') -> Dict[str, float]:
        """Train the model using historical tournament data."""
        # Prepare training data
        X, y = self.feature_engineer.prepare_training_data(start_year, end_year, gender)
        
        # Train model
        self.model.fit(X, y)
        
        # Calculate feature importance
        self.feature_importance = dict(zip(X.columns, self.model.feature_importances_))
        
        # Perform cross-validation using Brier score
        cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='neg_brier_score')
        
        # Convert negative Brier scores back to positive
        brier_scores = -cv_scores
        
        return {
            'brier_score': brier_scores.mean(),
            'brier_score_std': brier_scores.std()
        }
    
    def predict_matchup(self, team1_id: int, team2_id: int, 
                       season: int, gender: str) -> float:
        """Predict the probability of team1 beating team2."""
        # Create features for this matchup
        features = self.feature_engineer.create_matchup_features(
            team1_id, team2_id, season, gender
        )
        
        # Get raw probability
        prob = self.model.predict_proba(features)[0][1]
        
        # Clip probabilities to avoid extreme values
        # This helps optimize Brier score by avoiding overconfident predictions
        prob = np.clip(prob, 0.025, 0.975)
        
        return prob
    
    def generate_tournament_predictions(self, season: int, gender: str,
                                     team_ids: List[int]) -> pd.DataFrame:
        """Generate predictions for all possible tournament matchups."""
        predictions = []
        
        # Generate all possible matchups
        for i, team1_id in enumerate(team_ids):
            for team2_id in team_ids[i+1:]:
                # Create prediction ID following competition format:
                # SSSS_XXXX_YYYY where XXXX is the lower TeamID
                pred_id = f"{season}_{min(team1_id, team2_id)}_{max(team1_id, team2_id)}"
                
                # Get prediction probability
                # Always predict probability for the team with lower ID beating team with higher ID
                if team1_id < team2_id:
                    prob = self.predict_matchup(team1_id, team2_id, season, gender)
                else:
                    prob = 1 - self.predict_matchup(team2_id, team1_id, season, gender)
                    
                predictions.append({
                    'ID': pred_id,
                    'Pred': prob
                })
                
        predictions_df = pd.DataFrame(predictions)
        
        # Verify submission format
        assert all(predictions_df['Pred'].between(0, 1)), "All probabilities must be between 0 and 1"
        assert predictions_df['ID'].str.match(r'\d{4}_\d{4}_\d{4}$').all(), "Invalid ID format"
        
        return predictions_df
    
    def evaluate_predictions(self, predictions: pd.DataFrame, actual_results: pd.DataFrame) -> float:
        """Calculate Brier score for predictions."""
        merged = pd.merge(predictions, actual_results, on='ID')
        return brier_score_loss(merged['Actual'], merged['Pred'])
    
    def save_model(self, filepath: str) -> None:
        """Save the trained model to disk."""
        joblib.dump(self.model, filepath)
        
    def load_model(self, filepath: str) -> None:
        """Load a trained model from disk."""
        self.model = joblib.load(filepath) 