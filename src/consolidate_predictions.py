import pandas as pd
import os

def main():
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Load men's and women's predictions
    men_predictions = pd.read_csv(os.path.join(base_path, 'predictions_M_2025_only.csv'))
    women_predictions = pd.read_csv(os.path.join(base_path, 'predictions_W_2025_only.csv'))
    
    # Combine predictions
    combined_predictions = pd.concat([men_predictions, women_predictions])
    
    # Ensure the format is correct
    combined_predictions = combined_predictions[['ID', 'Pred']]
    
    # Save combined predictions
    output_path = os.path.join(base_path, 'predictions_2025_final_submission.csv')
    combined_predictions.to_csv(output_path, index=False)
    
    print(f"Combined predictions saved to {output_path}")
    print(f"Total predictions: {len(combined_predictions)}")
    print(f"Men's predictions: {len(men_predictions)}")
    print(f"Women's predictions: {len(women_predictions)}")
    
    # Show prediction statistics
    print("\nPrediction Statistics:")
    print(combined_predictions['Pred'].describe())

if __name__ == "__main__":
    main() 