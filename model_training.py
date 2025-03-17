#!/usr/bin/env python3
# Model training module for March Machine Learning Mania 2025

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss, log_loss, accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
import joblib
import os
import pickle

def train_model(X, y, model_type='logistic', save_model=True, model_dir='models', time_weight=True):
    """
    Train a machine learning model on the provided data.
    
    Parameters:
    -----------
    X : DataFrame
        Features for training
    y : array-like
        Target variable
    model_type : str
        Type of model to train ('logistic', 'random_forest', or 'gradient_boosting')
    save_model : bool
        Whether to save the trained model to disk
    model_dir : str
        Directory to save the model
    time_weight : bool
        Whether to apply time-based weighting (giving more importance to recent seasons)
    
    Returns:
    --------
    object
        Trained model
    """
    print(f"Training {model_type} model on {X.shape[0]} samples with {X.shape[1]} features")
    
    # Create directory for saving models if it doesn't exist
    if save_model and not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Apply time-based weighting if requested
    sample_weights = None
    if time_weight and 'Season' in X.columns:
        # Calculate weights based on season (more recent seasons get higher weights)
        min_season = X['Season'].min()
        max_season = X['Season'].max()
        season_range = max_season - min_season
        
        if season_range > 0:
            # Normalize seasons to [0, 1] range and then scale to [1, 3] range
            # This means the most recent season is 3x more important than the oldest
            sample_weights = 1 + 2 * ((X['Season'] - min_season) / season_range)
            print(f"Applied time-based weighting: seasons from {min_season} to {max_season}")
            print(f"Weight range: {sample_weights.min():.2f} to {sample_weights.max():.2f}")
    
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Adjust sample weights after split
    if sample_weights is not None:
        train_indices = X.index.isin(X_train.index)
        sample_weights_train = sample_weights[train_indices].values
    else:
        sample_weights_train = None
    
    # Remove non-numeric columns (like TeamID) for training
    non_numeric_cols = ['Team1ID', 'Team2ID']
    X_train_numeric = X_train.drop(columns=[col for col in non_numeric_cols if col in X_train.columns])
    X_val_numeric = X_val.drop(columns=[col for col in non_numeric_cols if col in X_val.columns])
    
    # Check for NaN values
    if X_train_numeric.isna().any().any():
        print(f"Warning: Training data contains {X_train_numeric.isna().sum().sum()} NaN values")
        print("Columns with NaN values:")
        print(X_train_numeric.isna().sum()[X_train_numeric.isna().sum() > 0])
    
    # Create a pipeline with imputation, scaling, and the model
    if model_type == 'logistic':
        # Create a simple logistic regression model
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(random_state=42, max_iter=1000, C=1.0))
        ])
        
    elif model_type == 'random_forest':
        # Create a simple random forest model
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('model', RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            ))
        ])
        
    elif model_type == 'gradient_boosting':
        # Create a simple gradient boosting model
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('model', GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                min_samples_split=5,
                subsample=0.8,
                random_state=42
            ))
        ])
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Fit the model with sample weights if available
    if sample_weights_train is not None:
        pipeline.fit(X_train_numeric, y_train, model__sample_weight=sample_weights_train)
    else:
        pipeline.fit(X_train_numeric, y_train)
    
    # Evaluate on validation set
    y_val_pred_proba = pipeline.predict_proba(X_val_numeric)[:, 1]
    val_log_loss = log_loss(y_val, y_val_pred_proba)
    val_brier_score = brier_score_loss(y_val, y_val_pred_proba)
    val_accuracy = accuracy_score(y_val, y_val_pred_proba > 0.5)
    val_auc = roc_auc_score(y_val, y_val_pred_proba)
    
    print(f"Validation Log Loss: {val_log_loss:.4f}")
    print(f"Validation Brier Score: {val_brier_score:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Validation AUC: {val_auc:.4f}")
    
    # Save the model if requested
    if save_model:
        model_path = os.path.join(model_dir, f"{model_type}_model.pkl")
        
        # Save the model and feature names
        model_info = {
            'model': pipeline,
            'feature_names': list(X_train_numeric.columns)
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_info, f)
        print(f"Model saved to {model_path}")
    
    return model_info

def evaluate_model(model_info, X, y):
    """
    Evaluate a trained model on the provided data.
    
    Parameters:
    -----------
    model_info : dict or object
        Trained model information (dict with 'model' and 'feature_names' keys) or just the model
    X : DataFrame
        Features for evaluation
    y : array-like
        Target variable
    
    Returns:
    --------
    float
        Brier score (lower is better)
    """
    # Extract model and feature names
    if isinstance(model_info, dict) and 'model' in model_info:
        model = model_info['model']
        feature_names = model_info.get('feature_names', None)
    else:
        model = model_info
        feature_names = None
    
    # Remove non-numeric columns (like TeamID) for evaluation
    non_numeric_cols = ['Team1ID', 'Team2ID']
    X_numeric = X.drop(columns=[col for col in non_numeric_cols if col in X.columns])
    
    # Ensure feature order matches training if feature names are available
    if feature_names is not None:
        # Get common columns
        common_cols = [col for col in feature_names if col in X_numeric.columns]
        
        # Check if we have all required features
        if len(common_cols) < len(feature_names):
            missing_cols = set(feature_names) - set(common_cols)
            print(f"Warning: Missing {len(missing_cols)} features: {missing_cols}")
            
            # Add missing columns with zeros
            for col in missing_cols:
                X_numeric[col] = 0
        
        # Reorder columns to match training
        X_numeric = X_numeric[feature_names]
    
    # Get probability predictions
    y_pred_proba = model.predict_proba(X_numeric)[:, 1]
    
    # Calculate metrics
    brier = brier_score_loss(y, y_pred_proba)
    log_loss_score = log_loss(y, y_pred_proba)
    accuracy = accuracy_score(y, y_pred_proba > 0.5)
    auc = roc_auc_score(y, y_pred_proba)
    
    print(f"Brier Score: {brier:.4f}")
    print(f"Log Loss: {log_loss_score:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC: {auc:.4f}")
    
    return brier

def load_model(model_path):
    """
    Load a trained model from disk.
    
    Parameters:
    -----------
    model_path : str
        Path to the saved model
    
    Returns:
    --------
    dict or object
        Loaded model information (dict with 'model' and 'feature_names' keys) or just the model
    """
    model_info = joblib.load(model_path)
    
    # Check if it's a dict with model and feature_names
    if isinstance(model_info, dict) and 'model' in model_info:
        print(f"Loaded model with {len(model_info.get('feature_names', []))} features")
        return model_info
    else:
        # Old format, just the model
        print("Loaded model (old format without feature names)")
        return {'model': model_info, 'feature_names': None}

def save_model(model_info, model_path):
    """
    Save the model and feature names to a file.
    
    Parameters:
    -----------
    model_info : dict
        Dictionary containing the model and feature names
    model_path : str
        Path to save the model
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Save the model
    with open(model_path, 'wb') as f:
        pickle.dump(model_info, f)
    
    print(f"Model saved to {model_path}")
    
    # Print model info
    if isinstance(model_info, dict):
        if 'model' in model_info:
            print(f"Model type: {type(model_info['model']).__name__}")
        if 'feature_names' in model_info:
            print(f"Number of features: {len(model_info['feature_names'])}")
    else:
        print(f"Model type: {type(model_info).__name__}") 