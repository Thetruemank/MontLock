"""
MontLock Model Trainer Module

This module trains a machine learning model on the extracted biometric features
and saves it for later use in authentication.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from feature_extractor import FeatureExtractor
import pickle


class ModelTrainer:
    def __init__(self, model_dir="models"):
        """
        Initialize the model trainer.
        
        Args:
            model_dir (str): Directory to save trained models
        """
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        
        # Create model directory if it doesn't exist
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
    
    def train_model(self, features_data, contamination=0.05):
        """
        Train an anomaly detection model using a statistical profile approach.
        
        Args:
            features_data (DataFrame): Features extracted from mouse movement data
            contamination (float): Expected proportion of outliers in the data
            
        Returns:
            tuple: (trained_model, scaler)
        """
        # Convert dict to DataFrame if necessary
        if isinstance(features_data, dict):
            features_data = pd.DataFrame([features_data])
            
        if len(features_data) == 0:
            raise ValueError("Cannot train on empty dataset")
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_data)
        
        # Train One-Class SVM model
        model = OneClassSVM(
            kernel='rbf',
            gamma='scale',
            nu=contamination
        )
        model.fit(features_scaled)
        
        # Test the model on its training data to see the score distribution
        scores = model.decision_function(features_scaled)
        print(f"Training scores - min: {min(scores):.4f}, max: {max(scores):.4f}, mean: {np.mean(scores):.4f}")
        
        # Calculate statistical profiles for each feature
        feature_profiles = {}
        
        # Use a more sensitive std multiplier for detection bounds
        std_multiplier = 2.5  # Reduced from 3.0 to 2.5 for more sensitivity
        
        for feature in features_data.columns:
            values = features_data[feature].values
            
            if len(values) > 1:
                mean = np.mean(values)
                std = np.std(values)
                min_val = np.min(values)
                max_val = np.max(values)
                
                # Calculate bounds based on mean and std
                lower_bound = mean - std_multiplier * std
                upper_bound = mean + std_multiplier * std
                
                feature_profiles[feature] = {
                    'mean': mean,
                    'std': std,
                    'min': min_val,
                    'max': max_val,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound
                }
            else:
                # Handle single sample case
                value = values[0]
                feature_profiles[feature] = {
                    'mean': value,
                    'std': value * 0.2,  # Reduced from 0.0 to 0.2 for more sensitivity
                    'min': value,
                    'max': value,
                    'lower_bound': value * 0.7,  # Changed from 0.5 to 0.7 for more sensitivity
                    'upper_bound': value * 1.3   # Changed from 1.5 to 1.3 for more sensitivity
                }
        
        # Save feature profiles
        self.feature_profiles = feature_profiles
        
        # Save profiles to file
        profile_path = os.path.join(self.model_dir, 'feature_profiles.pkl')
        with open(profile_path, 'wb') as f:
            pickle.dump(feature_profiles, f)
        
        return model, scaler
    
    def evaluate_model(self, features_data, test_size=0.2):
        """
        Evaluate the trained model using a train-test split.
        
        Args:
            features_data (pandas.DataFrame): Features extracted from mouse movements
            test_size (float): Proportion of data to use for testing
            
        Returns:
            dict: Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet")
        
        # Split data into train and test sets
        X_train, X_test = train_test_split(features_data, test_size=test_size, random_state=42)
        
        # Scale the data
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Get predictions (1 for inliers, -1 for outliers)
        y_train_pred = self.model.predict(X_train_scaled)
        y_test_pred = self.model.predict(X_test_scaled)
        
        # Convert to binary (1 for authorized, 0 for unauthorized)
        y_train_pred = np.where(y_train_pred == 1, 1, 0)
        y_test_pred = np.where(y_test_pred == 1, 1, 0)
        
        # For evaluation purposes, we assume all training data is from authorized user
        y_train_true = np.ones(len(X_train))
        y_test_true = np.ones(len(X_test))
        
        # Calculate metrics
        train_accuracy = np.mean(y_train_pred == y_train_true)
        test_accuracy = np.mean(y_test_pred == y_test_true)
        
        # Get decision scores
        train_scores = self.model.decision_function(X_train_scaled)
        test_scores = self.model.decision_function(X_test_scaled)
        
        # Plot score distributions
        plt.figure(figsize=(10, 6))
        plt.hist(train_scores, bins=50, alpha=0.5, label='Training Data')
        plt.hist(test_scores, bins=50, alpha=0.5, label='Test Data')
        plt.axvline(x=0, color='r', linestyle='--', label='Decision Boundary')
        plt.title('Decision Score Distribution')
        plt.xlabel('Decision Score')
        plt.ylabel('Count')
        plt.legend()
        
        # Save the plot
        plot_path = os.path.join(self.model_dir, 'score_distribution.png')
        plt.savefig(plot_path)
        
        return {
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'train_scores_mean': np.mean(train_scores),
            'test_scores_mean': np.mean(test_scores),
            'plot_path': plot_path
        }
    
    def save_model(self, model_name="montlock_model"):
        """
        Save the trained model and scaler to disk.
        
        Args:
            model_name (str): Base name for the saved model files
            
        Returns:
            tuple: Paths to the saved model and scaler files
        """
        if not hasattr(self, 'model') or not hasattr(self, 'scaler'):
            raise ValueError("Model and scaler must be trained before saving")
        
        # Create model directory if it doesn't exist
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Save model and scaler
        model_path = os.path.join(self.model_dir, f"{model_name}.pkl")
        scaler_path = os.path.join(self.model_dir, f"{model_name}_scaler.pkl")
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        
        print(f"Model saved to {model_path}")
        print(f"Scaler saved to {scaler_path}")
        
        return model_path, scaler_path
    
    def load_model(self, model_name="montlock_model"):
        """
        Load a trained model from disk.
        
        Args:
            model_name (str): Name of the model to load
            
        Returns:
            tuple: (model, scaler)
        """
        # Construct file paths
        model_path = os.path.join(self.model_dir, f"{model_name}.pkl")
        scaler_path = os.path.join(self.model_dir, f"{model_name}_scaler.pkl")
        profile_path = os.path.join(self.model_dir, "feature_profiles.pkl")
        
        # Check if files exist
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Model files not found for {model_name}")
        
        # Load model and scaler
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        # Load feature profiles if they exist
        if os.path.exists(profile_path):
            with open(profile_path, 'rb') as f:
                self.feature_profiles = pickle.load(f)
        else:
            self.feature_profiles = None
        
        return model, scaler
    
    def train_from_data_file(self, data_filepath, model_name="montlock_model"):
        """
        Train a model from a CSV file containing mouse movement data.
        
        Args:
            data_filepath (str): Path to the CSV file
            model_name (str): Name to save the model as
            
        Returns:
            tuple: (model, scaler)
        """
        # Check if file exists
        if not os.path.exists(data_filepath):
            raise FileNotFoundError(f"Data file not found: {data_filepath}")
        
        # Extract features using the feature extractor
        extractor = FeatureExtractor()
        features = extractor.extract_features_from_file(data_filepath)
        
        # Convert features dict to DataFrame if needed
        if isinstance(features, dict):
            features_df = pd.DataFrame([features])
        else:
            features_df = features
        
        # Train model
        self.model, self.scaler = self.train_model(features_df)
        
        # Save model
        self.save_model(model_name)
        
        # Skip evaluation if we only have one sample
        if len(features_df) <= 1:
            print("Skipping evaluation with single sample.")
        else:
            # Only evaluate if we have multiple samples
            evaluation = self.evaluate_model(features_df)
            print(f"Model evaluation: {evaluation}")
        
        return self.model, self.scaler


if __name__ == "__main__":
    # Example usage
    trainer = ModelTrainer()
    
    try:
        # Try to find the most recent data file
        data_dir = "data"
        if os.path.exists(data_dir):
            data_files = [f for f in os.listdir(data_dir) if f.startswith("mouse_data_") and f.endswith(".csv")]
            if data_files:
                # Sort by modification time (most recent first)
                data_files.sort(key=lambda x: os.path.getmtime(os.path.join(data_dir, x)), reverse=True)
                latest_file = os.path.join(data_dir, data_files[0])
                
                print(f"Training model using data from {latest_file}")
                model_path, scaler_path = trainer.train_from_data_file(latest_file)
                print(f"Training complete. Model saved to {model_path}")
            else:
                print("No data files found. Run data_collector.py first to generate data.")
        else:
            print("Data directory not found. Run data_collector.py first to generate data.")
    except Exception as e:
        print(f"Error during model training: {e}") 