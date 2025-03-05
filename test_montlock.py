"""
MontLock Test Script

This script demonstrates the functionality of MontLock by:
1. Collecting a small amount of training data
2. Training a model on this data
3. Running a brief monitoring session
4. Testing the cursor locking mechanism
"""

import time
import os
import sys
from data_collector import DataCollector
from feature_extractor import FeatureExtractor
from model_trainer import ModelTrainer
from monitor import Monitor
from cursor_locker import CursorLocker


def test_data_collection():
    """Test the data collection functionality."""
    print("\n=== Testing Data Collection ===")
    collector = DataCollector()
    
    print("Collecting mouse movement data for 10 seconds...")
    print("Please move your mouse naturally during this time.")
    
    data_file = collector.collect_training_data(duration=10)
    
    if data_file:
        print(f"Data collection successful: {data_file}")
        return data_file
    else:
        print("Data collection failed.")
        return None


def test_feature_extraction(data_file):
    """Test the feature extraction functionality."""
    print("\n=== Testing Feature Extraction ===")
    extractor = FeatureExtractor()
    
    try:
        features = extractor.extract_features_from_file(data_file)
        print("Feature extraction successful.")
        print("Extracted features:")
        
        # Print a few key features
        important_features = [
            'speed_mean', 'speed_std', 'reaction_time_mean', 
            'curvature_mean', 'straightness', 'pause_frequency'
        ]
        
        for feature in important_features:
            if feature in features:
                print(f"  {feature}: {features[feature]:.4f}")
        
        return features
    except Exception as e:
        print(f"Feature extraction failed: {e}")
        return None


def test_model_training(data_file):
    """Test the model training functionality."""
    print("\n=== Testing Model Training ===")
    trainer = ModelTrainer()
    
    try:
        print("Training model on collected data...")
        model_path, scaler_path = trainer.train_from_data_file(data_file, model_name="test_model")
        print(f"Model training successful:")
        print(f"  Model saved to: {model_path}")
        print(f"  Scaler saved to: {scaler_path}")
        return "test_model"
    except Exception as e:
        print(f"Model training failed: {e}")
        return None


def test_monitoring(model_name):
    """Test the monitoring functionality."""
    print("\n=== Testing Monitoring ===")
    
    try:
        monitor = Monitor(model_name=model_name, window_size=50)
        print("Starting monitoring for 10 seconds...")
        monitor.start_monitoring()
        
        # Run for a short time
        time.sleep(10)
        
        # Stop monitoring
        monitor.stop_monitoring()
        print("Monitoring test completed.")
        return True
    except Exception as e:
        print(f"Monitoring test failed: {e}")
        return False


def test_cursor_locking():
    """Test the cursor locking functionality."""
    print("\n=== Testing Cursor Locking ===")
    locker = CursorLocker()
    
    print("Locking cursor for 3 seconds...")
    print("Try to move your mouse during this time.")
    
    locker.lock_cursor(duration=3)
    time.sleep(4)  # Wait for lock to release
    
    print("Cursor locking test completed.")
    return True


def run_all_tests():
    """Run all tests in sequence."""
    print("Starting MontLock tests...")
    
    # Test data collection
    data_file = test_data_collection()
    if not data_file:
        print("Test failed at data collection stage.")
        return False
    
    # Test feature extraction
    features = test_feature_extraction(data_file)
    if not features:
        print("Test failed at feature extraction stage.")
        return False
    
    # Test model training
    model_name = test_model_training(data_file)
    if not model_name:
        print("Test failed at model training stage.")
        return False
    
    # Test monitoring
    monitoring_success = test_monitoring(model_name)
    if not monitoring_success:
        print("Test failed at monitoring stage.")
        return False
    
    # Test cursor locking
    locking_success = test_cursor_locking()
    if not locking_success:
        print("Test failed at cursor locking stage.")
        return False
    
    print("\n=== All Tests Completed Successfully ===")
    print("MontLock is functioning correctly.")
    return True


if __name__ == "__main__":
    run_all_tests() 