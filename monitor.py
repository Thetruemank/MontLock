"""
MontLock Monitor Module

This module continuously monitors mouse movements in real-time,
extracts features, and compares them against the trained model
to detect unauthorized users.
"""

import time
import threading
import numpy as np
import pandas as pd
from pynput import mouse
from collections import deque
import math
import os

from feature_extractor import FeatureExtractor
from model_trainer import ModelTrainer
from cursor_locker import CursorLocker


class Monitor:
    def __init__(self, model_name="montlock_model", window_size=100, overlap=0.5, threshold=-0.1):
        """
        Initialize the monitor.
        
        Args:
            model_name (str): Name of the model to load
            window_size (int): Number of mouse movements to collect before analysis
            overlap (float): Fraction of overlap between consecutive windows (0-1)
            threshold (float): Decision threshold for anomaly detection
        """
        self.model_name = model_name
        self.window_size = window_size
        self.overlap = overlap
        self.threshold = threshold
        
        # Load model and scaler
        self.trainer = ModelTrainer()
        try:
            self.model, self.scaler = self.trainer.load_model(model_name)
            print(f"Loaded model: {model_name}")
            
            # Check if feature profiles were loaded
            self.feature_profiles = getattr(self.trainer, 'feature_profiles', None)
            if self.feature_profiles:
                print("Loaded statistical profiles for features")
                
                # Define key features for detection
                self.key_features = [
                    'speed_mean', 'speed_std', 'speed_median', 
                    'reaction_time_mean', 'direction_change_frequency',
                    'acceleration_mean', 'acceleration_std',
                    'curvature_mean', 'curvature_std',
                    'straightness', 'pause_frequency'
                ]
                
                # Print ranges for key features
                print("Feature profiles for key metrics:")
                for feature in self.key_features:
                    if feature in self.feature_profiles:
                        profile = self.feature_profiles[feature]
                        print(f"  {feature}: {profile['lower_bound']:.2f} to {profile['upper_bound']:.2f}")
            else:
                print("No statistical profiles available")
        except FileNotFoundError:
            print(f"Model {model_name} not found. Please train a model first.")
            raise
        
        # Initialize feature extractor
        self.extractor = FeatureExtractor()
        
        # Initialize cursor locker
        self.locker = CursorLocker()
        
        # Data collection
        self.data_buffer = deque(maxlen=window_size)
        self.collecting = False
        self.monitoring = False
        self.last_position = None
        self.last_time = None
        
        # Monitoring thread
        self.monitor_thread = None
        self.lock = threading.Lock()
        
        # Statistics
        self.total_windows = 0
        self.unauthorized_windows = 0
        self.last_scores = deque(maxlen=10)  # Keep last 10 scores for smoothing
    
    def _on_move(self, x, y):
        """
        Callback function for mouse movement.
        
        Args:
            x (int): X-coordinate of mouse position
            y (int): Y-coordinate of mouse position
        """
        if not self.collecting:
            return
        
        current_time = time.time()
        
        with self.lock:
            if self.last_position is not None and self.last_time is not None:
                # Calculate time difference
                time_diff = current_time - self.last_time
                
                # Calculate distance and speed
                last_x, last_y = self.last_position
                distance = math.sqrt((x - last_x)**2 + (y - last_y)**2)
                speed = distance / time_diff if time_diff > 0 else 0
                
                # Calculate direction (angle in radians)
                direction = math.atan2(y - last_y, x - last_x)
                
                # Record the data point
                self.data_buffer.append({
                    'timestamp': current_time,
                    'x': x,
                    'y': y,
                    'time_diff': time_diff,
                    'distance': distance,
                    'speed': speed,
                    'direction': direction
                })
            
            # Update last position and time
            self.last_position = (x, y)
            self.last_time = current_time
    
    def _analyze_window(self):
        """
        Analyze the current window of mouse movements.
        
        Returns:
            tuple: (is_authorized, anomaly_score)
        """
        with self.lock:
            # Check if we have enough data
            if len(self.data_buffer) < self.window_size * 0.5:  # Allow partial windows
                return True, 0.0
            
            # Convert buffer to DataFrame
            df = pd.DataFrame(list(self.data_buffer))
            
            try:
                # Extract features
                features = self.extractor.extract_features(df)
                
                # Handle NaN values
                if features is None:
                    return True, 0.0
                
                for col in features.keys():
                    if pd.isna(features[col]):
                        features[col] = 0
                
                # Print key features for debugging
                print(f"Speed stats: mean={features['speed_mean']:.2f}, std={features['speed_std']:.2f}")
                print(f"Curvature: mean={features['curvature_mean']:.2f}, std={features['curvature_std']:.2f}")
                print(f"Straightness: {features['straightness']:.2f}")
                
                # Statistical profile-based detection
                if self.feature_profiles:
                    out_of_range_features = []
                    deviation_scores = []
                    
                    # Define bounds for detection - more sensitive approach
                    std_multiplier = 2.5  # Reduced from 3.0 to 2.5 for more sensitivity
                    
                    # Skip detection if there's very little movement
                    if features['speed_mean'] < 20:
                        # Skip statistical detection for very low movement
                        pass
                    else:
                        for feature in self.key_features:
                            if feature in features and feature in self.feature_profiles:
                                value = features[feature]
                                profile = self.feature_profiles[feature]
                                
                                # Skip straightness check during periods of low movement
                                if feature == 'straightness' and features['speed_mean'] < 50:
                                    continue
                                
                                # Skip certain features during low movement periods
                                if features['speed_mean'] < 100 and feature in ['speed_std', 'acceleration_mean', 'acceleration_std']:
                                    continue
                                
                                # Calculate bounds for detection
                                lower_bound = profile['mean'] - std_multiplier * profile['std']
                                upper_bound = profile['mean'] + std_multiplier * profile['std']
                                
                                # Check if value is outside the acceptable range
                                if value < lower_bound or value > upper_bound:
                                    # Calculate how many standard deviations away from the mean
                                    if profile['std'] > 0:
                                        deviation = abs(value - profile['mean']) / profile['std']
                                        deviation_scores.append(deviation)
                                        
                                        out_of_range_features.append({
                                            'feature': feature,
                                            'value': value,
                                            'expected_range': (lower_bound, upper_bound),
                                            'deviations': deviation
                                        })
                    
                    # Determine if the pattern is anomalous based on statistical profiles
                    is_anomalous = False
                    anomaly_reason = ""
                    
                    # More sensitive detection criteria
                    if len(out_of_range_features) >= 4 and features['speed_mean'] > 50:  # Reduced from 5 to 4
                        is_anomalous = True
                        anomaly_reason = f"{len(out_of_range_features)} features out of normal range"
                    
                    # Require fewer extreme deviations
                    elif any(score > 6 for score in deviation_scores) and len([s for s in deviation_scores if s > 6]) >= 2:  # Reduced from 8 to 6
                        is_anomalous = True
                        max_dev = max(deviation_scores)
                        anomaly_reason = f"Extreme deviation detected ({max_dev:.1f} std)"
                    
                    # Require lower average deviation
                    elif deviation_scores and np.mean(deviation_scores) > 5 and len(deviation_scores) >= 3:  # Reduced from 6 to 5
                        is_anomalous = True
                        avg_dev = np.mean(deviation_scores)
                        anomaly_reason = f"High average deviation ({avg_dev:.1f} std)"
                    
                    # Add detection for key biometric features that are strong indicators
                    elif any(feature_info['feature'] in ['curvature_mean', 'straightness', 'direction_change_frequency', 'speed_mean'] 
                             and feature_info['deviations'] > 4 for feature_info in out_of_range_features) and features['speed_mean'] > 50:  # Reduced from 5 to 4
                        is_anomalous = True
                        anomaly_reason = "Key biometric features out of range"
                    
                    # Print detection results
                    if is_anomalous:
                        print("STATISTICAL ANOMALY DETECTED: " + anomaly_reason)
                        for feature_info in out_of_range_features:
                            print(f"  - {feature_info['feature']}: {feature_info['value']:.2f} " +
                                  f"(expected: {feature_info['expected_range'][0]:.2f} to {feature_info['expected_range'][1]:.2f}, " +
                                  f"{feature_info['deviations']:.1f} std)")
                
                # Scale features
                features_scaled = self.scaler.transform(pd.DataFrame([features]))
                
                # Get anomaly score from model
                score = self.model.decision_function(features_scaled)[0]
                
                # Add to score history
                self.last_scores.append(score)
                
                # Smooth score using moving average
                smoothed_score = np.mean(self.last_scores)
                
                # Determine if authorized based on both model score and statistical profile
                is_authorized_by_model = smoothed_score >= self.threshold
                is_authorized = is_authorized_by_model
                
                # If we have statistical profiles, also consider those results
                # Only consider statistical anomalies if there's significant movement
                if self.feature_profiles and features['speed_mean'] > 50:  # Kept at 50
                    is_authorized = is_authorized and not is_anomalous
                
                # Update statistics
                self.total_windows += 1
                if not is_authorized:
                    self.unauthorized_windows += 1
                
                return is_authorized, smoothed_score
            
            except Exception as e:
                print(f"Error analyzing window: {e}")
                return True, 0.0  # Default to authorized on error
    
    def _monitoring_loop(self):
        """
        Main monitoring loop that periodically analyzes collected data.
        """
        step_size = int(self.window_size * (1 - self.overlap))
        steps_taken = 0
        
        while self.monitoring:
            # Wait until we have enough data
            if len(self.data_buffer) >= self.window_size * 0.5:
                # Only analyze every step_size new data points
                if steps_taken % step_size == 0:
                    is_authorized, score = self._analyze_window()
                    
                    print(f"Authorization check: {'PASS' if is_authorized else 'FAIL'} (score: {score:.4f})")
                    
                    if not is_authorized:
                        print("UNAUTHORIZED USER DETECTED! Locking cursor for 10 seconds.")
                        self.locker.lock_cursor(duration=10)
                
                steps_taken += 1
            
            # Sleep to reduce CPU usage
            time.sleep(0.1)
    
    def start_monitoring(self):
        """
        Start monitoring mouse movements.
        """
        if self.monitoring:
            print("Monitoring is already active.")
            return
        
        print("Starting MontLock monitoring...")
        
        # Reset data
        self.data_buffer.clear()
        self.last_position = None
        self.last_time = None
        self.total_windows = 0
        self.unauthorized_windows = 0
        self.last_scores.clear()
        
        # Start collecting data
        self.collecting = True
        
        # Start mouse listener
        self.listener = mouse.Listener(on_move=self._on_move)
        self.listener.start()
        
        # Start monitoring thread
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        print("MontLock is now protecting your computer.")
    
    def stop_monitoring(self):
        """
        Stop monitoring mouse movements.
        """
        if not self.monitoring:
            print("Monitoring is not active.")
            return
        
        print("Stopping MontLock monitoring...")
        
        # Stop monitoring thread
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        
        # Stop collecting data
        self.collecting = False
        
        # Stop mouse listener
        if hasattr(self, 'listener') and self.listener.is_alive():
            self.listener.stop()
        
        # Print statistics
        if self.total_windows > 0:
            unauthorized_percent = (self.unauthorized_windows / self.total_windows) * 100
            print(f"Monitoring statistics:")
            print(f"  Total windows analyzed: {self.total_windows}")
            print(f"  Unauthorized detections: {self.unauthorized_windows} ({unauthorized_percent:.1f}%)")
        
        print("MontLock monitoring stopped.")
    
    def adjust_threshold(self, new_threshold):
        """
        Adjust the decision threshold for anomaly detection.
        
        Args:
            new_threshold (float): New threshold value
        """
        self.threshold = new_threshold
        print(f"Decision threshold adjusted to {new_threshold}")


if __name__ == "__main__":
    # Example usage
    try:
        monitor = Monitor()
        monitor.start_monitoring()
        
        print("Press Ctrl+C to stop monitoring...")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        
        monitor.stop_monitoring()
    
    except FileNotFoundError:
        print("Model not found. Please run model_trainer.py first to train a model.")
    except Exception as e:
        print(f"Error: {e}") 