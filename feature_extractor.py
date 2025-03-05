"""
MontLock Feature Extractor Module

This module processes raw mouse movement data into biometric features
that can be used for user authentication.
"""

import pandas as pd
import numpy as np
from scipy import stats
import math
import os


class FeatureExtractor:
    def __init__(self):
        """Initialize the feature extractor."""
        pass
    
    def load_data(self, filepath):
        """
        Load mouse movement data from a CSV file.
        
        Args:
            filepath (str): Path to the CSV file
            
        Returns:
            pandas.DataFrame: Loaded data
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        data = pd.read_csv(filepath)
        return data
    
    def extract_features(self, data):
        """
        Extract biometric features from mouse movement data.
        
        Args:
            data (pandas.DataFrame): Mouse movement data
            
        Returns:
            dict: Dictionary of extracted features
        """
        if isinstance(data, str):
            data = self.load_data(data)
        
        # Ensure we have enough data points
        if len(data) < 10:
            raise ValueError("Not enough data points for feature extraction")
        
        features = {}
        
        # 1. Speed-related features
        speeds = data['speed'].values
        features['speed_mean'] = np.mean(speeds)
        features['speed_std'] = np.std(speeds)
        features['speed_median'] = np.median(speeds)
        features['speed_min'] = np.min(speeds)
        features['speed_max'] = np.max(speeds)
        features['speed_range'] = features['speed_max'] - features['speed_min']
        features['speed_skew'] = stats.skew(speeds)
        features['speed_kurtosis'] = stats.kurtosis(speeds)
        
        # 2. Reaction time features (time between direction changes)
        directions = data['direction'].values
        direction_changes = np.diff(directions)
        direction_change_indices = np.where(np.abs(direction_changes) > 0.5)[0]
        
        if len(direction_change_indices) > 1:
            reaction_times = np.diff(data.iloc[direction_change_indices]['timestamp'].values)
            features['reaction_time_mean'] = np.mean(reaction_times)
            features['reaction_time_std'] = np.std(reaction_times)
            features['reaction_time_median'] = np.median(reaction_times)
        else:
            # Default values if not enough direction changes
            features['reaction_time_mean'] = 0
            features['reaction_time_std'] = 0
            features['reaction_time_median'] = 0
        
        # 3. Direction change frequency
        features['direction_change_frequency'] = len(direction_change_indices) / len(data)
        
        # 4. Acceleration features
        accelerations = np.diff(speeds) / np.diff(data['timestamp'].values)
        features['acceleration_mean'] = np.mean(accelerations)
        features['acceleration_std'] = np.std(accelerations)
        features['acceleration_max'] = np.max(np.abs(accelerations))
        
        # 5. Jerk features (rate of change of acceleration)
        jerks = np.diff(accelerations) / np.diff(data['timestamp'].values[:-1])
        if len(jerks) > 0:
            features['jerk_mean'] = np.mean(jerks)
            features['jerk_std'] = np.std(jerks)
            features['jerk_max'] = np.max(np.abs(jerks))
        else:
            features['jerk_mean'] = 0
            features['jerk_std'] = 0
            features['jerk_max'] = 0
        
        # 6. Curvature features (how curved the mouse path is)
        x_coords = data['x'].values
        y_coords = data['y'].values
        
        # Calculate path curvature
        dx = np.diff(x_coords)
        dy = np.diff(y_coords)
        
        # Avoid division by zero
        dx[dx == 0] = 1e-10
        
        # Calculate angles between consecutive segments
        angles = np.arctan2(dy, dx)
        angle_changes = np.diff(angles)
        
        # Normalize angle changes to [-pi, pi]
        angle_changes = np.mod(angle_changes + np.pi, 2 * np.pi) - np.pi
        
        features['curvature_mean'] = np.mean(np.abs(angle_changes))
        features['curvature_std'] = np.std(angle_changes)
        
        # 7. Straightness features (ratio of direct distance to actual path length)
        if len(data) > 1:
            start_point = np.array([x_coords[0], y_coords[0]])
            end_point = np.array([x_coords[-1], y_coords[-1]])
            direct_distance = np.linalg.norm(end_point - start_point)
            
            path_distances = np.sqrt(dx**2 + dy**2)
            path_length = np.sum(path_distances)
            
            if path_length > 0:
                features['straightness'] = direct_distance / path_length
            else:
                features['straightness'] = 1.0
        else:
            features['straightness'] = 1.0
        
        # 8. Pause features (moments of very low speed)
        pause_threshold = 1.0  # pixels per second
        pauses = speeds < pause_threshold
        features['pause_frequency'] = np.sum(pauses) / len(speeds)
        
        # 9. Direction distribution features
        direction_bins = np.linspace(-np.pi, np.pi, 9)  # 8 direction bins
        hist, _ = np.histogram(data['direction'], bins=direction_bins, density=True)
        for i, count in enumerate(hist):
            features[f'direction_bin_{i}'] = count
        
        # 10. Movement patterns
        # Calculate time spent in each screen quadrant
        screen_width = np.max(x_coords) - np.min(x_coords)
        screen_height = np.max(y_coords) - np.min(y_coords)
        
        if screen_width > 0 and screen_height > 0:
            mid_x = np.min(x_coords) + screen_width / 2
            mid_y = np.min(y_coords) + screen_height / 2
            
            quadrant_1 = np.sum((x_coords >= mid_x) & (y_coords < mid_y)) / len(x_coords)
            quadrant_2 = np.sum((x_coords < mid_x) & (y_coords < mid_y)) / len(x_coords)
            quadrant_3 = np.sum((x_coords < mid_x) & (y_coords >= mid_y)) / len(x_coords)
            quadrant_4 = np.sum((x_coords >= mid_x) & (y_coords >= mid_y)) / len(x_coords)
            
            features['quadrant_1_time'] = quadrant_1
            features['quadrant_2_time'] = quadrant_2
            features['quadrant_3_time'] = quadrant_3
            features['quadrant_4_time'] = quadrant_4
        else:
            features['quadrant_1_time'] = 0.25
            features['quadrant_2_time'] = 0.25
            features['quadrant_3_time'] = 0.25
            features['quadrant_4_time'] = 0.25
        
        return features
    
    def extract_features_from_file(self, filepath):
        """
        Load data from file and extract features.
        
        Args:
            filepath (str): Path to the CSV file
            
        Returns:
            dict: Dictionary of extracted features
        """
        data = self.load_data(filepath)
        return self.extract_features(data)
    
    def extract_features_batch(self, data_list):
        """
        Extract features from multiple data sources.
        
        Args:
            data_list (list): List of DataFrames or file paths
            
        Returns:
            pandas.DataFrame: DataFrame of extracted features
        """
        all_features = []
        
        for data in data_list:
            features = self.extract_features(data)
            all_features.append(features)
        
        return pd.DataFrame(all_features)


if __name__ == "__main__":
    # Example usage
    extractor = FeatureExtractor()
    # Assuming there's a sample data file
    try:
        features = extractor.extract_features_from_file("data/mouse_data_sample.csv")
        print("Extracted features:")
        for key, value in features.items():
            print(f"{key}: {value}")
    except FileNotFoundError:
        print("Sample data file not found. Run data_collector.py first to generate data.") 