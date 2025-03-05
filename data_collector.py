"""
MontLock Data Collector Module

This module captures mouse movement data for training the biometric model.
It records mouse positions, timestamps, and calculates derived metrics.
"""

import time
import csv
import os
from datetime import datetime
from pynput import mouse
import numpy as np
import threading
import math

class DataCollector:
    def __init__(self, output_dir="data"):
        """
        Initialize the data collector.
        
        Args:
            output_dir (str): Directory to save collected data
        """
        self.output_dir = output_dir
        self.data = []
        self.collecting = False
        self.last_position = None
        self.last_time = None
        self.lock = threading.Lock()
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
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
                self.data.append({
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
    
    def start_collection(self, duration=60):
        """
        Start collecting mouse movement data for the specified duration.
        
        Args:
            duration (int): Duration in seconds to collect data
        
        Returns:
            bool: True if collection completed successfully
        """
        print(f"Starting data collection for {duration} seconds...")
        print("Please use your mouse naturally during this time.")
        
        self.data = []
        self.last_position = None
        self.last_time = None
        self.collecting = True
        
        # Start mouse listener
        listener = mouse.Listener(on_move=self._on_move)
        listener.start()
        
        # Collect for the specified duration
        time.sleep(duration)
        
        # Stop collection
        self.collecting = False
        listener.stop()
        
        print(f"Data collection complete. Collected {len(self.data)} data points.")
        return len(self.data) > 0
    
    def save_data(self, filename=None):
        """
        Save collected data to a CSV file.
        
        Args:
            filename (str, optional): Name of the output file
        
        Returns:
            str: Path to the saved file
        """
        if not self.data:
            print("No data to save.")
            return None
        
        if filename is None:
            # Generate filename based on current timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"mouse_data_{timestamp}.csv"
        
        filepath = os.path.join(self.output_dir, filename)
        
        # Save data to CSV
        with open(filepath, 'w', newline='') as csvfile:
            fieldnames = ['timestamp', 'x', 'y', 'time_diff', 'distance', 'speed', 'direction']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for row in self.data:
                writer.writerow(row)
        
        print(f"Data saved to {filepath}")
        return filepath
    
    def collect_training_data(self, duration=300, filename=None):
        """
        Collect and save training data in one operation.
        
        Args:
            duration (int): Duration in seconds to collect data
            filename (str, optional): Name of the output file
        
        Returns:
            str: Path to the saved file, or None if collection failed
        """
        success = self.start_collection(duration)
        if success:
            return self.save_data(filename)
        return None


if __name__ == "__main__":
    # Example usage
    collector = DataCollector()
    collector.collect_training_data(duration=30) 