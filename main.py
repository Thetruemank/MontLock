"""
MontLock - Mouse Movement Biometric Security System

This is the main entry point for the MontLock application.
It provides a command-line interface for training the model
and activating/deactivating the protection.
"""

import os
import sys
import time
import argparse
import threading
import signal
import json
from datetime import datetime

from data_collector import DataCollector
from feature_extractor import FeatureExtractor
from model_trainer import ModelTrainer
from monitor import Monitor


class MontLock:
    def __init__(self):
        """Initialize the MontLock application."""
        self.config_dir = os.path.join(os.path.expanduser("~"), ".montlock")
        self.config_file = os.path.join(self.config_dir, "config.json")
        self.pid_file = os.path.join(self.config_dir, "montlock.pid")
        
        # Create config directory if it doesn't exist
        if not os.path.exists(self.config_dir):
            os.makedirs(self.config_dir)
        
        # Load configuration
        self.config = self._load_config()
        
        # Initialize components
        self.collector = DataCollector()
        self.extractor = FeatureExtractor()
        self.trainer = ModelTrainer()
        self.monitor = None
    
    def _load_config(self):
        """
        Load configuration from file.
        
        Returns:
            dict: Configuration settings
        """
        default_config = {
            "model_name": "montlock_model",
            "training_duration": 300,  # 5 minutes
            "window_size": 100,
            "overlap": 0.5,
            "threshold": -0.2
        }
        
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    # Update with any missing default values
                    for key, value in default_config.items():
                        if key not in config:
                            config[key] = value
                    return config
            except Exception as e:
                print(f"Error loading config: {e}")
                return default_config
        else:
            # Save default config
            self._save_config(default_config)
            return default_config
    
    def _save_config(self, config):
        """
        Save configuration to file.
        
        Args:
            config (dict): Configuration settings
        """
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=4)
        except Exception as e:
            print(f"Error saving config: {e}")
    
    def _save_pid(self):
        """Save the current process ID to file."""
        try:
            with open(self.pid_file, 'w') as f:
                f.write(str(os.getpid()))
        except Exception as e:
            print(f"Error saving PID: {e}")
    
    def _is_running(self):
        """
        Check if MontLock is already running.
        
        Returns:
            bool: True if running, False otherwise
        """
        if not os.path.exists(self.pid_file):
            return False
        
        try:
            with open(self.pid_file, 'r') as f:
                pid = int(f.read().strip())
            
            # Check if process exists
            os.kill(pid, 0)
            return True
        except (OSError, ValueError):
            # Process not running or invalid PID
            return False
    
    def train(self, duration=None):
        """
        Train the model with the user's mouse movements.
        
        Args:
            duration (int, optional): Duration in seconds to collect training data
        """
        if duration is None:
            duration = self.config["training_duration"]
        
        print(f"Starting MontLock training session ({duration} seconds)...")
        print("Please use your mouse naturally during this time.")
        print("The system will collect data about your mouse movement patterns.")
        
        # Collect training data
        data_file = self.collector.collect_training_data(duration=duration)
        
        if data_file:
            print("\nTraining data collected successfully.")
            
            # Train the model
            try:
                self.trainer.train_from_data_file(data_file)
                
                print("\nTraining complete!")
                print(f"Your biometric profile has been created and saved.")
                
                # Update config
                self.config["model_name"] = "montlock_model"
                self._save_config(self.config)
                
                print("\nMontLock is now ready to protect your computer.")
                print("Run 'python main.py protect' to activate protection.")
                
                return True
            except Exception as e:
                print(f"\nError during model training: {e}")
                return False
        else:
            print("\nFailed to collect training data.")
            return False
    
    def protect(self, daemonize=True):
        """
        Start the protection service.
        
        Args:
            daemonize (bool): Whether to run as a background service
        """
        if self._is_running():
            print("MontLock is already running.")
            return False
        
        # Check if model exists
        model_name = self.config["model_name"]
        model_path = os.path.join("models", f"{model_name}.pkl")
        
        if not os.path.exists(model_path):
            print(f"Model not found: {model_path}")
            print("Please run 'python main.py train' first to create your biometric profile.")
            return False
        
        try:
            # Initialize monitor with config settings
            self.monitor = Monitor(
                model_name=model_name,
                window_size=self.config["window_size"],
                overlap=self.config["overlap"],
                threshold=self.config["threshold"]
            )
            
            if daemonize:
                # Save PID
                self._save_pid()
                
                # Start monitoring
                self.monitor.start_monitoring()
                
                print("MontLock protection activated.")
                print("Your computer is now protected by your biometric profile.")
                print("Run 'python main.py stop' to deactivate protection.")
                
                # Keep running
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    self.stop()
            else:
                # Start monitoring without daemonizing
                self.monitor.start_monitoring()
                return True
            
        except Exception as e:
            print(f"Error starting protection: {e}")
            return False
    
    def stop(self):
        """Stop the protection service."""
        if not self._is_running():
            print("MontLock is not running.")
            return False
        
        try:
            # Read PID
            with open(self.pid_file, 'r') as f:
                pid = int(f.read().strip())
            
            # If this is the process, stop monitoring
            if pid == os.getpid() and self.monitor:
                self.monitor.stop_monitoring()
                os.remove(self.pid_file)
                print("MontLock protection deactivated.")
                return True
            else:
                # Send signal to the process
                os.kill(pid, signal.SIGTERM)
                
                # Wait for process to terminate
                for _ in range(5):
                    try:
                        os.kill(pid, 0)
                        time.sleep(1)
                    except OSError:
                        break
                
                # Remove PID file
                if os.path.exists(self.pid_file):
                    os.remove(self.pid_file)
                
                print("MontLock protection deactivated.")
                return True
        
        except Exception as e:
            print(f"Error stopping protection: {e}")
            return False
    
    def status(self):
        """Check the status of the protection service."""
        if self._is_running():
            print("MontLock Status: ACTIVE")
            print("Your computer is currently protected by your biometric profile.")
            
            # Try to read PID
            try:
                with open(self.pid_file, 'r') as f:
                    pid = int(f.read().strip())
                print(f"Process ID: {pid}")
            except:
                pass
            
            return True
        else:
            print("MontLock Status: INACTIVE")
            print("Your computer is not currently protected.")
            
            # Check if model exists
            model_name = self.config["model_name"]
            model_path = os.path.join("models", f"{model_name}.pkl")
            
            if os.path.exists(model_path):
                print("A biometric profile exists and is ready to use.")
                print("Run 'python main.py protect' to activate protection.")
            else:
                print("No biometric profile found.")
                print("Run 'python main.py train' to create your profile.")
            
            return False
    
    def configure(self, **kwargs):
        """
        Update configuration settings.
        
        Args:
            **kwargs: Configuration settings to update
        """
        # Update config with provided values
        for key, value in kwargs.items():
            if key in self.config and value is not None:
                self.config[key] = value
        
        # Save updated config
        self._save_config(self.config)
        
        print("Configuration updated:")
        for key, value in self.config.items():
            print(f"  {key}: {value}")


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description="MontLock - Mouse Movement Biometric Security System")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train the model with your mouse movements")
    train_parser.add_argument("--duration", type=int, help="Duration in seconds to collect training data")
    
    # Protect command
    protect_parser = subparsers.add_parser("protect", help="Activate MontLock protection")
    
    # Stop command
    stop_parser = subparsers.add_parser("stop", help="Deactivate MontLock protection")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Check MontLock status")
    
    # Configure command
    config_parser = subparsers.add_parser("config", help="Configure MontLock settings")
    config_parser.add_argument("--threshold", type=float, help="Decision threshold for anomaly detection")
    config_parser.add_argument("--window-size", type=int, help="Number of mouse movements to analyze at once")
    config_parser.add_argument("--overlap", type=float, help="Fraction of overlap between consecutive windows")
    config_parser.add_argument("--training-duration", type=int, help="Default duration for training sessions")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create MontLock instance
    montlock = MontLock()
    
    # Execute command
    if args.command == "train":
        montlock.train(duration=args.duration)
    elif args.command == "protect":
        montlock.protect()
    elif args.command == "stop":
        montlock.stop()
    elif args.command == "status":
        montlock.status()
    elif args.command == "config":
        montlock.configure(
            threshold=args.threshold,
            window_size=args.window_size,
            overlap=args.overlap,
            training_duration=args.training_duration
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 