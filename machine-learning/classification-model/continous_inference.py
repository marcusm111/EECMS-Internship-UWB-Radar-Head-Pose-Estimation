"""
Continuous Inference Module for Head Movement Classification

This module provides functionality for continuously monitoring a directory for new sensor data files,
processing them as they arrive, and running inference using the trained head movement classification model.
It's designed to work alongside a MATLAB script that generates the sensor data files in real-time.

Example usage:
    model, device, classes, stats = load_model()
    start_continuous_inference(
        model=model,
        device=device, 
        class_names=classes, 
        norm_stats=stats,
        input_dir="sensor_data",
        archive_dir="processed_data",
        results_file="inference_results.csv"
    )
"""

import os
import time
import csv
import shutil
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Set

import torch

# Import from your existing modules
from inference import load_model, load_spectrogram, run_inference

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("continuous_inference.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def setup_directories(input_dir: str, archive_dir: str) -> Tuple[str, str]:
    """
    Set up the necessary directories for file monitoring.
    
    Args:
        input_dir: Directory to monitor for new sensor files
        archive_dir: Directory to move processed files to
        
    Returns:
        input_dir: Absolute path to input directory
        archive_dir: Absolute path to archive directory
    """
    # Create absolute paths
    input_dir = os.path.abspath(input_dir)
    archive_dir = os.path.abspath(archive_dir)
    
    # Create directories if they don't exist
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(archive_dir, exist_ok=True)
    
    logger.info(f"Monitoring directory: {input_dir}")
    logger.info(f"Archive directory: {archive_dir}")
    
    return input_dir, archive_dir


def setup_results_file(results_file: str) -> str:
    """
    Set up the CSV file for storing inference results.
    
    Args:
        results_file: Path to the results file
        
    Returns:
        results_file: Absolute path to results file
    """
    results_file = os.path.abspath(results_file)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    
    # Initialize the results file with headers if it doesn't exist
    if not os.path.exists(results_file):
        with open(results_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Timestamp', 
                'Front File', 
                'Left File', 
                'Right File', 
                'Prediction', 
                'Confidence',
                'Probabilities'
            ])
    
    logger.info(f"Results will be saved to: {results_file}")
    return results_file


def find_matching_sensor_files(
    front_file: str, 
    input_dir: str
) -> Tuple[Optional[str], Optional[str]]:
    """
    Find matching left and right sensor files for a given front sensor file.
    
    Args:
        front_file: Front sensor file name (contains 'sF')
        input_dir: Directory containing sensor files
        
    Returns:
        left_file: Matching left sensor file path or None
        right_file: Matching right sensor file path or None
    """
    # Create the expected filenames for left and right sensors
    base_name = front_file.replace('sF', 's')
    left_file = os.path.join(input_dir, base_name.replace('s', 'sL'))
    right_file = os.path.join(input_dir, base_name.replace('s', 'sR'))
    
    # Check if both files exist
    if os.path.exists(left_file) and os.path.exists(right_file):
        return left_file, right_file
    
    return None, None


def archive_files(
    front_file: str, 
    left_file: str, 
    right_file: str, 
    archive_dir: str
) -> bool:
    """
    Move processed sensor files to the archive directory.
    
    Args:
        front_file: Path to front sensor file
        left_file: Path to left sensor file
        right_file: Path to right sensor file
        archive_dir: Directory to move processed files to
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create a subdirectory with timestamp to group related files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.basename(front_file).replace('sF', '')
        group_dir = os.path.join(archive_dir, f"{timestamp}_{base_name.replace('.txt', '')}")
        os.makedirs(group_dir, exist_ok=True)
        
        # Move files to archive
        shutil.move(front_file, os.path.join(group_dir, os.path.basename(front_file)))
        shutil.move(left_file, os.path.join(group_dir, os.path.basename(left_file)))
        shutil.move(right_file, os.path.join(group_dir, os.path.basename(right_file)))
        
        logger.debug(f"Archived files to {group_dir}")
        return True
    except Exception as e:
        logger.error(f"Error archiving files: {e}")
        return False


def save_results(
    results_file: str,
    front_file: str,
    left_file: str,
    right_file: str,
    prediction: str,
    confidence: float,
    probabilities: Dict[str, float]
) -> None:
    """
    Save inference results to the CSV file.
    
    Args:
        results_file: Path to the results CSV file
        front_file: Path to front sensor file
        left_file: Path to left sensor file
        right_file: Path to right sensor file
        prediction: Predicted class
        confidence: Confidence score
        probabilities: Dictionary of class probabilities
    """
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Format probabilities as a string
        probs_str = ";".join([f"{k}:{v:.4f}" for k, v in probabilities.items()])
        
        with open(results_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                os.path.basename(front_file),
                os.path.basename(left_file),
                os.path.basename(right_file),
                prediction,
                f"{confidence:.4f}",
                probs_str
            ])
        
        logger.debug(f"Results saved to {results_file}")
    except Exception as e:
        logger.error(f"Error saving results: {e}")


def process_file_set(
    front_file: str,
    left_file: str,
    right_file: str,
    model: torch.nn.Module,
    device: torch.device,
    class_names: List[str],
    norm_stats: Dict[str, float],
    results_file: str,
    archive_dir: str
) -> bool:
    """
    Process a complete set of sensor files (front, left, right).
    
    Args:
        front_file: Path to front sensor file
        left_file: Path to left sensor file
        right_file: Path to right sensor file
        model: Trained PyTorch model
        device: PyTorch device
        class_names: List of class names
        norm_stats: Dictionary with mean and std values for normalization
        results_file: Path to the results CSV file
        archive_dir: Directory to move processed files to
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Load the spectrogram
        tensor = load_spectrogram(front_file)
        
        # Run inference
        prediction, confidence, probabilities = run_inference(
            model, tensor, device, class_names,
            mean=norm_stats['mean'], std=norm_stats['std']
        )
        
        # Log the results
        logger.info(f"Prediction: {prediction} with {confidence:.2%} confidence")
        logger.info(f"Top probabilities: {sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:3]}")
        
        # Save results to CSV
        save_results(
            results_file,
            front_file,
            left_file,
            right_file,
            prediction,
            confidence,
            probabilities
        )
        
        # Archive the files
        archive_files(front_file, left_file, right_file, archive_dir)
        
        return True
    except Exception as e:
        logger.error(f"Error processing files: {e}")
        return False


def continuous_inference_loop(
    model: torch.nn.Module,
    device: torch.device,
    class_names: List[str],
    norm_stats: Dict[str, float],
    input_dir: str,
    archive_dir: str,
    results_file: str,
    poll_interval: float = 0.5
) -> None:
    """
    Main loop for continuous inference.
    
    Args:
        model: Trained PyTorch model
        device: PyTorch device
        class_names: List of class names
        norm_stats: Dictionary with mean and std values for normalization
        input_dir: Directory to monitor for new sensor files
        archive_dir: Directory to move processed files to
        results_file: Path to the results CSV file
        poll_interval: Time in seconds between directory checks
    """
    processed_files = set()  # Track processed front files
    
    logger.info("Starting continuous inference loop...")
    logger.info(f"Monitoring directory: {input_dir}")
    logger.info(f"Poll interval: {poll_interval}s")
    
    try:
        while True:
            # Find all front sensor files in the directory
            front_files = [
                os.path.join(input_dir, f) for f in os.listdir(input_dir)
                if f.endswith(".txt") and "sF" in f and os.path.join(input_dir, f) not in processed_files
            ]
            
            if front_files:
                logger.debug(f"Found {len(front_files)} new front sensor files")
            
            # Process each front file if matching left and right files exist
            for front_file in front_files:
                left_file, right_file = find_matching_sensor_files(
                    os.path.basename(front_file), 
                    input_dir
                )
                
                if left_file and right_file:
                    logger.info(f"Processing complete set for {os.path.basename(front_file)}")
                    
                    # Process the file set
                    success = process_file_set(
                        front_file,
                        left_file,
                        right_file,
                        model,
                        device,
                        class_names,
                        norm_stats,
                        results_file,
                        archive_dir
                    )
                    
                    if success:
                        # Mark as processed
                        processed_files.add(front_file)
                        
                        # Keep the processed set at a reasonable size
                        if len(processed_files) > 1000:
                            # Remove oldest entries (assuming chronological processing)
                            processed_files = set(list(processed_files)[-500:])
            
            # Sleep to prevent high CPU usage
            time.sleep(poll_interval)
            
    except KeyboardInterrupt:
        logger.info("Continuous inference stopped by user")
    except Exception as e:
        logger.error(f"Continuous inference loop error: {e}")
        raise


def start_continuous_inference(
    model: torch.nn.Module = None,
    device: torch.device = None,
    class_names: List[str] = None,
    norm_stats: Dict[str, float] = None,
    input_dir: str = "sensor_data",
    archive_dir: str = "processed_data",
    results_file: str = "inference_results.csv",
    model_path: str = "final_head_movement_model.pth",
    config_path: str = "final_config.yaml",
    poll_interval: float = 0.5
) -> None:
    """
    Start the continuous inference system.
    
    Args:
        model: Trained PyTorch model (optional, will be loaded if None)
        device: PyTorch device (optional, will be set if None)
        class_names: List of class names (optional, will be loaded if None)
        norm_stats: Dictionary with mean and std values (optional, will be loaded if None)
        input_dir: Directory to monitor for new sensor files
        archive_dir: Directory to move processed files to
        results_file: Path to the results CSV file
        model_path: Path to the saved model weights file (used if model is None)
        config_path: Path to the configuration file (used if model is None)
        poll_interval: Time in seconds between directory checks
    """
    # Set up directories
    input_dir, archive_dir = setup_directories(input_dir, archive_dir)
    
    # Set up results file
    results_file = setup_results_file(results_file)
    
    # Load model if not provided
    if model is None or device is None or class_names is None or norm_stats is None:
        logger.info(f"Loading model from {model_path}")
        model, device, class_names, norm_stats = load_model(model_path, config_path)
    
    # Start the inference loop
    continuous_inference_loop(
        model=model,
        device=device,
        class_names=class_names,
        norm_stats=norm_stats,
        input_dir=input_dir,
        archive_dir=archive_dir,
        results_file=results_file,
        poll_interval=poll_interval
    )


if __name__ == "__main__":
    # Example usage when run as a script
    start_continuous_inference(
        input_dir="sensor_data",
        archive_dir="processed_data",
        results_file="results/inference_results.csv",
        poll_interval=0.5
    )