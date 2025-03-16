"""
Head Movement Classification Inference Module

This module provides functionality for loading trained models and performing inference
on time-frequency map data to classify head movements. It handles loading model weights,
preparing input data, and generating predictions with confidence scores.

Example usage:
    model, device, classes, stats = load_model()
    tensor = load_spectrogram("path/to/spectrogram.pt")
    prediction, confidence, probabilities = run_inference(
        model, tensor, device, classes, stats['mean'], stats['std']
    )
"""

import os
import logging
from typing import Dict, List, Tuple, Union, Optional

import torch
import torch.nn.functional as F

from model import load_modified_model
from utils import load_config
from timefrequencydataset import TimeFrequencyMapDataset
from preprocessing import torch_genfromtxt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model(
    model_path: str = "final_head_movement_model.pth", 
    config_path: str = "final_config.yaml"
) -> Tuple[torch.nn.Module, torch.device, List[str], Dict[str, float]]:
    """
    Load the trained model with parameters from the configuration file.
    
    Args:
        model_path: Path to the saved model weights file
        config_path: Path to the configuration file
        
    Returns:
        model: Loaded PyTorch model
        device: Device the model is loaded on
        class_names: List of class names
        normalization_stats: Dictionary with mean and std values for normalization
        
    Raises:
        FileNotFoundError: If model weights or config file not found
    """
    # Check if files exist
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights file not found: {model_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load configuration
    config = load_config(config_path)
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create model with same architecture as training
    model_params = config["default_params"]["model_params"]
    model = load_modified_model(
        num_classes=config["num_classes"],
        **model_params
    ).to(device)
    
    # Load the trained weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set model to evaluation mode
    
    # Default class names if dataset can't be accessed
    default_class_names = [
        "Down", "No_Movement", "Right30", "Right60", "Right90", 
        "Left90", "Left30", "Left60", "Up"
    ]
    
    # Use default stats for normalization
    normalization_stats = {
        'mean': 0.0,
        'std': 1.0
    }
    
    # Get class names and normalization stats if possible
    training_dir = os.path.join(config["data"], "training_data")
    tensor_path = os.path.join(training_dir, "tensor_data")
    
    # Try to get class names and stats from dataset
    try:
        if os.path.exists(tensor_path):
            dataset = TimeFrequencyMapDataset(tensor_path, compute_stats=True)
            class_names = dataset.classes
            if hasattr(dataset, 'mean') and hasattr(dataset, 'std'):
                normalization_stats['mean'] = dataset.mean.item()
                normalization_stats['std'] = dataset.std.item()
        else:
            logger.warning(f"Tensor path not found: {tensor_path}")
            logger.warning(f"Using default class names: {default_class_names}")
            class_names = default_class_names
    except Exception as e:
        logger.warning(f"Error accessing dataset: {str(e)}")
        logger.warning(f"Using default class names: {default_class_names}")
        class_names = default_class_names
    
    return model, device, class_names, normalization_stats


def load_spectrogram(file_path: str) -> torch.Tensor:
    """
    Load a spectrogram file and convert it to tensor format.
    
    Args:
        file_path: Path to the spectrogram file (.pt or triplet of .txt files)
        
    Returns:
        Processed tensor ready for inference (shape: [3, H, W])
        
    Raises:
        ValueError: If input file format is invalid
        FileNotFoundError: If required files are missing
    """
    if file_path.endswith('.pt'):
        # Direct tensor file
        return torch.load(file_path)
    
    # For text files, need to determine if it's front, left, or right
    # and find the corresponding files
    if not file_path.endswith('.txt'):
        raise ValueError("Input file must be either .pt or .txt format")
    
    # Extract base path and sensor type
    file_dir = os.path.dirname(file_path)
    file_name = os.path.basename(file_path)
    
    # Determine sensor type from filename
    if 'sF' in file_name:
        # This is a front sensor file, need to find matching left and right
        sensor_type = 'F'
    elif 'sL' in file_name:
        sensor_type = 'L'
    elif 'sR' in file_name:
        sensor_type = 'R'
    else:
        raise ValueError(f"Cannot determine sensor type from filename: {file_name}")
    
    # Create paths for all three sensor files
    base_name = file_name.replace(f's{sensor_type}', 's')
    front_path = os.path.join(file_dir, base_name.replace('s', 'sF'))
    left_path = os.path.join(file_dir, base_name.replace('s', 'sL'))
    right_path = os.path.join(file_dir, base_name.replace('s', 'sR'))
    
    # Check if all files exist
    if not (os.path.exists(front_path) and os.path.exists(left_path) and os.path.exists(right_path)):
        raise FileNotFoundError(f"Could not find all three sensor files for {file_name}")
    
    # Load the text files
    tensor_front = torch_genfromtxt(front_path)
    tensor_left = torch_genfromtxt(left_path)
    tensor_right = torch_genfromtxt(right_path)
    
    # Stack into a 3-channel tensor
    combined_tensor = torch.stack([tensor_front, tensor_left, tensor_right], 0)
    
    return combined_tensor


def run_inference(
    model: torch.nn.Module, 
    tensor: torch.Tensor, 
    device: torch.device, 
    class_names: List[str], 
    mean: float = 0.0, 
    std: float = 1.0
) -> Tuple[str, float, Dict[str, float]]:
    """
    Run inference on a single spectrogram tensor.
    
    Args:
        model: Trained PyTorch model
        tensor: Input tensor of shape [3, H, W] representing the spectrogram 
        device: Device to run inference on
        class_names: List of class names
        mean: Mean for normalization
        std: Standard deviation for normalization
        
    Returns:
        prediction: Class name predicted
        confidence: Confidence score (0-1)
        probabilities: Dictionary mapping class names to probabilities
        
    Raises:
        ValueError: If tensor has incorrect shape
    """
    # Ensure the model is in evaluation mode
    model.eval()
    
    # Make sure tensor has the right shape (3 channels for front, left, right)
    if tensor.dim() == 2:
        # If it's a single 2D tensor, we need 3 channels
        logger.warning(
            "Input is a single 2D tensor. Model expects 3 channels (front, left, right). "
            "Duplicating the channel 3 times as a workaround."
        )
        # We'll duplicate the channel 3 times as a workaround
        tensor = tensor.unsqueeze(0).repeat(3, 1, 1)
    elif tensor.dim() != 3 or tensor.shape[0] != 3:
        raise ValueError(f"Expected tensor with shape [3, H, W], got {tensor.shape}")
    
    # Normalize the input tensor
    normalized_tensor = (tensor - mean) / std
    
    # Add batch dimension and move to device
    input_tensor = normalized_tensor.unsqueeze(0).to(device)
    
    # Run inference
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]
        prediction_idx = torch.argmax(probabilities).item()
        confidence = probabilities[prediction_idx].item()
    
    # Get prediction class name
    prediction = class_names[prediction_idx]
    
    # Create dictionary of class probabilities
    prob_dict = {class_name: probabilities[i].item() for i, class_name in enumerate(class_names)}
    
    return prediction, confidence, prob_dict


def main():
    """
    Example usage of the inference functionality as a command-line tool.
    """
    try:
        # Load the model
        model, device, class_names, norm_stats = load_model()
        print(f"Model loaded successfully with {len(class_names)} classes: {class_names}")
        print(f"Normalization stats: mean={norm_stats['mean']:.4f}, std={norm_stats['std']:.4f}")
        
        # Example: Run inference on a sample file
        # This would be replaced with actual file path
        sample_path = input("Enter path to spectrogram file (.pt or .txt): ")
        
        if os.path.exists(sample_path):
            try:
                # Load and process the spectrogram
                tensor = load_spectrogram(sample_path)
                
                # Run inference
                prediction, confidence, probabilities = run_inference(
                    model, tensor, device, class_names, 
                    mean=norm_stats['mean'], std=norm_stats['std']
                )
                
                # Print results
                print(f"\nPrediction: {prediction} with {confidence:.2%} confidence")
                print("\nClass probabilities:")
                
                # Print class probabilities sorted by confidence
                for class_name, prob in sorted(probabilities.items(), key=lambda x: x[1], reverse=True):
                    print(f"  {class_name}: {prob:.2%}")
                    
            except Exception as e:
                logger.error(f"Error processing file: {str(e)}")
        else:
            print(f"Sample file not found: {sample_path}")
            print("Please provide a valid path to a spectrogram file.")
            
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")


if __name__ == "__main__":
    main()