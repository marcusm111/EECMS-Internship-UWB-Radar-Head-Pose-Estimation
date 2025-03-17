"""
Utility Functions for Head Movement Classification

This module provides utility functions for the head movement classification project:
1. Dataset normalization and splitting
2. Configuration file handling
3. Directory validation

These utilities support the main training pipeline and ensure consistent
data handling throughout the project.
"""

import os
import logging
from typing import Dict, List, Tuple, Any, Optional

import torch
import numpy as np
import yaml
from torch.utils.data import Dataset, Subset
from sklearn.model_selection import StratifiedShuffleSplit

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def is_directory_empty(directory: str) -> bool:
    """
    Check if a directory is empty or contains only subdirectories.
    
    Args:
        directory: Path to the directory to check
        
    Returns:
        True if the directory doesn't exist or has no files, False otherwise
    """
    # Check if the directory exists
    if not os.path.exists(directory):
        return True
    
    # Check if the directory contains any files
    with os.scandir(directory) as it:
        for entry in it:
            if entry.is_file():
                return False  # Directory contains at least one file
    
    return True  # Directory is empty or contains only subdirectories


class NormalizedDataset(Dataset):
    """
    Dataset wrapper that applies normalization to another dataset.
    
    This wraps a PyTorch Dataset (typically a Subset) and applies
    normalization to the tensor data before returning it.
    
    Attributes:
        subset: The underlying dataset to normalize
        mean: Mean value for normalization
        std: Standard deviation for normalization
    """
    
    def __init__(self, subset: Dataset, mean: torch.Tensor, std: torch.Tensor):
        """
        Initialize the normalized dataset.
        
        Args:
            subset: The dataset to normalize
            mean: Mean value for normalization
            std: Standard deviation for normalization
        """
        self.subset = subset
        self.mean = mean
        self.std = std

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.subset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a normalized sample from the dataset.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            Normalized tensor and corresponding label
        """
        x, y = self.subset[idx]
        return (x - self.mean) / self.std, y


def calculate_stats(subset: Subset) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate mean and standard deviation for a dataset subset.
    
    Args:
        subset: Dataset subset to compute statistics for
        
    Returns:
        mean: Mean value of all elements in the subset
        std: Standard deviation of all elements in the subset
    """
    if len(subset) == 0:
        logger.warning("Cannot calculate statistics on empty subset")
        return torch.tensor(0.0), torch.tensor(1.0)
    
    # Concatenate all tensors in the subset
    tensors = torch.cat([subset[i][0] for i in range(len(subset))])
    
    # Calculate statistics
    mean = tensors.mean().item()
    std = tensors.std().item()
    
    # Use a minimum std value to avoid division by zero
    std = max(std, 1e-7)
    
    return torch.tensor(mean), torch.tensor(std)


def create_normalised_subsets(
    raw_dataset: Dataset, 
    device: torch.device
) -> Tuple[NormalizedDataset, NormalizedDataset, NormalizedDataset, torch.Tensor]:
    """
    Create normalized train, validation, and test subsets with stratified sampling.
    
    Args:
        raw_dataset: Complete dataset to split
        device: PyTorch device for tensors
        
    Returns:
        train: Normalized training dataset
        val: Normalized validation dataset
        test: Normalized test dataset
        class_weights: Tensor of class weights for handling class imbalance
    """
    # Get all labels for stratified splitting
    all_labels = [raw_dataset[i][1] for i in range(len(raw_dataset))]
    
    # Create stratified split indices
    # 70% training, 15% validation, 15% test
    train_val_split = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    train_idx, temp_idx = next(train_val_split.split(np.zeros(len(all_labels)), all_labels))
    
    # Split the remaining 30% into validation and test
    val_test_split = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    val_labels = [all_labels[i] for i in temp_idx]
    val_idx_in_temp, test_idx_in_temp = next(val_test_split.split(np.zeros(len(temp_idx)), val_labels))
    
    # Convert val_idx and test_idx to be relative to original dataset
    val_idx = [temp_idx[i] for i in val_idx_in_temp]
    test_idx = [temp_idx[i] for i in test_idx_in_temp]
    
    # Create subsets
    train_subset = Subset(raw_dataset, train_idx)
    val_subset = Subset(raw_dataset, val_idx)
    test_subset = Subset(raw_dataset, test_idx)
    
    # Compute class weights for handling imbalanced classes
    class_counts = {}
    for label in all_labels:
        label_item = label.item() if isinstance(label, torch.Tensor) else label
        if label_item not in class_counts:
            class_counts[label_item] = 0
        class_counts[label_item] += 1
    
    # Compute inverse frequency class weights
    num_samples = len(all_labels)
    num_classes = len(class_counts)
    weights = [num_samples / (num_classes * class_counts[c]) for c in sorted(class_counts.keys())]
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
    
    # Calculate normalization statistics from training data only
    train_mean, train_std = calculate_stats(train_subset)
    logger.info(f"Normalization stats - Mean: {train_mean.item():.4f}, Std: {train_std.item():.4f}")

    # Create normalized datasets
    train = NormalizedDataset(train_subset, train_mean, train_std)
    val = NormalizedDataset(val_subset, train_mean, train_std)
    test = NormalizedDataset(test_subset, train_mean, train_std)
    
    # Log class distribution
    class_to_idx = {name: i for i, name in enumerate(raw_dataset.classes)}
    
    logger.info("\nClass Distribution:")
    logger.info(f"{'Class Name':<15} | {'Index':<5} | {'Train':<6} | {'Val':<6} | {'Test':<6}")
    
    for name, idx in class_to_idx.items():
        train_count = sum(1 for i in range(len(train_subset)) if train_subset[i][1].item() == idx)
        val_count = sum(1 for i in range(len(val_subset)) if val_subset[i][1].item() == idx)
        test_count = sum(1 for i in range(len(test_subset)) if test_subset[i][1].item() == idx)
        
        logger.info(f"{name:<15} | {idx:<5} | {train_count:<6} | {val_count:<6} | {test_count:<6}")

    return train, val, test, class_weights


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If the configuration file doesn't exist
        yaml.YAMLError: If the YAML is malformed
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)  # Use safe_load to avoid arbitrary code execution
        
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration: {e}")
        raise


def save_config(config_data: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to a YAML file.
    
    Args:
        config_data: Configuration dictionary to save
        config_path: Path where to save the YAML file
    """
    try:
        with open(config_path, "w") as f:
            yaml.dump(config_data, f, default_flow_style=False)
        
        logger.info(f"Configuration saved to {config_path}")
    except Exception as e:
        logger.error(f"Error saving configuration to {config_path}: {e}")