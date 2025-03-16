"""
TimeFrequencyMap Dataset Module

This module defines a PyTorch Dataset for time-frequency map data used in head movement 
classification. It handles loading time-frequency maps from files, computing dataset 
statistics, and providing normalized data for model training.

The dataset expects a directory structure where each subdirectory represents a class
and contains tensor files (.pt) with time-frequency maps.
"""

import os
import logging
from typing import Optional, Tuple, List

import torch
from torch.utils.data import Dataset

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TimeFrequencyMapDataset(Dataset):
    """
    Dataset for time-frequency maps used in head movement classification.
    
    This dataset loads pre-processed tensor files (.pt) containing time-frequency
    maps from different sensors (front, left, right) for head movement data.
    
    Attributes:
        map_dataset_dir (str): Directory containing the dataset
        classes (List[str]): List of class names (subdirectory names)
        class_to_idx (dict): Mapping from class names to indices
        data (List[torch.Tensor]): List of loaded tensors
        labels (List[int]): List of corresponding labels
        mean (Optional[torch.Tensor]): Mean value for normalization
        std (Optional[torch.Tensor]): Standard deviation for normalization
    """
    
    def __init__(self, map_dataset_dir: str, compute_stats: bool = True):
        """
        Initialize the dataset by loading all tensor files from the specified directory.
        
        Args:
            map_dataset_dir: Path to dataset directory
            compute_stats: Whether to compute mean and std for normalization
        """
        self.map_dataset_dir = map_dataset_dir
        
        # Get class names from subdirectories
        try:
            self.classes = sorted(os.listdir(map_dataset_dir))
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        except FileNotFoundError:
            logger.error(f"Dataset directory not found: {map_dataset_dir}")
            raise
        
        if not self.classes:
            logger.warning(f"No class directories found in {map_dataset_dir}")
        
        self.data = []   # List to hold all preloaded tensors
        self.labels = [] # List to hold corresponding labels
        
        # Load all data into memory
        for class_name in self.classes:
            class_dir = os.path.join(map_dataset_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            
            if not os.path.isdir(class_dir):
                continue
                
            # Load each tensor file in this class directory
            loaded_files = 0
            for map_file in os.listdir(class_dir):
                if not map_file.endswith('.pt'):
                    continue
                    
                file_path = os.path.join(class_dir, map_file)
                try:
                    tensor = torch.load(file_path)
                    self.data.append(tensor)
                    self.labels.append(class_idx)
                    loaded_files += 1
                except Exception as e:
                    logger.warning(f"Error loading {file_path}: {e}")
            
            logger.info(f"Loaded {loaded_files} tensor files for class '{class_name}'")
        
        logger.info(f"Total dataset size: {len(self.data)} samples across {len(self.classes)} classes")
        
        # Compute dataset statistics if requested
        if compute_stats:
            self.mean, self.std = self._compute_stats()
            logger.info(f"Dataset statistics - Mean: {self.mean.item():.4f}, Std: {self.std.item():.4f}")
        else:
            self.mean = None
            self.std = None

    def _compute_stats(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute mean and standard deviation across all tensors in the dataset.
        
        This is used for normalizing the data during training.
        
        Returns:
            mean: Mean value of all elements in the dataset
            std: Standard deviation of all elements in the dataset
        """
        if not self.data:
            logger.warning("Cannot compute statistics on empty dataset")
            return torch.tensor(0.0), torch.tensor(1.0)
            
        sum_total = 0.0
        sum_sq_total = 0.0
        count_total = 0
        
        for tensor in self.data:
            # Calculate sum and sum of squares for mean and variance
            sum_total += tensor.sum().item()
            sum_sq_total += (tensor ** 2).sum().item()
            count_total += tensor.numel()
        
        # Calculate mean and standard deviation
        mean = sum_total / count_total
        var = (sum_sq_total / count_total) - (mean ** 2)
        
        # Use a minimum std value to avoid division by zero
        std = max(var ** 0.5, 1e-7)
        
        return torch.tensor(mean), torch.tensor(std)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset at the specified index.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            tensor: The time-frequency map tensor, normalized if stats are available
            label: The class label as a tensor
        """
        tensor = self.data[idx]
        
        # Apply normalization if stats are available
        if self.mean is not None and self.std is not None:
            tensor = (tensor - self.mean) / self.std
            
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return tensor, label