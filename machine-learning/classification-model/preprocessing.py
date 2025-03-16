"""
Preprocessing Module for Head Movement Classification

This module handles data preprocessing tasks for the head movement classification project:
1. Creating directory structures for organizing data
2. Converting raw text files directly to PyTorch tensors
3. Cleaning data by handling NaN, infinity, and invalid values while processing
4. Streamlined pipeline from raw sensor data to tensor format

The module expects raw data to be organized by movement class, with each sample
having corresponding front (sF), left (sL), and right (sR) sensor measurements.
"""

import os
import math
import logging
from typing import Dict, List, Optional, Tuple, Union

import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def torch_genfromtxt(file_path: str, delimiter: str = ",", dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    Load a matrix from a text file into a PyTorch tensor.
    Similar to numpy's genfromtxt but returns a PyTorch tensor.
    Also handles cleaning (NaN, infinity, invalid values) during loading.
    
    Args:
        file_path: Path to the text file
        delimiter: Character used to separate values in the file
        dtype: PyTorch data type for the tensor
        
    Returns:
        PyTorch tensor containing the cleaned data from the file
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file contains invalid data
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    data = []
    inf_count = 0
    nan_count = 0
    invalid_count = 0
    
    try:
        with open(file_path, "r") as f:
            for line_num, line in enumerate(f, 1):
                # Skip empty lines
                if not line.strip():
                    continue
                    
                # Split line into elements and clean/convert to floats
                elements = line.strip().split(delimiter)
                cleaned_row = []
                
                for elem in elements:
                    elem = elem.strip()
                    if not elem:
                        cleaned_row.append(0.0)  # Replace empty with 0
                        invalid_count += 1
                        continue
                        
                    try:
                        val = float(elem)
                        if math.isinf(val):
                            cleaned_row.append(0.0)
                            inf_count += 1
                        elif math.isnan(val):
                            cleaned_row.append(0.0)
                            nan_count += 1
                        else:
                            cleaned_row.append(val)  # Keep valid numbers
                    except ValueError:
                        cleaned_row.append(0.0)
                        invalid_count += 1
                
                if cleaned_row:  # Only append if we have data
                    data.append(cleaned_row)
                    
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {e}")
        raise
        
    # Check if we have data to convert
    if not data:
        logger.warning(f"No valid data found in {file_path}, returning empty tensor")
        return torch.zeros((0, 0), dtype=dtype)
    
    # Log cleaning summary if any issues found
    if inf_count + nan_count + invalid_count > 0:
        logger.info(f"Cleaned while loading {file_path}:")
        if inf_count > 0:
            logger.info(f"  → Infinities replaced: {inf_count}")
        if nan_count > 0:
            logger.info(f"  → NaNs replaced: {nan_count}")
        if invalid_count > 0:
            logger.info(f"  → Invalid entries replaced: {invalid_count}")
    
    # Convert to tensor and handle ragged rows (if necessary)
    try:
        tensor = torch.tensor(data, dtype=dtype)
        return tensor
    except Exception as e:
        logger.error(f"Error creating tensor from {file_path}: {e}")
        # If rows have different lengths, try to handle by padding
        max_len = max(len(row) for row in data)
        padded_data = [row + [0.0] * (max_len - len(row)) for row in data]
        logger.warning(f"Padded irregular data with zeros in {file_path}")
        return torch.tensor(padded_data, dtype=dtype)


def build_data_directory_structure(data_path: str) -> None:
    """
    Build the streamlined directory structure for data organization.
    Eliminates the clean_data intermediate directory.
    
    Args:
        data_path: Root path for data
    """
    os.makedirs(os.path.join(data_path, "training_data"), exist_ok=True)
    os.makedirs(os.path.join(data_path, "test_data"), exist_ok=True)
    os.makedirs(os.path.join(data_path, "training_data", "raw_data"), exist_ok=True)
    os.makedirs(os.path.join(data_path, "training_data", "tensor_data"), exist_ok=True)
    
    logger.info(f"Created streamlined directory structure in {data_path}")


def build_dataset_structure(dir_path: str) -> None:
    """
    Create class subdirectories for organizing samples by movement type.
    
    Args:
        dir_path: Directory where class subdirectories will be created
    """
    # Define all possible movement classes
    movement_classes = [
        "Down", "No_Movement", "Right30", "Right60", "Right90",
        "Left90", "Left30", "Left60", "Up"
    ]
    
    # Make class directories
    for class_name in movement_classes:
        os.makedirs(os.path.join(dir_path, class_name), exist_ok=True)
        
    logger.info(f"Created class directories in {dir_path}")


def process_raw_to_tensor(raw_dir_path: str, tensor_dir_path: str) -> None:
    """
    Process raw data files directly to tensors, skipping the intermediate clean_data step.
    This function combines cleaning and tensor creation in one step for memory efficiency.
    
    Args:
        raw_dir_path: Path to raw data directory
        tensor_dir_path: Path to save processed tensor files
    """
    # Create tensor directory if it doesn't exist
    os.makedirs(tensor_dir_path, exist_ok=True)
    
    # Create class directories in tensor path
    build_dataset_structure(tensor_dir_path)
    
    # Count variables for logging
    total_samples = 0
    successful_samples = 0
    failed_samples = 0
    
    # Process each class directory
    for class_dir in os.listdir(raw_dir_path):
        class_full_dir = os.path.join(raw_dir_path, class_dir)
        
        if not os.path.isdir(class_full_dir):
            continue
            
        class_samples = 0
        class_successful = 0
        
        # Create destination class directory
        os.makedirs(os.path.join(tensor_dir_path, class_dir), exist_ok=True)
        
        # Get all files in this class
        measurements_list = os.listdir(class_full_dir)
        
        # Process front sensor files and find matching left/right files
        for measurement in measurements_list:
            # Only process front sensor files to avoid duplicates
            if measurement.endswith(".txt") and "sF" in measurement:
                total_samples += 1
                class_samples += 1
                
                # Create filenames for corresponding left and right measurements
                measurement_list = list(measurement)
                
                # Create left name
                measurement_list[1] = "L"
                left_corresponding_measurement = "".join(measurement_list)
                
                # Create right name
                measurement_list[1] = "R"
                right_corresponding_measurement = "".join(measurement_list)
                
                # Create paths
                measurement_path = os.path.join(class_full_dir, measurement)
                left_measurement_path = os.path.join(class_full_dir, left_corresponding_measurement)
                right_measurement_path = os.path.join(class_full_dir, right_corresponding_measurement)
                
                # Check if all three files exist
                if os.path.exists(left_measurement_path) and os.path.exists(right_measurement_path): 
                    try:
                        # Load and clean tensor data from each file in one step
                        tensor_front = torch_genfromtxt(measurement_path)
                        tensor_left = torch_genfromtxt(left_measurement_path)
                        tensor_right = torch_genfromtxt(right_measurement_path)
                        
                        # Stack into a 3-channel tensor
                        combined_tensor = torch.stack([tensor_front, tensor_left, tensor_right], 0)
                        
                        # Extract sample tag from filename
                        tag = measurement[2:-4]
                        
                        # Save combined tensor
                        current_tensor_path = os.path.join(tensor_dir_path, class_dir, tag + ".pt")
                        
                        # Check for invalid tensor shape
                        if combined_tensor.dim() != 3 or combined_tensor.shape[0] != 3:
                            logger.warning(f"Invalid tensor shape {combined_tensor.shape} for {measurement}")
                            failed_samples += 1
                            continue
                            
                        torch.save(combined_tensor, current_tensor_path)
                        successful_samples += 1
                        class_successful += 1
                        
                    except Exception as e:
                        logger.error(f"Error processing {measurement}: {e}")
                        failed_samples += 1
                else:
                    logger.warning(f"Missing corresponding files for {measurement}")
                    failed_samples += 1
        
        logger.info(f"Class {class_dir}: Processed {class_successful}/{class_samples} samples successfully")
    
    logger.info(f"Total: Successfully created {successful_samples}/{total_samples} tensor files")
    logger.info(f"Tensor files saved to {tensor_dir_path}")


def verify_training_directory(data_path: str, rebuild_tensors: bool = False) -> bool:
    """
    Verify that the training directory has the required structure and data.
    Directly processes raw data to tensors, skipping the clean_data step.
    
    Args:
        data_path: Path to the training data directory
        rebuild_tensors: Whether to rebuild tensors even if they exist
        
    Returns:
        True if verification successful, False otherwise
    """
    # Verify paths
    raw_dir_path = os.path.join(data_path, "raw_data")
    tensor_path = os.path.join(data_path, "tensor_data")
    
    # Check if raw directory exists
    raw_dir_exists = os.path.exists(raw_dir_path)
    raw_dir = os.listdir(raw_dir_path) if raw_dir_exists else []
    
    # Check if tensor directory exists and has data
    tensor_dir_exists = os.path.exists(tensor_path)
    tensor_dir_has_data = False
    
    if tensor_dir_exists:
        for class_dir in os.listdir(tensor_path):
            class_path = os.path.join(tensor_path, class_dir)
            if os.path.isdir(class_path) and len(os.listdir(class_path)) > 0:
                tensor_dir_has_data = True
                break
    
    # If raw directory doesn't exist or is empty, we can't proceed
    if not raw_dir_exists or len(raw_dir) == 0:
        logger.error("Raw data directory not found or empty. Please add raw data.")
        return False
    
    # Build tensor datasets if needed
    if rebuild_tensors or not tensor_dir_exists or not tensor_dir_has_data:
        logger.info("Creating tensors directly from raw data...")
        process_raw_to_tensor(raw_dir_path, tensor_path)
    else:
        logger.info("Tensor data already exists. Skipping processing.")
        # Optional validation of tensor files could be added here
    
    return True


def check_raw_data_quality(directory: str, delimiter: str = ",") -> List[Dict[str, Union[str, bool]]]:
    """
    Check all .txt files in a directory for NaN, infinity, or non-numeric values.
    Useful for data quality assessment before processing.
    
    Args:
        directory: Path to directory containing .txt files
        delimiter: Column delimiter used in the files
        
    Returns:
        List of dictionaries with information about problematic files
    """
    problematic_files = []
    files_checked = 0

    for root, dirs, files in os.walk(directory):
        for file in files:
            files_checked += 1
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                has_nan = False
                has_inf = False
                has_invalid = False

                with open(file_path, 'r') as f:
                    for line_num, line in enumerate(f, 1):
                        elements = line.strip().split(delimiter)
                        
                        for col_num, element in enumerate(elements, 1):
                            element = element.strip()
                            if not element:
                                continue  # Skip empty elements
                            
                            try:
                                val = float(element)
                            except ValueError:
                                logger.warning(f"⚠️ Invalid value '{element}' in {file_path}")
                                logger.warning(f"  → Line {line_num}, Column {col_num}")
                                has_invalid = True
                                continue
                            
                            if math.isnan(val):
                                logger.warning(f"🚫 NaN detected in {file_path}")
                                logger.warning(f"  → Line {line_num}, Column {col_num}")
                                has_nan = True
                            elif math.isinf(val):
                                logger.warning(f"🚫 Infinity detected in {file_path}")
                                logger.warning(f"  → Line {line_num}, Column {col_num}")
                                has_inf = True

                # Record files with any issues
                if has_nan or has_inf or has_invalid:
                    problematic_files.append({
                        'path': file_path,
                        'has_nan': has_nan,
                        'has_inf': has_inf,
                        'has_invalid': has_invalid
                    })

    # Print summary
    logger.info("\n=== Data Quality Check Summary ===")
    if not problematic_files:
        logger.info(f"✅ All {files_checked} files are clean - no NaNs, infs, or invalid values found")
    else:
        logger.warning(f"❌ Found issues in {len(problematic_files)} files:")
        for file_info in problematic_files:
            issues = []
            if file_info['has_nan']: issues.append("NaNs")
            if file_info['has_inf']: issues.append("Infs")
            if file_info['has_invalid']: issues.append("Invalid values")
            logger.warning(f"  - {file_info['path']}: {', '.join(issues)}")

    return problematic_files