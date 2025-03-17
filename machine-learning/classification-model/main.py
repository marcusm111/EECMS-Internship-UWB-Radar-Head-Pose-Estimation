"""
Head Movement Classification Training Pipeline

This script implements the full training pipeline for the head movement classification model.
It handles:
1. Data loading and preprocessing
2. Model selection and configuration
3. Training and validation
4. Testing and evaluation
5. Model saving and visualization
6. Metrics tracking with Weights & Biases

The entire pipeline is controlled through the config.yaml file.
"""

import os
import platform
import numpy as np
from typing import Dict, List, Tuple, Any

import torch
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

# Local imports
from timefrequencydataset import TimeFrequencyMapDataset
from model import load_modified_model, bayesian_optimisation
from utils import create_normalised_subsets, load_config, save_config
from visualisations import visualize_feature_space
from preprocessing import build_data_directory_structure, verify_training_directory

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Windows-specific environment variables for wandb
if os.name == 'nt' or platform.system() == "Windows":  
    os.environ["WANDB_SYMLINK"] = "false"
    os.environ["WANDB_ALWAYS_COPY"] = "true"

# Configure logging
import logging
logger = logging.getLogger(__name__)


def init_wandb_safe(project_name="radar-head-movement", run_name="trial-1"):
    """
    Initialize wandb with error handling and fallback to offline mode.
    
    Args:
        project_name: Name of the wandb project
        run_name: Name of the wandb run
        
    Returns:
        True if wandb initialized successfully, False if using dummy/offline mode
    """
    # First, set Windows-specific environment variables
    if os.name == 'nt' or platform.system() == "Windows":
        os.environ["WANDB_SYMLINK"] = "false"
        os.environ["WANDB_ALWAYS_COPY"] = "true"
    
    # Define global wandb at the beginning to avoid syntax error
    global wandb
    
    try:
        # Try to initialize wandb normally
        wandb.init(
            project=project_name, 
            name=run_name,
            settings=wandb.Settings(start_method="thread")
        )
        logger.info("wandb initialized successfully")
        return True
    except Exception as e:
        logger.warning(f"Could not initialize wandb: {e}")
        logger.info("Falling back to offline mode")
        
        # Try offline mode
        try:
            os.environ["WANDB_MODE"] = "offline"
            wandb.init(
                project=project_name, 
                name=run_name,
                settings=wandb.Settings(start_method="thread")
            )
            logger.info("wandb initialized in offline mode")
            return True
        except Exception as e_offline:
            logger.warning(f"Could not initialize wandb in offline mode: {e_offline}")
            
            # Create a dummy wandb module that does nothing
            class DummyWandb:
                def log(self, *args, **kwargs): pass
                def save(self, *args, **kwargs): pass
                def finish(self, *args, **kwargs): pass
                def Image(self, *args, **kwargs): return None
                def Table(self, *args, **kwargs): return None
                def plot(self, *args, **kwargs): return {"confusion_matrix": lambda **kw: None}
                plot = type('', (), {"confusion_matrix": lambda **kw: None})()
            
            # Replace the global wandb instance
            wandb = DummyWandb()
            logger.info("Using dummy wandb implementation")
            return False

def setup_environment(config: Dict[str, Any]) -> Tuple[torch.device, str, str]:
    """
    Set up the training environment and directory structure.
    
    Args:
        config: Configuration dictionary loaded from YAML
        
    Returns:
        device: PyTorch device (CPU or CUDA)
        data_path: Path to data directory
        tensor_path: Path to processed tensor data
    """
    # Initialize device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set up data paths
    data_path = config["data"]
    os.makedirs(data_path, exist_ok=True)
    
    # Create required directories - streamlined structure without clean_data
    training_dir = os.path.join(data_path, "training_data")
    build_data_directory_structure(data_path)
    
    # Verify data and create tensors directly from raw data if needed
    verify_training_directory(training_dir, config["rebuild_tensors"])
    
    # Tensor data path
    tensor_path = os.path.join(training_dir, "tensor_data")
    
    return device, data_path, tensor_path

def prepare_datasets(tensor_path: str, device: torch.device) -> Tuple[Dataset, Dataset, Dataset, torch.Tensor, List[str]]:
    """
    Load and prepare datasets for training, validation and testing.
    
    Args:
        tensor_path: Path to the tensor data
        device: PyTorch device
        
    Returns:
        train: Training dataset
        val: Validation dataset
        test: Test dataset
        class_weights: Tensor of class weights for handling imbalanced classes
        class_names: List of class names
    """
    # Load raw timefrequencymap dataset
    raw_dataset = TimeFrequencyMapDataset(tensor_path, compute_stats=False)
    
    # Split for training and get class weights
    train, val, test, class_weights = create_normalised_subsets(raw_dataset, device)
    
    # Store class names for later use
    class_names = raw_dataset.classes
    
    return train, val, test, class_weights, class_names


def get_model_and_parameters(
    config: Dict[str, Any], 
    train: Dataset, 
    val: Dataset, 
    device: torch.device, 
    class_weights: torch.Tensor
) -> Tuple[torch.nn.Module, Dict[str, Any], Dict[str, Any]]:
    """
    Configure and get the model based on configuration.
    
    Args:
        config: Configuration dictionary
        train: Training dataset
        val: Validation dataset
        device: PyTorch device
        class_weights: Class weights for balanced training
        
    Returns:
        model: Configured PyTorch model
        model_params: Model architecture parameters
        training_params: Training hyperparameters
    """
    if config["bayesian_optimisation"]:
        # Find and use best parameters using Bayesian optimization
        print("Using Bayesian optimization to find optimal parameters")
        params = bayesian_optimisation(train, val, device, class_weights, config["num_classes"])
        
        model_params = {
            'model_type': params['model_type'],
            'use_pretrained': params['use_pretrained'],
            'hidden_dims': [params[f'layer_{i}_dim'] for i in range(params['num_layers'])],
            'use_dropout': params.get('use_dropout', False),
            'dropout_rate': params.get('dropout_rate', 0.0),
            'use_batchnorm': params.get('use_batchnorm', False),
            'num_unfrozen_layers': params.get('resnet_unfrozen_layers', params.get('vit_unfrozen_layers', 0))
        }
    else:
        # Use default parameters from config
        print("Using default parameter values from config")
        params = config["default_params"]
        model_params = params["model_params"]

    # Load the model
    model = load_modified_model(num_classes=config["num_classes"], **model_params).to(device)
    
    return model, model_params, params


def setup_training(
    model: torch.nn.Module,
    train: Dataset,
    val: Dataset,
    test: Dataset,
    params: Dict[str, Any],
    device: torch.device
) -> Tuple[DataLoader, DataLoader, DataLoader, torch.optim.Optimizer, Dict[str, Any]]:
    """
    Set up training components including dataloaders and optimizer.
    
    Args:
        model: PyTorch model
        train: Training dataset
        val: Validation dataset
        test: Test dataset
        params: Training parameters
        device: PyTorch device
        
    Returns:
        train_loader: DataLoader for training
        val_loader: DataLoader for validation
        test_loader: DataLoader for testing
        optimizer: PyTorch optimizer
        optimizer_config: Optimizer configuration
    """
    # Create full training set (train + validation)
    full_train = ConcatDataset([train, val])
    
    # Set up data loaders
    train_loader = DataLoader(
        full_train, 
        batch_size=params['batch_size'], 
        shuffle=True,
        pin_memory=(device.type == 'cuda'),
        drop_last=True
    )
    
    val_loader = DataLoader(
        val, 
        batch_size=params['batch_size'],
        pin_memory=(device.type == 'cuda'),
        drop_last=True
    )
    
    test_loader = DataLoader(
        test, 
        batch_size=params['batch_size'],
        pin_memory=(device.type == 'cuda'),
        drop_last=True
    )
    
    # Configure optimizer
    optimizer_config = {
        'params': filter(lambda p: p.requires_grad, model.parameters()),
        'lr': params['lr'],
        'weight_decay': params['weight_decay']
    }
    
    # Create optimizer based on configuration
    if params['optimizer'] == 'sgd':
        optimizer_config['momentum'] = params['momentum']
        optimizer = optim.SGD(**optimizer_config)
    else:
        optimizer = optim.Adam(**optimizer_config)
    
    return train_loader, val_loader, test_loader, optimizer, optimizer_config


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    patience: int,
    val_dataset_size: int
) -> float:
    """
    Train the model with early stopping based on validation accuracy.
    
    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        optimizer: PyTorch optimizer
        criterion: Loss function
        device: PyTorch device
        patience: Number of epochs to wait for improvement before stopping
        val_dataset_size: Size of validation dataset for accuracy calculation
        
    Returns:
        best_acc: Best validation accuracy achieved
    """
    best_acc = 0
    no_improve = 0
    
    # Training loop with progress bar
    for epoch in tqdm(range(100), desc="Training"):
        # Training phase
        model.train()
        epoch_loss = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            epoch_loss += loss.item()
        
        # Log training loss safely
        try:
            wandb.log({"final_train_loss": epoch_loss/len(train_loader)})
        except Exception as e:
            logger.warning(f"Could not log to wandb: {e}")

        # Validation phase
        model.eval()
        val_correct = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_correct += (outputs.argmax(1) == labels).sum().item()
        
        # Calculate validation accuracy
        val_acc = val_correct/val_dataset_size

        # Log metrics safely
        try:
            wandb.log({
                "final_train_loss": epoch_loss/len(train_loader),
                "val_acc": val_acc
            })
        except Exception as e:
            logger.warning(f"Could not log to wandb: {e}")
                
        # Early stopping logic
        if val_acc > best_acc + 0.001:  # Improvement threshold
            best_acc = val_acc
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

    return best_acc


def evaluate_model(
    model: torch.nn.Module,
    test_loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    class_names: List[str]
) -> Tuple[float, float, np.ndarray, List[int], List[int]]:
    """
    Evaluate the model on test data and compute metrics.
    
    Args:
        model: Trained PyTorch model
        test_loader: DataLoader for test data
        criterion: Loss function
        device: PyTorch device
        class_names: List of class names
        
    Returns:
        test_acc: Test accuracy
        avg_test_loss: Average test loss
        cm_normalized: Normalized confusion matrix
        all_preds: List of all predictions
        all_labels: List of all true labels
    """
    model.eval()
    all_preds = []
    all_labels = []
    test_loss = 0
    correct = 0
    total = 0
    
    # Evaluation loop
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Accumulate loss
            test_loss += loss.item() * inputs.size(0)
            
            # Get predictions
            _, predicted = torch.max(outputs.data, 1)
            
            # Update statistics
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Collect predictions and labels for confusion matrix
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    avg_test_loss = test_loss / total
    test_acc = correct / total
    
    print(f"\nFinal Test Accuracy: {100 * test_acc:.2f}%")

    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    print("\nConfusion Matrix:")
    print(cm)
    
    # Normalize confusion matrix for better interpretation
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Pretty-print normalized confusion matrix
    print("\nConfusion Matrix (Normalized):")
    print("True \\ Predicted", *class_names, sep="\t")
    for i, (row, class_name) in enumerate(zip(cm_normalized, class_names)):
        print(f"{class_name:15}", *["%.2f" % val for val in row], sep="\t")
    
    return test_acc, avg_test_loss, cm_normalized, all_preds, all_labels


def save_model_and_config(
    model: torch.nn.Module,
    config: Dict[str, Any],
    model_params: Dict[str, Any],
    params: Dict[str, Any],
    test_acc: float,
    avg_test_loss: float,
    all_preds: List[int],
    all_labels: List[int],
    class_names: List[str],
    train_mean: float,
    train_std: float
) -> Tuple[str, str]:
    """
    Save the trained model and configuration.
    """
    # Paths for saving
    final_model_path = "final_head_movement_model.pth"
    final_config_path = "final_config.yaml"
    
    # Save model weights
    torch.save(model.state_dict(), final_model_path)
    
    # Try to save to wandb, but catch errors
    try:
        wandb.save(final_model_path)
    except Exception as e:
        logger.warning(f"Could not save model to wandb: {e}")
        logger.info("Continuing without wandb model saving")

    # Create final configuration with updated parameters
    final_config = {
        "data": config["data"],
        "rebuild_tensors": config["rebuild_tensors"],
        "num_classes": config["num_classes"],
        "train_model": True,  # Since we just trained a model
        "bayesian_optimisation": config["bayesian_optimisation"],
        "default_params": {
            "model_params": model_params,
            "batch_size": params['batch_size'],
            "lr": params['lr'],
            "weight_decay": params['weight_decay'],
            "optimizer": params['optimizer'],
            "patience": params['patience']
        },
        # Add normalization statistics to config
        "normalization_stats": {
            "mean": float(train_mean),
            "std": float(train_std)
        },
        # Add class names to config for consistent inference
        "class_names": class_names
    }
    
    # If using SGD, include momentum parameter
    if params.get('optimizer') == 'sgd' and 'momentum' in params:
        final_config["default_params"]["momentum"] = params['momentum']
        
    # Save final configuration to YAML file
    save_config(final_config, final_config_path)
    print(f"Final configuration saved to {final_config_path}")
    logger.info(f"Class names saved to config: {class_names}")
    
    # Try to save to wandb, but catch errors
    try:
        wandb.save(final_config_path)
    except Exception as e:
        logger.warning(f"Could not save config to wandb: {e}")
        logger.info("Continuing without wandb config saving")

    # Log final test metrics to wandb with error handling
    try:
        wandb.log({
            "final_test_accuracy": test_acc,
            "final_test_loss": avg_test_loss,
            "confusion_matrix": wandb.plot.confusion_matrix(
                preds=all_preds,
                y_true=all_labels,
                class_names=class_names
            ),
            "final_model": wandb.Table(
                columns=["Path"],
                data=[[final_model_path]]
            ),
            "final_config": wandb.Table(
                columns=["Path"],
                data=[[final_config_path]]
            )
        })
    except Exception as e:
        logger.warning(f"Could not log final metrics to wandb: {e}")
        logger.info("Continuing without wandb metrics logging")
    
    return final_model_path, final_config_path


def main():
    """
    Main function to run the full training pipeline.
    """
    # Load configuration
    config = load_config("config.yaml")
    
    # Set up environment
    device, data_path, tensor_path = setup_environment(config)
    
    # Prepare datasets
    train, val, test, class_weights, class_names = prepare_datasets(tensor_path, device)
    
    # Extract normalization stats from training dataset (assuming NormalizedDataset)
    train_mean = train.mean
    train_std = train.std

    # Define loss function with class weights
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
    
    # Initialize wandb for experiment tracking (safely)
    init_wandb_safe(project_name="radar-head-movement", run_name="trial-1")
    
    # Get model and parameters
    model, model_params, params = get_model_and_parameters(
        config, train, val, device, class_weights
    )
    
    # Set up training components
    train_loader, val_loader, test_loader, optimizer, optimizer_config = setup_training(
        model, train, val, test, params, device
    )
    
    # Train the model
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        patience=params['patience'],
        val_dataset_size=len(val)
    )
    
    # Evaluate model on test data
    test_acc, avg_test_loss, cm_normalized, all_preds, all_labels = evaluate_model(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=device,
        class_names=class_names
    )
    
    # Generate feature space visualization
    try:
        visualize_feature_space(
            model=model,
            dataloader=test_loader,
            device=device,
            class_names=class_names 
        )
    except Exception as e:
        logger.warning(f"Could not visualize feature space: {e}")
        logger.info("Continuing without feature space visualization")
    
    # Save model and configuration
    final_model_path, final_config_path = save_model_and_config(
        model=model,
        config=config,
        model_params=model_params,
        params=params,
        test_acc=test_acc,
        avg_test_loss=avg_test_loss,
        all_preds=all_preds,
        all_labels=all_labels,
        class_names=class_names,
        train_mean=train_mean.item(), 
        train_std=train_std.item()   
    )
    
    print(f"Training complete. Model saved to {final_model_path}")
    print(f"Configuration saved to {final_config_path}")


if __name__ == "__main__":
    main()