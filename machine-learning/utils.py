import torch 
from torch.utils.data import Dataset, Subset
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import yaml

class NormalizedDataset(Dataset):
    def __init__(self, subset, mean, std):
        self.subset = subset
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        x, y = self.subset[idx]
        return (x - self.mean) / self.std, y

def calculate_stats(subset):
    tensors = torch.cat([subset[i][0] for i in range(len(subset))])
    mean = tensors.mean().item()
    std = tensors.std().item()
    return torch.tensor(mean), torch.tensor(std)

def create_normalised_subsets(raw_dataset, device):
    targets = [raw_dataset[i][1].item() for i in range(len(raw_dataset))]
    print("Actual class order:", raw_dataset.classes)

    # First split: Train+Val vs Test
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_val_indices, test_indices = next(sss.split(np.zeros(len(raw_dataset)), targets))
    
    # Second split: Train vs Val (relative to train_val_indices)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_rel_indices, val_rel_indices = next(sss.split(
        np.zeros(len(train_val_indices)), 
        [targets[i] for i in train_val_indices]
    ))
    
    # Map to original dataset indices
    train_abs_indices = train_val_indices[train_rel_indices]
    val_abs_indices = train_val_indices[val_rel_indices]

    # Create subsets
    train_subset = Subset(raw_dataset, train_abs_indices)
    val_subset = Subset(raw_dataset, val_abs_indices)
    test_subset = Subset(raw_dataset, test_indices)

    labels = [train_subset[i][1].item() for i in range(len(train_subset))] 
    class_counts = torch.bincount(torch.tensor(labels), minlength=5) + 1
    class_weights = (1.0 / class_counts.float())
    class_weights = (class_weights / class_weights.sum()).to(device)

    train_mean, train_std = calculate_stats(train_subset)

    train = NormalizedDataset(train_subset, train_mean, train_std)
    val = NormalizedDataset(val_subset, train_mean, train_std)
    test = NormalizedDataset(test_subset, train_mean, train_std)

    class_to_idx = {name: i for i, name in enumerate(raw_dataset.classes)}
    opposite_pairs = [
        (class_to_idx["Left"], class_to_idx["Right"]),
        (class_to_idx["Down"], class_to_idx["Up"])
    ]
    print("\nClass Verification:")
    print(f"{'Class Name':<15} | {'Index':<5} | {'Opposite Pair'}")
    for name, idx in class_to_idx.items():
        opposites = [pair for pair in opposite_pairs if idx in pair]
        print(f"{name:<15} | {idx:<5} | {opposites}")

    return train, val, test, class_weights, opposite_pairs

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)  # Use safe_load to avoid arbitrary code execution
    return config
