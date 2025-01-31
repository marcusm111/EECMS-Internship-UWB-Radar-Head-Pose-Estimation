from torch.utils.data import Dataset
import os
import torch

def torch_genfromtxt(file_path, delimiter=",", dtype=torch.float32):
    """
    Load a matrix from a text file into a PyTorch tensor.
    Mimics basic functionality of `np.genfromtxt`.
    """
    data = []
    with open(file_path, "r") as f:
        for line in f:
            # Split line into elements and convert to floats
            elements = line.strip().split(delimiter)
            row = [float(e) for e in elements if e]  # Skip empty strings
            data.append(row)
    # Convert to tensor and handle ragged rows (if necessary)
    tensor = torch.tensor(data, dtype=dtype)
    return tensor

class TimeFrequencyMapDataset(Dataset):
    def __init__(self, map_dataset_dir, compute_stats=True):
        self.map_dataset_dir = map_dataset_dir
        self.classes = sorted(os.listdir(map_dataset_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.samples = self._load_samples()
        
        # Only compute stats if requested
        if compute_stats:
            self.mean, self.std = self._compute_stats()
        else:
            self.mean = None
            self.std = None

    def _load_samples(self):
        samples = []
        for class_name in self.classes:
            class_dir = os.path.join(self.map_dataset_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            for map_file in os.listdir(class_dir):
                file_path = os.path.join(class_dir, map_file)
                tensor = torch_genfromtxt(file_path, delimiter=",")
                tensor = tensor.unsqueeze(0)  # Add channel dimension
                samples.append((tensor, class_idx))
        return samples

    def _compute_stats(self):
        sum_total = 0.0
        sum_sq_total = 0.0
        count_total = 0
        for x, _ in self.samples:
            sum_total += x.sum().item()
            sum_sq_total += (x ** 2).sum().item()
            count_total += x.numel()
        mean = sum_total / count_total
        var = (sum_sq_total / count_total) - (mean ** 2)
        std = max(var ** 0.5, 1e-7)
        return torch.tensor(mean), torch.tensor(std)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tensor, label = self.samples[idx]
        # Only normalize if stats were computed
        if self.mean is not None and self.std is not None:
            tensor = (tensor - self.mean) / self.std
        return tensor, torch.tensor(label, dtype=torch.long)