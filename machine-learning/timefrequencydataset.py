from torch.utils.data import Dataset
import os
import numpy as np
import torch
import torch.nn.functional as F

def pad_or_truncate(matrix, target_cols=2000):
    if matrix.shape[1] > target_cols:
        return matrix[:, :target_cols]
    else:
        return np.pad(
            matrix,
            ((0, 0), (0, target_cols - matrix.shape[1])),
            mode='constant',
            constant_values=0
        )

class TimeFrequencyMapDataset(Dataset):
    def __init__(self, map_dataset_dir, device):
        self.map_dataset_dir = map_dataset_dir
        self.classes = sorted(os.listdir(map_dataset_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.samples = self._load_samples()
        self.device = device

    def _load_samples(self):
        samples = []
        for class_name in self.classes:
            class_dir = os.path.join(self.map_dataset_dir, class_name)
            for file in os.listdir(class_dir):
                if file.endswith(".txt"):
                    file_path = os.path.join(class_dir, file)
                    samples.append((file_path, self.class_to_idx[class_name]))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        map_path, label = self.samples[idx]
        
        # Load and preprocess
        matrix = np.genfromtxt(map_path, delimiter=",")
        matrix = pad_or_truncate(matrix)
        
        # Convert to tensor and add channel dimension
        tensor = torch.from_numpy(matrix).float().unsqueeze(0)  # (1, H, W)
        
        # Resize with aspect ratio preservation
        tensor = self._adaptive_resize(tensor)
        return tensor, label

    def _adaptive_resize(self, x):
        # Input: (1, H, W) tensor
        _, orig_h, orig_w = x.shape
        target_h = 224
        target_w = 224

        # Calculate scaling factors
        scale_h = target_h / orig_h
        scale_w = target_w / orig_w
        scale = min(scale_h, scale_w)
        
        # Resize first
        new_h = int(orig_h * scale)
        new_w = int(orig_w * scale)
        x = F.interpolate(x.unsqueeze(0), 
                        size=(new_h, new_w), 
                        mode='bilinear', 
                        align_corners=False).squeeze(0)
        
        # Pad to target size
        pad_h = max(target_h - new_h, 0)
        pad_w = max(target_w - new_w, 0)
        x = F.pad(x, (0, pad_w, 0, pad_h), 
                mode='constant', value=0)
        
        return x.to(self.device)  # Shape (1, 224, 224)