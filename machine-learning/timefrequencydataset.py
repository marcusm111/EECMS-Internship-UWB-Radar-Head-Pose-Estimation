from torch.utils.data import Dataset 
import os
import numpy as np
import torch.nn.functional as F
import torch

def pad_or_truncate(matrix, target_cols=2000):
   if matrix.shape[1] > target_cols:
       return matrix[:, :target_cols]
   else:
       return np.pad(
           matrix,
           ((0, 0), (0, target_cols - matrix.shape[1])),
           mode='constant',
           constant_values=1
       )
   
def resize_for_vgg(x):
    # Maintain aspect ratio by padding height
    target_ratio = 224/224
    current_ratio = x.shape[1]/x.shape[0]
    
    if current_ratio > target_ratio:
        # Pad height first
        pad_height = int(x.shape[1]/target_ratio - x.shape[0])
        x = F.pad(x, (0, 0, 0, pad_height))
    
    # Resize to 224x224
    x = F.interpolate(x.unsqueeze(0).unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)
    return x.squeeze()

class TimeFrequencyMapDataset(Dataset):
    def __init__(self, map_dataset_dir):
        self.map_dataset_dir = map_dataset_dir 

        self.classes = sorted(os.listdir(map_dataset_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        self.maps = []
        self.labels = []

        for class_name in self.classes:
            class_dir = os.path.join(map_dataset_dir, class_name)
            class_idx = self.class_to_idx[class_name]

            for map_dir in os.listdir(class_dir):
                if map_dir.endswith("txt"):
                    self.maps.append(os.path.join(class_dir, map_dir))
                    self.labels.append(class_idx)

    def __len__(self):
        return len(self.maps)
    
    def __getitem__(self, idx):
        map_path = self.maps[idx]
        map = np.genfromtxt(map_path, delimiter=",")
        label = self.labels[idx]

        # Ensure map size is consistent
        map = pad_or_truncate(map)

        map = torch.from_numpy(map)

        map = resize_for_vgg(map)

        return map, label