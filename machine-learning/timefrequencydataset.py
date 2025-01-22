from torch.utils.data import Dataset 
import os
import numpy as np

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
                    self.maps.append(os.path.join(map_dir, class_dir))
                    self.labels.append(class_idx)

    def __len__(self):
        return len(self.maps)
    
    def __getitem__(self, idx):
        map_path = self.maps[idx]
        map = np.genfromtxt(map_path, delimiter=",")
        label = self.labels[idx]

        return map, label