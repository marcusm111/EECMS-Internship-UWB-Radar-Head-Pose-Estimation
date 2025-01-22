import os
import numpy as np
import torch
from timefrequencydataset import TimeFrequencyMapDataset

data_path = os.path.join("..", "data")

TestDataset = TimeFrequencyMapDataset(data_path)
