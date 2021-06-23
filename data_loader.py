import numpy as np

import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class CSVDataset(Dataset):
    def __init__(self, data, target):
        self.data = StandardScaler().fit_transform(data).astype(np.float32)
        self.target = target.astype(np.long)

    def __len__(self):
        return len(self.target)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.data[idx])
        y = torch.tensor(self.target[idx], dtype=torch.long)
        return x, y
