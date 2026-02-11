import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

WINDOW_SIZE = 48  # 4 hours @ 5-minute

class SidedDataset(Dataset):
    def __init__(self, path, split="train"):
        df = pd.read_parquet(path)
        df = df.sort_values("Time")

        # Normalize aggregate
        self.mean = df["Aggregate"].mean()
        self.std = df["Aggregate"].std()
        df["Aggregate"] = (df["Aggregate"] - self.mean) / self.std

        targets = ["EVSE", "PV", "CS", "CHP", "BA"]
        data = df[["Aggregate"] + targets].values

        n = len(data)
        train_end = int(n * 0.7)
        val_end = int(n * 0.85)

        if split == "train":
            data = data[:train_end]
        elif split == "val":
            data = data[train_end:val_end]
        else:
            data = data[val_end:]

        self.samples = []
        for i in range(len(data) - WINDOW_SIZE):
            x = data[i : i + WINDOW_SIZE, 0:1]
            y = data[i : i + WINDOW_SIZE, 1:]
            self.samples.append((x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )