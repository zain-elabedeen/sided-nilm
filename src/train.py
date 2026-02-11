import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import SidedDataset
from model import TCN
import json
import os

# Hyperparameters

# Default:
# 	•	Window size: 48
# 	•	Batch size: 64
# 	•	Learning rate: 1e-3
# 	•	Epochs: 30
# 	•	Hidden dimension: 32

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs("experiments/exp_001", exist_ok=True)

train_ds = SidedDataset("data/processed/sided_processed.parquet", "train")
val_ds = SidedDataset("data/processed/sided_processed.parquet", "val")

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64)

model = TCN().to(DEVICE)
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

best_val = float("inf")
for epoch in range(30):
    model.train()
    train_loss = 0

    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        preds = model(x)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            val_loss += criterion(model(x), y).item()

    val_loss /= len(val_loader)

    print(f"Epoch {epoch} | Train {train_loss/len(train_loader):.4f} | Val {val_loss:.4f}")

    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), "experiments/exp_001/model.pth")

config = {"epochs":30, "best_val_loss":best_val}
with open("experiments/exp_001/config.json", "w") as f:
    json.dump(config, f)