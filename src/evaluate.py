import torch
from torch.utils.data import DataLoader
from dataset import SidedDataset
from model import TCN

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
test_ds = SidedDataset("data/processed/sided_processed.parquet", "test")
test_loader = DataLoader(test_ds, batch_size=64)

model = TCN().to(DEVICE)
model.load_state_dict(torch.load("experiments/exp_001/model.pth"))
model.eval()

mae = 0
with torch.no_grad():
    for x,y in test_loader:
        x,y = x.to(DEVICE), y.to(DEVICE)
        preds = model(x)
        mae += torch.mean(torch.abs(preds - y)).item()

print("Test MAE:", mae/len(test_loader))