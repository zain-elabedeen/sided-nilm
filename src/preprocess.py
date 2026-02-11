import pandas as pd
import os

# Load the Parquet version of SIDED
df = pd.read_parquet("data/sided.parquet")

# Downsample to 5-min if needed (optional)
df = df.resample("5T", on="Time").mean().dropna()

# Save processed
os.makedirs("data/processed", exist_ok=True)
df.to_parquet("data/processed/sided_processed.parquet")

print("Preprocessing complete!")