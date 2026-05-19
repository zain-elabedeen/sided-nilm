# SIDED NILM

Temporal convolutional network pipeline for non-intrusive load monitoring on the
SIDED dataset. The model learns to disaggregate an aggregate power signal into
five target channels:

- `EVSE`
- `PV`
- `CS`
- `CHP`
- `BA`

The pipeline preprocesses the source data, builds fixed-length time-series
windows, trains a PyTorch TCN model, evaluates it on a held-out test split, and
exports the trained model to ONNX.

## Project Layout

```text
.
├── requirements.txt
├── run_training.py
└── src
    ├── dataset.py
    ├── evaluate.py
    ├── export.py
    ├── model.py
    ├── preprocess.py
    ├── run_training.py
    └── train.py
```

Generated files are written under:

```text
data/processed/sided_processed.parquet
experiments/exp_001/model.pth
experiments/exp_001/config.json
experiments/exp_001/model.onnx
```

## Requirements

- Python 3.10 or newer
- A local SIDED dataset file at `data/sided.parquet`
- PyTorch-compatible CPU or CUDA environment

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data Format

Place the raw SIDED parquet file here:

```text
data/sided.parquet
```

The parquet file is expected to include:

- `Time`
- `Aggregate`
- `EVSE`
- `PV`
- `CS`
- `CHP`
- `BA`

`src/preprocess.py` resamples the data to five-minute intervals using `Time`,
drops missing rows, and writes the processed parquet file to
`data/processed/sided_processed.parquet`.

## Run the Full Pipeline

From the repository root:

```bash
python run_training.py
```

This runs:

1. `src/preprocess.py`
2. `src/train.py`
3. `src/evaluate.py`
4. `src/export.py`

## Run Individual Steps

Preprocess the dataset:

```bash
python src/preprocess.py
```

Train the model:

```bash
python src/train.py
```

Evaluate the best checkpoint:

```bash
python src/evaluate.py
```

Export the trained model to ONNX:

```bash
python src/export.py
```

## Model

The model in `src/model.py` is a temporal convolutional network with:

- Three causal-style convolution blocks
- Dilation rates of `1`, `2`, and `4`
- Batch normalization and ReLU activations
- A final `1x1` convolution that predicts the five appliance channels

Inputs use a window size of `48`, representing four hours of data at five-minute
resolution. The input tensor shape is:

```text
batch_size x 48 x 1
```

The output tensor shape is:

```text
batch_size x 48 x 5
```

## Training Defaults

The current training script uses:

- Train/validation/test split: `70% / 15% / 15%`
- Batch size: `64`
- Epochs: `30`
- Optimizer: Adam
- Learning rate: `1e-3`
- Loss: L1 loss / MAE
- Hidden channels: `32`

The best validation checkpoint is saved to:

```text
experiments/exp_001/model.pth
```

## Evaluation

`src/evaluate.py` loads the best checkpoint and reports mean absolute error on
the test split.

## ONNX Export

`src/export.py` exports the trained PyTorch checkpoint to:

```text
experiments/exp_001/model.onnx
```

The exported graph uses a dynamic batch dimension and ONNX opset `14`.
