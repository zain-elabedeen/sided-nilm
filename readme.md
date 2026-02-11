# Installation
`git clone https://github.com/YOUR_USERNAME/sided-nilm.git`

`cd sided-nilm`

`pip install -r requirements.txt`

Place the SIDED dataset inside:
`data/sided.parquet`

# Running the Full Pipeline
`python run_training.py`

# SIDED model training pipeline

### src/preprocess.py

Prepares the raw SIDED dataset for training:
1.	Loads sided.parquet
2.	Resamples to 5-minute intervals (if needed)
3.	Cleans missing values
4.	Saves processed version to: `data/processed/sided_processed.parquet`

### src/dataset.py

Transforms time-series into windowed training samples:
1.	Loads processed dataset
2.	Sorts by time
3.	Normalizes Aggregate column
4.	Splits data temporally:	70% train, 15% validation, 15% test
5.	Creates sliding windows

### src/model.py

Defines the TCN architecture:
- 3 Temporal Convolution Blocks
-  Increasing dilation (1, 2, 4)
-  Batch normalization
- ReLU activation
- 1x1 convolution output layer

src/train.py

Trains the TCN model:
1.	Loads training + validation datasets
2.	Creates DataLoaders
3.	Initializes:
-	Adam optimizer
-	L1 (MAE) loss
4.	Training loop:
-	Forward pass
-	Compute loss
-	Backpropagation
-	Validation evaluation

5.	Saves best model to: `experiments/exp_001/model.pth`

### src/evaluate.py

Evaluates final model on test dataset:
- Loads best model
- Runs inference on test split
- Computes MAE

### src/export.py

Exports trained model to ONNX.

Exported file: `experiments/exp_001/model.onnx`
