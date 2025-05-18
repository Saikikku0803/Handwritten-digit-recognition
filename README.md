# Handwritten-digit-recognition

This repository provides code for recognizing handwritten digits using a variety of machine learning methods, applied to the USPS dataset.

## Dataset

The project uses the [USPS dataset](https://github.com/Saikikku0803/Handwritten-digit-recognition/tree/main/input/usps.h5), stored in `input/usps.h5`.  
The dataset is loaded and preprocessed via the `load_usps_dataset` function in [`utils/data_loader.py`](utils/data_loader.py).

## Implemented Methods

The following methods are available for handwritten digit recognition:

- Simple Mean Comparison
- SVD Classification
- HOSVD Classification
- Random Forest
- MLP (Multi-Layer Perceptron)
- CNN (Convolutional Neural Network)
- RNN (Recurrent Neural Network)
- LSTM (Long Short-Term Memory)
- GCN (Graph Convolutional Network)

## Directory Structure

```text
.
├── config.py                # Configuration (dataset paths, image size, training params, etc.)
├── digit_images/            # Output directory for digit images and results
├── input/
│   └── usps.h5              # USPS dataset file
├── main.py                  # Main script to run experiments
├── methods/
│   ├── cnn_model.py         # CNN model implementation
│   ├── gnn_model.py         # GCN model implementation
│   ├── hosvd_classification.py # HOSVD classification
│   ├── lstm_method.py       # LSTM method
│   ├── mlp_model.py         # MLP model
│   ├── random_forest.py     # Random Forest method
│   ├── rnn_model.py         # RNN model
│   ├── simple_mean.py       # Simple Mean Comparison
│   └── svd_classification.py # SVD classification
└── utils/
    ├── data_loader.py       # Dataset loading and preprocessing
    ├── metrics.py           # Classification metrics
    └── visualizer.py        # Result visualization
```
## Usage

To run a handwritten digit recognition experiment with all implemented methods, use:

```bash
python main.py
