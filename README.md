# Handwritten-digit-recognition
This repository contains code for recognizing handwritten digits using various machine learning methods. The dataset used is the USPS dataset.

Dataset
The dataset used is the USPS dataset, which is stored in the 
input/usps.h5
 file. The dataset is loaded using the load_usps_dataset function in 
utils/data_loader.py
 (
utils/data_loader.py
).

Methods
The following methods are implemented for handwritten digit recognition:

Simple Mean Comparison
SVD Classification
HOSVD Classification
Random Forest
MLP (Multi-Layer Perceptron)
CNN (Convolutional Neural Network)
RNN (Recurrent Neural Network)
LSTM (Long Short-Term Memory)
GCN (Graph Convolutional Network)
Directory Structure

config.py
 (
config.py
): Configuration file containing dataset paths, image dimensions, training parameters, and random forest parameters.
digit_images/: Directory to store digit images.

input/usps.h5
 (
input/usps.h5
): USPS dataset file.

main.py
 (
main.py
): Main script to run the different methods for handwritten digit recognition.
methods/: Directory containing the implementation of various methods.
cnn_model.py (
methods/cnn_model.py
): Implementation of the CNN model.
gnn_model.py (
methods/gnn_model.py
): Implementation of the GCN model.
hosvd_classification.py (
methods/hosvd_classification.py
): Implementation of the HOSVD classification method.
lstm_method.py (
methods/lstm_method.py
): Implementation of the LSTM method.
mlp_model.py (
methods/mlp_model.py
): Implementation of the MLP model.
random_forest.py (
methods/random_forest.py
): Implementation of the Random Forest method.
rnn_model.py (
methods/rnn_model.py
): Implementation of the RNN model.
simple_mean.py (
methods/simple_mean.py
): Implementation of the Simple Mean Comparison method.
svd_classification.py (
methods/svd_classification.py
): Implementation of the SVD classification method.
utils/: Directory containing utility functions.
data_loader.py (
utils/data_loader.py
): Functions to load and preprocess the dataset.
metrics.py (
utils/metrics.py
): Functions to compute classification metrics.
visualizer.py (
utils/visualizer.py
): Functions to visualize the results.
Usage
To run the different methods for handwritten digit recognition, execute the 
main.py
 script. The script will load the dataset, train the models, and evaluate their performance.

python main.py
Results
The results of the different methods, including accuracy and confusion matrices, will be saved in the digit_images/ directory.

License
This project is licensed under the MIT License. See the LICENSE file for details.
