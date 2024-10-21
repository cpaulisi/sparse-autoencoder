# Sparse Autoencoder
> A repository for experimenting with use cases related to sparse autoencoders. 

## How It Works

This repository implements a basic sparse autoencoder using PyTorch. Here's a brief overview of how it works:

1. **Architecture**: The sparse autoencoder consists of an encoder and a decoder. The encoder compresses the input data into a lower-dimensional representation, while the decoder attempts to reconstruct the original input from this compressed representation.

2. **Sparsity**: The key feature of a sparse autoencoder is that it encourages sparsity in the hidden layer activations. This means that for any given input, only a small number of hidden units should be active.

3. **Loss Function**: The model uses two components in its loss function:
   - Reconstruction Loss: Measures how well the decoder can reconstruct the input (typically using Mean Squared Error).
   - Sparsity Loss: Penalizes the model if the average activation of hidden units deviates from a small target value.

4. **Training**: The model is trained using backpropagation and optimized using Adam optimizer. During training, it learns to reconstruct the input data while maintaining sparse activations in the hidden layer.

5. **Usage**: After training, the encoder can be used to generate sparse representations of input data, which can be useful for feature extraction, dimensionality reduction, or as a preprocessing step for other machine learning tasks.

To get started, check out the `src/sparse_autoencoder/model.py` file for the implementation details and example usage.


## Setup Guide

To set up and run this project, follow these steps:

1. **Prerequisites**:
   - Ensure you have Python 3.11 or later installed on your system.
   - Install [Poetry](https://python-poetry.org/docs/#installation) for dependency management.

2. **Clone the repository**:
   ```
   git clone https://github.com/your-username/sparse-autoencoder.git
   cd sparse-autoencoder
   ```

3. **Install dependencies**:
   Run the following command to install all required dependencies:
   ```
   poetry install
   ```

4. **Activate the virtual environment**:
   ```
   poetry shell
   ```

5. **Run the code**:
   You can now run the sparse autoencoder code. For example:
   ```
   python src/sparse_autoencoder/model.py
   ```

6. **Customization**:
   - Modify the hyperparameters in `src/sparse_autoencoder/model.py` to experiment with different configurations.
   - Adjust the input size and hidden size according to your specific dataset and requirements.

7. **Using the trained model**:
   After training, you can use the `encode()` and `decode()` functions to process new data with your trained sparse autoencoder.

Note: Make sure to prepare your dataset and adjust the data loading process in the script according to your specific data format and location.


