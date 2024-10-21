# Sparse Autoencoder

> A repository for experimenting with use cases related to sparse autoencoders. Adapted from https://web.stanford.edu/class/cs294a/sparseAutoencoder.pdf.

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

5. **Train the model**:
   To train the sparse autoencoder, run the following command:
   ```
   python examples/train.py
   ```
   This script will:
   - Load the Fashion MNIST dataset
   - Create and initialize the sparse autoencoder model
   - Train the model for the specified number of epochs
   - Print the loss for each epoch

   You can modify the hyperparameters in `examples/train.py` to experiment with different settings:
   - `hidden_size`: Size of the hidden layer
   - `batch_size`: Number of samples per batch
   - `num_epochs`: Number of training epochs
   - `learning_rate`: Learning rate for the optimizer
   - `rho`: Desired sparsity parameter
   - `beta`: Weight of the sparsity penalty term

6. **Use the trained model**:
   After training, you can use the model to encode new data or reconstruct inputs. Here's a basic example:
   ```python
   import torch
   from sparse_autoencoder import SparseAutoencoder

   # Load your trained model
   model = SparseAutoencoder(input_size=784, hidden_size=100)
   model.load_state_dict(torch.load('path_to_saved_model.pth'))

   # Encode new data
   new_data = torch.randn(1, 784)  # Example input
   encoded = model.encode(new_data)

   # Reconstruct input
   reconstructed = model.decode(encoded)
   ```

Remember to adjust the model parameters and file paths according to your specific setup and trained model.

