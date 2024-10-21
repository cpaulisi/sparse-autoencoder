"""
Sparse Autoencoder Model

This module implements a Sparse Autoencoder using PyTorch. A Sparse Autoencoder
is an unsupervised learning algorithm that applies backpropagation to learn a
compressed, sparse representation of input data.

The main components of this module are:

1. SparseAutoencoder: A PyTorch nn.Module class that defines the architecture
   of the sparse autoencoder.
2. Training loop: Implements the training process for the sparse autoencoder,
   including forward pass, loss computation, and backpropagation.

The sparse autoencoder consists of an encoder that compresses the input data
into a lower-dimensional representation, and a decoder that attempts to
reconstruct the original input from this compressed representation. The key
feature is the encouragement of sparsity in the hidden layer activations.

The loss function combines reconstruction error (using Mean Squared Error) and
a sparsity penalty term (using Kullback-Leibler divergence).

Usage:
    model = SparseAutoencoder(input_size, hidden_size)
    # Train the model using the provided training loop
    # Use model.encoder(input_data) to get sparse representations

Note: This module assumes the use of PyTorch for neural network operations and
      the presence of a custom 'sparsity' module for sparsity-related calculations.

"""

from typing import Union, Type
import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
from .sparsity import sparse_loss

mse = nn.MSELoss()

class SparseAutoencoder(nn.Module):
    """
    A sparse autoencoder network.

    This class implements a sparse autoencoder, which is an unsupervised learning
    algorithm that applies backpropagation to learn a compressed, sparse
    representation of the input data.

    Attributes:
        encoder (nn.Linear): The encoder layer of the autoencoder.
        decoder (nn.Linear): The decoder layer of the autoencoder.
    """
    def __init__(self, input_size, hidden_size):
        super(SparseAutoencoder, self).__init__()
        self.encoder = nn.Linear(input_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, input_size)
        
    def forward(self, x):
        """
        Forward pass through the autoencoder.

        Args:
            x (torch.Tensor): Input tensor to be encoded and reconstructed.

        Returns:
            tuple: A tuple containing:
                - output (torch.Tensor): The reconstructed input after passing through the encoder and decoder.
                - activation (torch.Tensor): The activation of the hidden layer (encoded representation).

        The forward pass consists of:
        1. Encoding the input using the encoder layer and applying sigmoid activation.
        2. Decoding the encoded representation using the decoder layer and applying sigmoid activation.
        """
        activation = torch.sigmoid(self.encoder(x))
        output = torch.sigmoid(self.decoder(activation))
        return output, activation
    
    def train(self, 
            trainloader: torch.utils.data.DataLoader, 
            criterion: Union[Type[_Loss], None] = None,
            num_epochs: int = 20, 
            learning_rate: float = 1e-3,
            beta: float = 0.5, 
            rho: float = 0.05
        ):
        """
        Train the sparse autoencoder in-place.

        This method implements the training loop for the sparse autoencoder, including
        forward and backward passes, loss computation, and optimization.

        Args:
            trainloader (torch.utils.data.DataLoader): DataLoader for the training data.
            criterion (Type[_Loss]): A loss criterion for our optimization technique.
            num_epochs (int, optional): Number of training epochs. Defaults to 20.
            learning_rate (float, optional): Learning rate for the optimizer. Defaults to 1e-3.
            beta (float, optional): Weight of the sparsity penalty term. Defaults to 0.5.
            rho (float, optional): Desired sparsity parameter. Defaults to 0.05.

        The training process includes:
        1. Setting up the optimizer (Adam) and loss criterion (MSE).
        2. Iterating over the specified number of epochs.
        3. For each batch:
           - Performing a forward pass through the autoencoder.
           - Computing the reconstruction loss (MSE) and sparsity loss.
           - Combining the losses and performing backpropagation.
           - Updating the model parameters.
        4. Printing the loss for each epoch.

        Note:
            This method modifies the model's parameters in-place.
        """

        if criterion is None: 
            criterion = mse
        # specify optimizer here in-place
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        # iterate through epochs
        for epoch in range(num_epochs):
            for data in trainloader:
                # get data
                img, _ = data
                img = img.view(img.size(0), -1)
                # zero gradients
                optimizer.zero_grad()
                # forward pass
                outputs, activation = self(img)
                # compute loss
                mse_loss = criterion(outputs, img)
                sparsity_loss = sparse_loss(rho, activation)
                loss = mse_loss + beta * sparsity_loss
                # backward and optimize
                loss.backward()
                optimizer.step()
            
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

        print("Training finished!")

    def encode(self, x: torch.Tensor):
        """
        Encode the input data using the trained encoder.

        Args:
            x (torch.Tensor): Input data to be encoded.

        Returns:
            torch.Tensor: Encoded representation of the input data.

        Note:
            This method sets the model to evaluation mode and disables gradient computation.
        """
        self.eval()
        with torch.no_grad():
            _, activation = self(x)
        return activation

    def decode(self, h: torch.Tensor):
        """
        Decode the encoded representation using the trained decoder.

        Args:
            h (torch.Tensor): Encoded representation to be decoded.

        Returns:
            torch.Tensor: Reconstructed output from the encoded representation.

        Note:
            This method sets the model to evaluation mode and disables gradient computation.
        """
        self.eval()
        with torch.no_grad():
            output = torch.sigmoid(self.decoder(h))
        return output