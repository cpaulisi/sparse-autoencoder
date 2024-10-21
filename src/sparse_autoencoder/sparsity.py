"""
Sparsity Module for Sparse Autoencoder

This module provides functions to compute sparsity-related metrics and losses
for a sparse autoencoder. It includes implementations of Kullback-Leibler (KL)
divergence and sparsity loss calculation.

The main components of this module are:

1. kl_divergence: Computes the KL divergence between target and estimated sparsity.
2. sparse_loss: Calculates the sparsity loss for the autoencoder's hidden layer activations.

These functions are crucial for encouraging sparsity in the hidden layer
representations of a sparse autoencoder, which helps in learning more efficient
and meaningful features from the input data.

Example usage:
    rho = 0.05  # target sparsity
    activation = model.encode(input_data)  # hidden layer activation
    loss = sparse_loss(rho, activation)

Note: This module assumes the use of PyTorch for tensor operations.
"""

import torch

def kl_divergence(rho: float, rho_hat: torch.Tensor) -> torch.Tensor:
    """
    Compute the Kullback-Leibler (KL) divergence between two Bernoulli distributions.

    This function calculates the KL divergence between a target sparsity (rho)
    and the estimated sparsity (rho_hat) of the hidden layer activations.

    Args:
        rho (float): The target sparsity parameter, typically a small value close to 0.
        rho_hat (torch.Tensor): The estimated sparsity, computed as the mean activation
                                of the hidden layer.

    Returns:
        torch.Tensor: The KL divergence between rho and rho_hat.

    Note:
        The KL divergence is used as part of the sparsity penalty in the loss function
        of the sparse autoencoder. It encourages the average activation of hidden units
        to be close to the target sparsity.
    """
    # the divergence will be a torch tensor
    divergence: torch.Tensor
    divergence = rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat))
    return divergence

def sparse_loss(rho: float, activation: torch.Tensor) -> torch.Tensor:
    """
    Compute the sparsity loss for the sparse autoencoder.

    This function calculates the sparsity loss using the Kullback-Leibler (KL) divergence
    between the target sparsity and the average activation of the hidden layer.

    Args:
        rho (float): The target sparsity parameter, typically a small value close to 0.
        activation (torch.Tensor): The activation values of the hidden layer.

    Returns:
        torch.Tensor: The sparsity loss, computed as the sum of KL divergences for each unit
                      in the hidden layer.

    Note:
        This loss encourages the average activation of hidden units to be close to the
        target sparsity, promoting a sparse representation in the hidden layer.
    """
    rho_hat: torch.Tensor = torch.mean(activation, dim=0)
    return torch.sum(kl_divergence(rho, rho_hat))