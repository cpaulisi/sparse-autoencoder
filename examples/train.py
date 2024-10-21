"""
Sparse Autoencoder Training Module

This module implements the training process for a Sparse Autoencoder using PyTorch.
It includes model creation, data loading, and the training loop with loss computation
and optimization.

The module uses a SparseAutoencoder model and trains it on a given dataset,
encouraging sparsity in the hidden layer representations.
"""
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from sparse_autoencoder import SparseAutoencoder

# model size configuration
input_size = 784  # for MNIST
hidden_size = 100

# the batch size
batch_size = 128

# hyperparameters
num_epochs = 20
learning_rate = 2e-3
rho = 0.05  # desired sparsity parameter
beta = 0.6  # weight of the sparsity penalty term


# Create the model
model = SparseAutoencoder(input_size, hidden_size)

# image transformations
transform = transforms.Compose([
    transforms.ToTensor(),
])

trainset = datasets.FashionMNIST(
    root='../data/input',
    train=True, 
    download=True,
    transform=transform
)

testset = datasets.FashionMNIST(
    root='../data/input',
    train=False,
    download=True,
    transform=transform
)

# trainloader
trainloader = DataLoader(
    trainset, 
    batch_size=batch_size,
    shuffle=True
)
#testloader
testloader = DataLoader(
    testset, 
    batch_size=batch_size, 
    shuffle=False
)

model.train(
    trainloader=trainloader, 
    learning_rate=learning_rate, 
    beta=beta, rho=rho)