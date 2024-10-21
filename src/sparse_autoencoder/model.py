import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class SparseAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SparseAutoencoder, self).__init__()
        self.encoder = nn.Linear(input_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, input_size)
        
    def forward(self, x):
        activation = torch.sigmoid(self.encoder(x))
        output = torch.sigmoid(self.decoder(activation))
        return output, activation

def kl_divergence(rho, rho_hat):
    return rho * torch.log(rho / rho_hat) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat))

def sparse_loss(rho, activation):
    rho_hat = torch.mean(activation, dim=0)
    return torch.sum(kl_divergence(rho, rho_hat))

# Hyperparameters
input_size = 784  # for MNIST
hidden_size = 100
learning_rate = 1e-3
num_epochs = 20
batch_size = 128
rho = 0.05  # desired sparsity parameter
beta = 0.5  # weight of the sparsity penalty term

# Create the model
model = SparseAutoencoder(input_size, hidden_size)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Assuming you have a DataLoader called train_loader for your dataset

for epoch in range(num_epochs):
    for data in train_loader:
        img, _ = data
        img = img.view(img.size(0), -1)
        
        # Forward pass
        outputs, activation = model(img)
        
        # Compute loss
        mse_loss = criterion(outputs, img)
        sparsity_loss = sparse_loss(rho, activation)
        loss = mse_loss + beta * sparsity_loss
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

print("Training finished!")

# To use the trained autoencoder
def encode(x):
    model.eval()
    with torch.no_grad():
        _, activation = model(x)
    return activation

def decode(h):
    model.eval()
    with torch.no_grad():
        output = torch.sigmoid(model.decoder(h))
    return output