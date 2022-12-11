import matplotlib.pyplot as plt
import numpy as np
import torch

from torch import float64
from torch import nn, optim


# --------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------
class NeuralNetwork(nn.Module):
    
    def __init__(self, num_inputs, num_outputs, num_hidden=None, num_layers=2):
        super().__init__()
        
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        
        if num_hidden is None:
            num_hidden = num_inputs
        
        self.layers = torch.nn.ModuleList()
        for _ in range(num_layers-1):
            self.layers.append(nn.Linear(num_inputs, num_hidden, dtype=float64))
            num_inputs = num_hidden
        self.layers.append(nn.Linear(num_inputs, num_outputs, dtype=float64))
            
        
    def forward(self, x):
        for layer in self.layers:
            x = torch.sigmoid(layer(x))
        
        return x


class Logistic(nn.Module):
    
    def __init__(self, num_inputs, num_outputs):
        super().__init__()
        
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        
        self.linear = nn.Linear(num_inputs, num_outputs, dtype=float64)
        
    def forward(self, X):
        return torch.sigmoid(self.linear(X))


# --------------------------------------------------------------------------------
# General Functions
# --------------------------------------------------------------------------------
def train_model(model, data, criterion=None, optimizer=None):
    X = torch.from_numpy(data.values[:,:-1])
    y = torch.from_numpy(data.values[:, -1])
    
    if criterion is None:
        criterion = nn.BCELoss()

    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=0.001)

    for _ in range(10000):
        training_step(X, y, model, criterion, optimizer)
        
        
def training_step(X, y, model, criterion, optimizer):
    model.zero_grad()
        
    y_pred = model(X)
    
    loss = criterion(y_pred.squeeze(), y)
    loss.backward()
    
    optimizer.step()
    
    return loss.detach().numpy()
