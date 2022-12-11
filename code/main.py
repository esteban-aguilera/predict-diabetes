# %%
# --------------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------------
import importlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import warnings

from sklearn import metrics

# package imports
import src
from src import preprocess, models


src._chdir_to_project_path()


# --------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------
warnings.filterwarnings("error")


# --------------------------------------------------------------------------------
# Load Data
# --------------------------------------------------------------------------------
data = preprocess.load_diabetes_data()


# %%
importlib.reload(models)


model = models.Logistic(data.shape[1] - 1, 1)
models.train_model(model, data)

with torch.no_grad():
    y_pred = model(torch.tensor(data.values[:,:-1]))

print(metrics.roc_auc_score(data.values[:,-1], y_pred.squeeze().numpy()))


# %%
importlib.reload(models)


model = models.NeuralNetwork(data.shape[1] - 1, 1)
models.train_model(model, data)

with torch.no_grad():
    y_pred = model(torch.tensor(data.values[:,:-1]))

print(metrics.roc_auc_score(data.values[:,-1], y_pred.squeeze().numpy()))


# %%
