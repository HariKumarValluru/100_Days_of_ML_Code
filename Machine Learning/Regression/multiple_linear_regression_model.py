# Mulitple Lineear Regression without scikit learn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('datasets/50_Startups.csv')

n,_ = dataset.shape

x0 = np.ones(n)
X = dataset.iloc[:, 0:3].values
X = np.array([x0, X[0], X[1], X[2]]).T

# Initial Coefficients
B = np.array([0, 0, 0])

Y = dataset.iloc[:, -1].values
alpha = 0.0001