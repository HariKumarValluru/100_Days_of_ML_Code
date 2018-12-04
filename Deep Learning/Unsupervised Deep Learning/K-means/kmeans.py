# K-Means Clustering

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Datasets/Mall_Customers.csv')
X = dataset.iloc[:, 3:].values