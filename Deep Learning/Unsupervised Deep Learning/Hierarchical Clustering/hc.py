# Hierarchical Clustering

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Datasets/Mall_Customers.csv')
X = dataset.iloc[:, 3:].values
