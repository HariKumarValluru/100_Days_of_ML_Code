# K Nearest Neighbors without scikit learn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# loading the dataset
dataset = pd.read_csv("Datasets/Social_Network_Ads.csv")

X = dataset.iloc[:, 2:4].values
y = dataset.iloc[:, -1].values