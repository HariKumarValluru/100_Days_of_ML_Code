# K-nearst neighbours
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# loading the dataset
dataset = pd.read_csv("Datasets/KNN_Project_Data.csv")

X = dataset.iloc[:, 0:10].values
y = dataset.iloc[:, -1].values

# splitting the dataset in to training and test sets
from Utils.ml_utils import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                    random_state=101)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)