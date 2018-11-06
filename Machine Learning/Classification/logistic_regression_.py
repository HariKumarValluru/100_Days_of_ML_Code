# Logistic Regression without scikit learn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing utilities
from Utils.ml_utils import train_test_split

ad_data = pd.read_csv("Datasets/advertising.csv")

X = ad_data.iloc[:, [0,1,2,3,6]].values
y = ad_data.iloc[:, -1].values

# Normalize the X set