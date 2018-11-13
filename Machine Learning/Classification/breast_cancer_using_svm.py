#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import datasets

#%%
# loading the breast  cancer dataset
cancer = datasets.load_breast_cancer()

# creating dataset from cancer data
dataset = pd.DataFrame(cancer['data'], columns=cancer['feature_names'])

#%%
X = dataset.iloc[:].values
y = cancer['target']

# splitting the dataset in to training and test sets
from Utils.ml_utils import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=101)
#%%
