# Mulitple Lineear Regression without scikit learn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linear_model.ml_utils import train_test_split, compute_b0_bn

# load data
dataset = pd.read_csv("datasets/50_Startups.csv")

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Remove the dummy variable trap - we remove the first column
X = X[:, 1:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, 
                                                    random_state=101)

ret = compute_b0_bn((y_test.shape[0], 1), X_test)

y_pred = X.dot(ret[1])