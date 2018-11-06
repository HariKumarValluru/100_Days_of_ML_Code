# Mulitple Lineear Regression without scikit learn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Utils.ml_utils import cost_function, gradient_descent

dataset = pd.read_csv('datasets/student.csv')

X = dataset.iloc[:, 0:2].values
y = dataset.iloc[:, -1].values

n,_ = dataset.shape
x0 = np.ones(n)
X = np.column_stack((x0, X))

# Initial Coefficients
B = np.array([0, 0, 0])
alpha = 0.0001
inital_cost = cost_function(X, y, B)

# 100000 Iterations
newB, cost_history = gradient_descent(X, y, B, alpha, 100000)

Y_pred = X.dot(newB)

from linear_model.ml_utils import rmse, r2_score

print("RMSE: {0:.3f}".format(rmse(y, Y_pred)))

print("R2 Score: {0:.4f}".format(r2_score(y, Y_pred)))