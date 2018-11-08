# K Nearest Neighbors without scikit learn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# loading the dataset
dataset = pd.read_csv("Datasets/Social_Network_Ads.csv")

X = dataset.iloc[:, 2:4].values
y = dataset.iloc[:, -1].values

# splitting the dataset in to training and test sets
from Utils.ml_utils import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, 
                                                    random_state=101)

# Feature Scaling
from Utils.ml_utils import standardize
X_train = standardize(X_train)
X_test = standardize(X_test)

# Fitting the model to trainning set
from Models.neighbors import KNN
model = KNN(n_neighbors=1)
model.fit(X_train, y_train)

# predicting the test set
y_pred = model.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# tracking the error rate
erro_rate = []
for i in range(1,50):
    model = KNN(n_neighbors=i)
    model.fit(X_train, y_train)
    
    # predicting the test set
    y_pred_i = model.predict(X_test)
    
    erro_rate.append(np.mean(y_pred_i != y_test))

# Visualising the error rate vs k values
plt.plot(range(1,50), erro_rate, color = 'blue', marker='o', markerfacecolor="red",
         markersize=10)
plt.title("Error Rate vs K value")
plt.xlabel("K")
plt.ylabel("Error Rate")

