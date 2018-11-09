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
from Utils.ml_utils import standardize
X_train = standardize(X_train)
X_test = standardize(X_test)

# Fitting the model to trainning set
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=21, p=2)
model.fit(X_train, y_train)

# predicting the test set
y_pred = model.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# tracking the error rate
erro_rate = []
for i in range(1,50):
    model = KNeighborsClassifier(n_neighbors=i)
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


from Utils.ml_utils import accuracy_score
accuracy = accuracy_score(y_test, y_pred)

print ("Accuracy:", accuracy)

#Reduce dimensions to 2d using pca and plot the results
from Utils.plotting import Plot
Plot().plot_in_2d(X_test, y_pred, title="K Nearest Neighbors", accuracy=accuracy)