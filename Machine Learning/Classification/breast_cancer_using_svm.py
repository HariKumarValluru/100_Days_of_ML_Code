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
from sklearn.svm import SVC

model = SVC()
model.fit(X_train, y_train)

# predicting the test set
y_pred = model.predict(X_test)
#%%
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#%%
# finding the right parameters
from sklearn.grid_search import GridSearchCV

params = {'C': [0.1,1,10,100,1000], 'gamma': [1,.1,.01,.001,.0001]}
grid = GridSearchCV(SVC(), params, verbose=3)
grid.fit(X_train, y_train)

#%%
# printing the best parameters
grid.best_params_

y_grid_pred = grid.predict(X_test)

print(confusion_matrix(y_test, y_grid_pred))
print(classification_report(y_test, y_grid_pred))

#%%
# finding the accuracy
from Utils.ml_utils import accuracy_score

accuracy = accuracy_score(y_test, y_grid_pred) *100
print("Accuracy: {:.2f}%".format(accuracy))