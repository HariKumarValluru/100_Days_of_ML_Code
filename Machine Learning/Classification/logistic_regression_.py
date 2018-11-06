# Logistic Regression without scikit learn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing utilities
from Utils.ml_utils import train_test_split
from Utils.ml_utils import normalize

ad_data = pd.read_csv("Datasets/advertising.csv")

X = ad_data.iloc[:, [0,1,2,3,6]].values
y = ad_data.iloc[:, -1].values

# Normalize the X set
X = normalize(X)

# Spliting the data into trainig and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                    random_state=42)

# initializing the Model and fitting the training data
from Models import LogisticRegression

model = LogisticRegression(learning_rate=0.001)
model.fit(X_train, y_train)

# predicting the test data
y_pred = model.predict(X_test)

from Utils.ml_utils import accuracy_score
accuracy = accuracy_score(y_test, y_pred) *100
print("Accuracy: {:.2f}%".format(accuracy))