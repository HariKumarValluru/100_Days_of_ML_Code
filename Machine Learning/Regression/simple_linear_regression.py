import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from linear_model import LinearRegression
from linear_model.ml_utils import train_test_split

dataset = pd.read_csv("datasets/EcommerceCustomers.csv")

X = dataset.iloc[:, 6:-1].values
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, 
                                                    random_state=101)

model = LinearRegression()
model.fit(X_train, y_train)

# Predicting the test set
y_pred = model.predict(X_test)

# Visualising the training set results
plt.scatter(X_train, y_train, color="red")
plt.plot(X_train, model.predict(X_train), color="blue")
plt.title("Linear Regression (Training Set)")
plt.xlabel("Length of Membership")
plt.ylabel("Yearly Amount Spent")

# Visualising the test set results
plt.scatter(X_test, y_test, color="red")
plt.plot(X_train, model.predict(X_train), color="blue")
plt.title("Linear Regression (Test Set)")
plt.xlabel("Length of Membership")
plt.ylabel("Yearly Amount Spent")

# visualising the test values vs predicted values
plt.scatter(y_test, y_pred)
plt.xlabel("Y Test (True Values)")
plt.ylabel("Predicted Values")

# Evaluating the model performance
from sklearn import metrics

# Variance Score
metrics.explained_variance_score(y_test, y_pred)

# exploring the residuals to make sure everything is okay with our data
sns.distplot((y_test - y_pred), bins = 50)
