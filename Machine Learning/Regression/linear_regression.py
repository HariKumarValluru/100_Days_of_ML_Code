# Analyze the customer data of a eCommerce company

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# loading the dataset
customers = pd.read_csv("EcommerceCustomers.csv")

# getting the object types
customers.info()

# getting the statistical information of the numerical columns
customers.describe()

# Comparing Time on Website and Yearly Amount Spent columns
sns.jointplot('Time on Website', 'Yearly Amount Spent', data = customers)

# Comparing Time on App and Yearly Amount Spent columns
sns.jointplot('Time on App', 'Yearly Amount Spent', data = customers)

# Comparing Time on App and Length of Membership
sns.jointplot('Time on App', 'Length of Membership', data = customers, 
              kind = 'hex')

sns.pairplot(customers)

# Creating a linear model plot for Yearly Amount Spent vs Length of Membership
sns.lmplot('Yearly Amount Spent', 'Length of Membership', data = customers)

# Spliting the data into training and test sets
X = customers.iloc[:, 3:7].values
y = customers.iloc[:, -1].values


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, 
                                                    random_state = 101)

# Training our model on our training data
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

# coefficients of the model
model.coef_

# predicting the test data
y_pred = model.predict(X_test)

# visualising the test values vs predicted values
plt.scatter(y_test, y_pred)
plt.xlabel("Y Test (True Values)")
plt.ylabel("Predicted Values")

# Evaluating the model performance
from sklearn import metrics

# Calculating Mean Absoulte Error
print("MAE", metrics.mean_absolute_error(y_test, y_pred))

# Calculating Mean Squared Error
print("MSE", metrics.mean_squared_error(y_test, y_pred))

# Calculating Root Mean Squared Error
print("RMSE", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# Variance Score
metrics.explained_variance_score(y_test, y_pred)

# exploring the residuals to make sure everything is okay with our data
sns.distplot((y_test - y_pred), bins = 50)

# coeffecients
pd.DataFrame(model.coef_, customers.iloc[:, 3:7].columns,columns = ['Coeffecient'])
