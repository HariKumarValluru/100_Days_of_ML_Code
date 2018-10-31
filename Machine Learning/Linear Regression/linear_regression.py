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

