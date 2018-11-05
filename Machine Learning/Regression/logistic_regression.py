# logistic Regression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ad_data = pd.read_csv("datasets/advertising.csv")

ad_data['Age'].plot.hist(bins=50)

sns.jointplot('Age', 'Area Income', data=ad_data)

sns.jointplot('Age', 'Daily Time Spent on Site', data=ad_data, kind='kde', color="red")

sns.jointplot('Daily Time Spent on Site', 'Daily Internet Usage', data=ad_data, color="green")

sns.pairplot(ad_data, hue='Clicked on Ad')

X = ad_data.iloc[:, [0,1,2,3,6]].values
y = ad_data.iloc[:, -1].values

from linear_model.ml_utils import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
