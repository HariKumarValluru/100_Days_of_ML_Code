"""
Using Decision Trees and Random Forest to predict whether or not the borrower
paid back their loan in full.
Dataset: LendingClub.com
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

loans = pd.read_csv("Datasets/loan_data.csv")

loans[loans['credit.policy'] == 1]['fico'].hist(bins=35, label="Credit Policy = 1",
     alpha=.6)
loans[loans['credit.policy'] == 0]['fico'].hist(bins=35, label="Credit Policy = 0",
     alpha=.6)
plt.legend()
plt.xlabel("FICO score")

loans[loans['not.fully.paid'] == 1]['fico'].hist(bins=35, label="Not Fully Paid = 1", 
     alpha=.6)
loans[loans['not.fully.paid'] == 0]['fico'].hist(bins=35, label="Not Fully Paid = 0", 
     alpha=.6)
plt.legend()
plt.xlabel("FICO score")

sns.countplot('purpose', data=loans, hue="not.fully.paid", palette = 'Set1')

sns.jointplot('fico', 'int.rate', data=loans)

sns.lmplot('fico', 'int.rate', data=loans, hue='credit.policy', col='not.fully.paid',
           palette='Set1')

# Creating new feature columns with dummy variables
cat_feats = ['purpose']
final_data = pd.get_dummies(loans, columns=cat_feats, drop_first=True)

X = final_data.drop('not.fully.paid', axis=1).iloc[:].values
y = final_data.iloc[:, 12].values

# Splitting the dataset into training and test set
from Utils.ml_utils import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                        random_state=101)

# Training a Decision Tree Model
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)

d_pred = dtree.predict(X_test)


from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test, d_pred))
print(classification_report(y_test, d_pred))
