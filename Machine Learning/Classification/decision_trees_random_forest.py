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