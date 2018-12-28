""" predicting the likelihood of e-signing a loan based on financial history """

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn

dataset = pd.read_csv('Datasets/financial_data.csv')

dataset.head()
dataset.columns
dataset.describe()

dataset.isna().any() 

dataset2 = dataset.drop(columns = ['entry_id', 'pay_schedule', 'e_signed'])

fig = plt.figure(figsize=(15, 12))
plt.suptitle('Histograms of Numerical Columns', fontsize=20)
for i in range(dataset2.shape[1]):
    plt.subplot(6, 3, i + 1)
    f = plt.gca()
    f.set_title(dataset2.columns.values[i])

    vals = np.size(dataset2.iloc[:, i].unique())
    if vals >= 100:
        vals = 100
    
    plt.hist(dataset2.iloc[:, i], bins=vals, color='#3F5D7D')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

dataset2.corrwith(dataset.e_signed).plot.bar(
        figsize = (20, 10), title = "Correlation with E Signed", fontsize = 15,
        rot = 45, grid = True)

sn.set(style="white")

corr = dataset2.corr()

mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(18, 15))

cmap = sn.diverging_palette(220, 10, as_cmap=True)

sn.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

train = pd.read_csv('Datasets/financial_data.csv')

train = train.drop(columns = ['months_employed'])
train['personal_account_months'] = (train.personal_account_m + (train.personal_account_y * 12))
train[['personal_account_m', 'personal_account_y', 'personal_account_months']].head()
train = train.drop(columns = ['personal_account_m', 'personal_account_y'])

train = pd.get_dummies(train)
train.train

train = train.drop(columns = ['pay_schedule_semi-monthly'])

response = train["e_signed"]
users = train['entry_id']
train = train.drop(columns = ["e_signed", "entry_id"])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train,
                                                    response,
                                                    test_size = 0.2,
                                                    random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train2 = pd.DataFrame(sc_X.fit_transform(X_train))
X_test2 = pd.DataFrame(sc_X.transform(X_test))
X_train2.columns = X_train.columns.values
X_test2.columns = X_test.columns.values
X_train2.index = X_train.index.values
X_test2.index = X_test.index.values
X_train = X_train2
X_test = X_test2


