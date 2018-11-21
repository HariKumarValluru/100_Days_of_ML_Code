"""
Drug Review dataset provides patient reviews on specific drugs along with 
related conditions and a 10 star patient rating reflecting overall patient 
satisfaction.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train_set = pd.read_csv('Datasets/drugsComTrain.tsv', sep="\t")
test_set = pd.read_csv('Datasets/drugsComTest.tsv', sep="\t")

train_set.head()
test_set.head()

train_set.info()
test_set.info()

train_set.describe()
test_set.describe()

# adding length, train, test on to training and test set
train_set['text length'] = train_set['review'].apply(len)

test_set['text length'] = test_set['review'].apply(len)

train_set['is_train'] = 1 
test_set['is_train'] = 0

g = sns.FacetGrid(train_set, 'rating')
g.map(plt.hist, 'text length', bins=70)

sns.boxplot('rating', 'text length', data=train_set)

sns.countplot('rating', data=train_set, palette='rainbow')

ratings = train_set.groupby('rating').mean()

ratings

ratings.corr()

sns.heatmap(ratings.corr(), cmap='coolwarm', annot=True)

# merging training and test sets
merged_set = pd.concat([train_set,test_set],axis=0, ignore_index=True)

train_data = merged_set.loc[merged_set.is_train == 1].values
train_data.reset_index(inplace=True,drop=True)

test_data = merged_set.loc[merged_set.is_train == 0].values
test_data.reset_index(inplace=True,drop=True)

y = merged_set['rating'].values #ratings
X = merged_set['review'].values #covariates or our independent variables

# create a CountVectorizer object
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()

X = cv.fit_transform(X)

from sklearn.tree import DecisionTreeClassifier
m = DecisionTreeClassifier()
predictions = np.zeros(y.shape) #creating an empty prediction array

from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=2)
for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
 
    m.fit(X_train, y_train)
    probs = m.predict_proba(X_test)[:, 1] #calculating the probability
    predictions[test_idx] = probs