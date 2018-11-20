"""
Classify Yelp reviews into 1 star or 5 star based on text content in the 
reviews.
"""
# libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# reading the dataset
yelp = pd.read_csv("Datasets/yelp.csv")

yelp.head(2)

yelp.info()

yelp.describe()

# creating a new column for length
yelp['text length'] = yelp['text'].apply(len)

# exploring the data
sns.set_style('white')

g = sns.FacetGrid(yelp, col='stars')
g.map(plt.hist, 'text length', bins=60)

sns.boxplot('stars', 'text length', data = yelp)

sns.countplot('stars', data=yelp)

stars = yelp.groupby('stars').mean()

stars

stars.corr()

sns.heatmap(stars.corr(), cmap = 'coolwarm', annot=True)

yelp_class = yelp[(yelp['stars'] == 1) | (yelp['stars'] == 5)]

yelp.info

X = yelp_class['text']
y = yelp_class['stars']

# create a CountVectorizer object
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer()

X = cv.fit_transform(X)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                    random_state=101)

# training model
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()

rf.fit(X_train, y_train)

predictions = rf.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))