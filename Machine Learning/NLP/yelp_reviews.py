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
