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

train_set['train_test'] = 'train'
test_set['train_test'] = 'test'

g = sns.FacetGrid(train_set, 'rating')
g.map(plt.hist, 'text length', bins=70)

sns.boxplot('rating', 'text length', data=train_set)

sns.countplot('rating', data=train_set, palette='rainbow')

ratings = train_set.groupby('rating').mean()

ratings

ratings.corr()

sns.heatmap(ratings.corr(), cmap='coolwarm', annot=True)

# merging training and test sets
merged_set = pd.concat([train_set,test_set],axis=0)

