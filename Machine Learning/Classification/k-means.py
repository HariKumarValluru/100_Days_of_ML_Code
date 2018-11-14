# k-means clustering

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')

# loading dataset and setting first columns as an index
dataset = pd.read_csv("Datasets/College_Data.csv", index_col=0)

# visualising Room.Board vs Grad.Rate
sns.scatterplot('Room.Board', 'Grad.Rate', data=dataset, hue='Private')

# visualising Outstate vs F.Undergrad
sns.scatterplot('Outstate', 'F.Undergrad', data=dataset, hue='Private')

g = sns.FacetGrid(dataset, hue='Private', height=6, aspect=2)
g = g.map(plt.hist, 'Outstate', bins=20, alpha=.6)

from sklearn.cluster import KMeans
model = KMeans(n_clusters=2)
model.fit(dataset.drop('Private', axis=1))

# Evaluting the clustering
def converter(private):
    if private == 'Yes':
        return 1
    else:
        return 0
    
dataset['Cluster'] = dataset['Private'].apply(converter)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(dataset['Cluster'], model.labels_))
print(classification_report(dataset['Cluster'], model.labels_))