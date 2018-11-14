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

#%%
mall_data = pd.read_csv('Datasets/Mall_Customers.csv')
sns.pairplot(mall_data)
X = mall_data.iloc[:, [3, 4]].values
#%%
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
#%%
# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)
#%%
# Visualising the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1', edgecolors='black')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2', edgecolors='black')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3', edgecolors='black')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4', edgecolors='black')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5', edgecolors='black')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids', edgecolors='black')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()