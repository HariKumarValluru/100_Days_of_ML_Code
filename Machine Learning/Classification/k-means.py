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