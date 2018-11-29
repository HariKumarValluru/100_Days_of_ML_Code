# Using Bank Authentication Data Set classify whether or not a Bank Note was authentic.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv("Dataset/bank_note_data.csv")

dataset.head()

sns.countplot('Class', data=dataset)

sns.pairplot(dataset, hue='Class')

# standard scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(dataset.drop('Class', axis=1))

scaled_features = scaler.transform(dataset.drop('Class', axis=1))

features = pd.DataFrame(scaled_features, columns=dataset.columns[:-1])
features.head()

X = features
y = dataset['Class']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                    random_state=101)
