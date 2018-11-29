# Using Bank Authentication Data Set classify whether or not a Bank Note was 
# authentic.

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