# Analyse 911 call data

# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# load the dataset
dataset = pd.read_csv('911.csv')

# checking the info
dataset.info()

# getting top 5 zip codes for 911 calls
dataset['zip'].value_counts().head(5)

# getting top 5 towpships for 911 calls
dataset['twp'].value_counts().head(5)

# getting unique titles count
dataset['title'].nunique()

# create a new column as 'Reason'
dataset['Reason'] = dataset['title'].apply(lambda title: title.split(":")[0])

# most common reason to call 911
dataset['Reason'].value_counts()

# visualise the 911 calls by reason
sns.countplot('Reason', data=dataset)

# converting the timestamp from str to timestamp object
dataset['timeStamp'] = pd.to_datetime(dataset['timeStamp'])

# creating 3 columns for timestamp
dataset['Hour'] = dataset['timeStamp'].apply(lambda time: time.hour)
dataset['Month'] = dataset['timeStamp'].apply(lambda time: time.month)
dataset['Day of Week'] = dataset['timeStamp'].apply(lambda time: time.dayofweek)

# converting day of week to week name
dmap = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat', 6: 'Sun'}
dataset['Day of Week'] = dataset['Day of Week'].map(dmap)

# visualize the day of week with reason column
sns.countplot('Day of Week', data = dataset, hue='Reason')
# relocate the legend 
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)

# visualize the month with reason column
sns.countplot('Month', data = dataset, hue='Reason')
# relocate the legend 
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)

# fix missing months
bymonth = dataset.groupby('Month').count()

# count call per month
bymonth['lat'].plot()

# creating a linear model plot for number of calls per month
sns.lmplot('Month','twp', data=bymonth.reset_index())