# Categorical Plots

# Importing the library
import seaborn as sns

# loading dataset
tips = sns.load_dataset('tips')

tips.head()

# Bar Plot
sns.barplot(tips['sex'], tips['total_bill'], data=tips)

import numpy as np
sns.barplot(tips['sex'], tips['total_bill'], data=tips, estimator=np.std)

# Count plot
sns.countplot(tips['sex'], data=tips)

# Box plot
sns.boxplot('day', 'total_bill', data=tips)
sns.boxplot('day', 'total_bill', data=tips, hue='smoker')

# violin plot
sns.violinplot('day', 'total_bill', data=tips)
sns.violinplot('day', 'total_bill', data=tips, hue='sex')
sns.violinplot('day', 'total_bill', data=tips, hue='sex', split=True)

# Strip plot
sns.stripplot('day', 'total_bill', data=tips)
sns.stripplot('day', 'total_bill', data=tips, jitter=True)
sns.stripplot('day', 'total_bill', data=tips, jitter=True, hue='sex')
sns.stripplot('day', 'total_bill', data=tips, jitter=True, hue='sex', split=True)

# Swarm Plot
sns.swarmplot('day', 'total_bill', data=tips)

# Swarm and Violin Plot
sns.violinplot('day', 'total_bill', data=tips)
sns.swarmplot('day', 'total_bill', data=tips, color='black')