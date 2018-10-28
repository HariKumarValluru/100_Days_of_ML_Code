# Visualise titanic data with seaborn

# Importing the libraries
import seaborn as sns
import matplotlib.pyplot as plt

# load the titanic dataset
sns.set_style('whitegrid')
titanic = sns.load_dataset('titanic')

# joint grid
sns.jointplot('fare','age',data=titanic)

# dist plot
sns.distplot(titanic['fare'], kde=False, color='red', bins=30)

# Box plot
sns.boxplot('class', 'age', data=titanic, palette='rainbow')

# Swarm plot
sns.swarmplot('class', 'age', data=titanic, palette='Set2')

# Bar plot
sns.countplot('sex', data=titanic)

# Heatmap
tc = titanic.corr()
sns.heatmap(tc, cmap='coolwarm')

# Facet grid
g = sns.FacetGrid(data=titanic, col='sex')
g.map(sns.distplot, 'age')