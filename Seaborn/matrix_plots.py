# Matrix Plots

# Importing the library
import seaborn as sns

# loading dataset
tips = sns.load_dataset('tips')
flights = sns.load_dataset('flights')

tc = tips.corr()

sns.heatmap(tc)
sns.heatmap(tc, annot=True)
sns.heatmap(tc, annot=True, cmap="coolwarm")

fp = flights.pivot_table(index='month', columns='year', values='passengers')
sns.heatmap(fp)
sns.heatmap(fp, cmap='magma')
sns.heatmap(fp, cmap='magma', linecolor='white', linewidths=1)

# Cluster map
sns.clustermap(fp)
sns.clustermap(fp, cmap='coolwarm')
sns.clustermap(fp, cmap='coolwarm', standard_scale=1)