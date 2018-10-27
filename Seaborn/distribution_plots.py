# Learning Seaborn Distribution Plots

# Importing the library
import seaborn as sns

# loading dataset
tips = sns.load_dataset('tips')

tips.head()

# dist plot allows us to show distribution of univariate set of observations
sns.distplot(tips['total_bill'], kde=False, bins=30)

# Joint plot is allows us to matchup
sns.jointplot(x=tips['total_bill'], y=tips['tip'], data=tips, kind='hex')
sns.jointplot(x=tips['total_bill'], y=tips['tip'], data=tips, kind='reg')
sns.jointplot(tips['total_bill'], tips['tip'], data=tips, kind='kde')
sns.jointplot(tips['total_bill'], tips['tip'], data=tips)

'''
pairplot is do the jointplot for every possible combination of the numerical 
columns in the dataframe
'''
sns.pairplot(tips)
sns.pairplot(tips, hue='sex')
sns.pairplot(tips, hue='sex', palette="coolwarm")

sns.rugplot(tips['total_bill'])

sns.kdeplot(tips['total_bill'])