# Style and color

# Importing the libraries
import seaborn as sns

# load the dataset
tips = sns.load_dataset('tips') 

sns.set_style('darkgrid')
sns.set_style('white')
sns.set_style('ticks')

sns.set_context('poster', font_scale=3)
sns.set_context('notebook')
sns.countplot('sex', data=tips)
