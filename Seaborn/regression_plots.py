# Regression Plots

# Importing the libraries
import seaborn as sns

# load the dataset
tips = sns.load_dataset('tips') 

# LM plot
sns.lmplot('total_bill', 'tip', data=tips )
sns.lmplot('total_bill', 'tip', data=tips, hue='sex')
sns.lmplot('total_bill', 'tip', data=tips, hue='sex', markers=['o','v'])
sns.lmplot('total_bill', 'tip', data=tips, hue='sex', markers=['o','v'], 
           scatter_kws={'s': 100})
sns.lmplot('total_bill', 'tip', data=tips)
sns.lmplot('total_bill', 'tip', data=tips, col = 'sex')
sns.lmplot('total_bill', 'tip', data=tips, col = 'sex', row='time')
sns.lmplot('total_bill', 'tip', data=tips, col = 'day', row='time', hue='sex')
sns.lmplot('total_bill', 'tip', data=tips, col = 'day', hue='sex')
sns.lmplot('total_bill', 'tip', data=tips, col = 'day', hue='sex', aspect=0.6, 
           size=9)