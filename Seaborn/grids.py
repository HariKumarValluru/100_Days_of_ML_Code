# Grids

# Importing the library
#import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# load the dataset
iris = sns.load_dataset('iris')

sns.pairplot(iris)

g = sns.PairGrid(iris)
g.map(sns.distplot)
g.map(plt.scatter)
g.map(sns.kdeplot)

tips = sns.load_dataset('tips')
g = sns.FacetGrid(tips, col='time', row='smoker')
g.map(sns.distplot, 'total_bill')