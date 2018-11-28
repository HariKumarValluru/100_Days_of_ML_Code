import pandas as pd

dataset = pd.read_csv('Dataset/iris.csv')

dataset.head()

# renaming column names
dataset.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

dataset.head()

dataset['target'] = dataset['target'].apply(int)

dataset.head()

X = dataset.drop('target', axis=1).values
y = dataset['target']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)