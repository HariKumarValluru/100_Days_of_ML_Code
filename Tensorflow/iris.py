import pandas as pd

dataset = pd.read_csv('Dataset/iris.csv')

dataset.head()

# renaming column names
dataset.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'target']

dataset.head()

dataset['target'] = dataset['target'].apply(int)

dataset.head()

X = dataset.drop('target', axis=1)
y = dataset['target']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

import tensorflow as tf

# feature columns
feat_cols  = []
for col in X.columns:
    feat_cols.append(tf.feature_column.numeric_column(col))
    
input_func = tf.estimator.inputs.pandas_input_fn(x=X_train, y=y_train, batch_size=10, 
                                    num_epochs=5, shuffle=True)

classifier = tf.estimator.DNNClassifier(hidden_units = [10,20,10], n_classes=3,
                                        feature_columns=feat_cols)

classifier.train(input_fn=input_func, steps=50)

pred_fn = tf.estimator.inputs.pandas_input_fn(x=X_test, batch_size=len(X_test))