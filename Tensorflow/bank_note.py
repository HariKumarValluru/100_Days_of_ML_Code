# Using Bank Authentication Data Set classify whether or not a Bank Note was authentic.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv("Dataset/bank_note_data.csv")

dataset.head()

sns.countplot('Class', data=dataset)

sns.pairplot(dataset, hue='Class')

# standard scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(dataset.drop('Class', axis=1))

scaled_features = scaler.transform(dataset.drop('Class', axis=1))

features = pd.DataFrame(scaled_features, columns=dataset.columns[:-1])
features.head()

X = features
y = dataset['Class']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                    random_state=101)

import tensorflow as tf

cols = []
for col in features.columns:
    cols.append(tf.feature_column.numeric_column(col))
    
classifier = tf.estimator.DNNClassifier(hidden_units=[10,20,10], n_classes=2,
                                        feature_columns=cols)

input_func = tf.estimator.inputs.pandas_input_fn(x=X_train, y=y_train, 
                                                 batch_size=20, shuffle=True)

classifier.train(input_func, steps=500)

pred_fn = tf.estimator.inputs.pandas_input_fn(x=X_test, batch_size=len(X_test),
                                              shuffle=False )
note_predictions = list(classifier.predict(pred_fn))

preds = []
for pred in note_predictions:
    preds.append(pred['class_ids'][0])

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
print(confusion_matrix(y_test, preds))
print(classification_report(y_test, preds))
print(accuracy_score(y_test, preds))



