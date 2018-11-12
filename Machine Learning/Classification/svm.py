import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

iris = datasets.load_iris()
X = iris.data[:, :2]
Y = iris.target

svm = Pipeline((
    ("scaler",StandardScaler()),
    ("linear_svc",LinearSVC(C=1.0,loss='hinge')),
    ))

svm.fit(X,Y)
svm.predict([[4.3,3]])