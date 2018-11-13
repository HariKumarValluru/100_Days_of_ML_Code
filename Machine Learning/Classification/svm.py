import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import svm
import matplotlib.pyplot as plt
from Utils.plotting import Plot

iris = datasets.load_iris()
X = iris.data[:, :2]
Y = iris.target

C = 1.0  # SVM regularization parameter

svm_clf = Pipeline((
    ("scaler",StandardScaler()),
    ("linear_svc",svm.LinearSVC(C=C,loss='hinge')),
    ))

svm_clf.fit(X,Y)
svm_clf.predict([[4.3,3]])

models = (svm.SVC(kernel='linear', C=C),
          svm.LinearSVC(C=C))
models = (clf.fit(X,Y) for clf in models)

# Set-up 1x2 grid for plotting.
fig, sub = plt.subplots(1, 2, figsize=(12, 5))

X0, X1 = X[:, 0], X[:, 1]
xx, yy = Plot.make_meshgrid(X0, X1)

# title for the plots
titles = ('SVC with linear kernel',
          'LinearSVC (linear kernel)')

for clf, title, ax in zip(models, titles, sub.flatten()):
    Plot.plot_contours(ax, clf, xx, yy,
                  cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=Y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('Sepal length')
    ax.set_ylabel('Sepal width')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)

plt.show()