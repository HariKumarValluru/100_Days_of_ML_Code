import numpy as np
import seaborn as sns
from sklearn import datasets
from Utils.ml_utils import normalize, train_test_split

data = datasets.load_iris()
X = normalize(data.data[data.target != 0])
y = data.target[data.target != 0]
y[y == 1] = -1
y[y == 2] = 1

# splitting the dataset in to training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, 
                                                    random_state=101)

from Models.svm import SVM
clf = SVM(C=1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

from Utils.ml_utils import accuracy_score
accuracy = accuracy_score(y_test, y_pred) *100
print("Accuracy: {:.2f}%".format(accuracy))