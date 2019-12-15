# utf-8
import pandas
import numpy
import matplotlib.pyplot as plt
import matplotlib.image as mi
import os
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import datasets, model_selection, metrics


iris = datasets.load_iris()
print(iris.keys())
print(iris['target'])
print(type(iris['target']))
print(type(iris['data']))
# n_sample = iris['data'].shape[0]
# n_feature = iris['data'].shape[1]
n_sample, n_feature = iris['data'].shape
print("样本数", n_sample)
print("特征数", n_feature)


def load_data():
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        iris['data'], iris['target'], test_size=0.2, random_state=1
    )
    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = load_data()
clf = DecisionTreeClassifier(max_depth=3)
clf.fit(X_train, y_train)

result = clf.predict(X_test)
true_count = 0
for i in range(len(X_test)):
    if result[i] == y_test[i]:
        true_count += 1

true_rate = true_count / len(X_test)
print(true_rate)
print(clf.score(X_test, y_test))
print(clf.decision_path(X_test))
print(metrics.classification_report(result, y_test))
with open("tree.dot", 'w') as f:
    f = export_graphviz(clf, out_file=f)

excute = r"dot -Tpng tree.dot -o tree.png"
print(excute)
b = os.system(excute)
m = mi.imread('tree.png')
plt.imshow(m)
plt.axis('off')
plt.show()
