"""
import pandas as pd
from sklearn import svm
file = 'data/train.csv'

train_data = pd.read_csv(file)

print(train_data.head())

print(train_data.columns)

#features = Sex, Age, Pclass, Cabin, SibSp, Parch, Embarked, Name, Ticket
#label = Survived

#'PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp','Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'

#SVM
#Bayesian logisitic regression
kernel = 'rbf'
svm.SVC()
"""

# Extract features using sliding window and form the training dataset, test dataset

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture

import numpy as np

X, y = make_classification(n_samples=10000, n_features=6,
                            n_informative=3, n_redundant=0,
                            random_state=0, shuffle=True)

print(X.shape)  # 10000x6
print(y.shape)  # 10000

# TODO: Feature extraction using sliding window

train_features, test_features, train_labels, test_labels = train_test_split(X, y,
                                                                            test_size=0.25, random_state=42)
# TODO: K-fold cross validation

print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

clf = RandomForestClassifier(n_estimators=100, max_depth=3, oob_score=True
                             )

clf.fit(X, y)

print(clf.feature_importances_)
#print(clf.oob_decision_function_)
print(clf.oob_score_)

predictions = clf.predict(test_features)
errors = abs(predictions - test_labels)
print("M A E: ", round(np.mean(errors), 2))


# Visualization
feature_list = [1, 2, 3, 4, 5, 6]
from sklearn.tree import export_graphviz
import pydot
# Pull out one tree from the forest
tree = clf.estimators_[5]
# Export the image to a dot file
export_graphviz(tree, out_file='tree.dot', feature_names=feature_list, rounded=True, precision=1)
# Use dot file to create a graph
(graph, ) = pydot.graph_from_dot_file('tree.dot')
# Write graph to a png file
#graph.write_png('tree_.png')

# TODO: Confusion matrix, Accuracy


# GMM

gmm = GaussianMixture(n_components=3, covariance_type='full')
gmm.fit(X, y)
