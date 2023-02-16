# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 18:15:14 2023

@author: eugen
"""
 
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from joblib import dump


data = load_iris() #['data']

X = data['data']
y = data['target']


train_X, test_X, train_y, test_y = train_test_split(X,y, test_size=0.2, random_state=44)


clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_X,train_y)

y_pred = clf.predict(test_X)
accuracy = accuracy_score(test_y, y_pred)


print(accuracy)

dump(clf, 'DT.joblib')


