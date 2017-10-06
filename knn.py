import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.cross_validation import train_test_split
import pandas as pd
from sklearn.naive_bayes import GaussianNB




x = pd.read_csv('/home/omar/Documents/python_programs/graduation/datasets/tkn4.csv')
y = pd.read_csv('/home/omar/Documents/python_programs/graduation/datasets/labels4.csv')
score = []
X_train,X_test,Y_train,Y_test = train_test_split(x,y,test_size=0.4,random_state=0)
knn = KNeighborsClassifier(n_neighbors=22)
knn.fit(X_train,Y_train)
z=knn.predict(X_test)
print metrics.accuracy_score(Y_test,z)

# print max(score)