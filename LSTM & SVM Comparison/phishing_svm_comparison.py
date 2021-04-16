# -*- coding: utf-8 -*-
"""phishing_svm.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/10ZunrQeKfnNp1lQmNru1rWw1oI47ePXe
"""

# -*- coding: utf-8 -*-

#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC

import pickle
import csv
#import features_extraction
#from sklearn.externals import joblib

#importing the dataset
f=[]
label=[]
with open ("/content/all-data.csv", 'r' ,newline="") as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for row in reader:
        f.append(row[2:-1])
        if row[-1] == '1':
            label.append(1)
        else:
            label.append(0)
            
for i in range(len(f)):
    for j in range(22):
        if f[i][j] =='1':
            f[i][j] = 1
        elif f[i][j] =='0':
            f[i][j] = 0
        else:
            f[i][j] = -1

x_train = np.array(f).reshape(len(f),22).astype(float)
y_train= np.array(label).reshape(len(label),1).astype(float)
print( x_train.shape,y_train.shape)
#applying grid search to find best performing parameters 
from sklearn.model_selection import GridSearchCV
parameters = [{'C':[1, 10, 100, 1000], 'gamma': [ 0.1, 0.2,0.3, 0.5]}]
grid_search = GridSearchCV(SVC(kernel='rbf' ),  parameters,cv =4, n_jobs= -1)

#fitting kernel SVM  with best parameters calculated 
#x= np.array(f).reshape(len(f),22).astype(float)


x_train1 = x_train[1458:]
y_train1 = y_train[1458:]
print( x_train1.shape,y_train1.shape)
classifier = SVC(C=1, kernel = 'rbf', gamma = 0.1 , random_state = 0)
classifier.fit(x_train1, y_train1.ravel())
grid_search.fit(x_train1, y_train1.ravel())

#printing best parameters 
print("Best Accurancy =" +str( grid_search.best_score_))
print("best parameters =" + str(grid_search.best_params_)) 
score_train = classifier.score(x_train1, y_train1.ravel())
print("score first split", score_train)

x_train1 = x_train[0:729],x_train[1458:]
x_train1 = np.array(x_train1).reshape(1458,22).astype(float)
print(x_train1.shape)
y_train1 = y_train[0:729] , y_train[1458:]
y_train1 = np.array(y_train1).reshape(1458,1).astype(float)

x_train1 = x_train[0:729],x_train[1458:]
x_train1 = np.array(x_train1).reshape(1458,22).astype(float)
y_train1 = y_train[0:729] , y_train[1458:]
y_train1 = np.array(y_train1).reshape(1458,1).astype(float)
print( x_train1.shape,y_train1.shape)
grid_search.fit(x_train1, y_train1.ravel())
classifier = SVC(C=10, kernel = 'rbf', gamma = 0.1 , random_state = 0)
#printing best parameters 
print("Best Accurancy =" +str( grid_search.best_score_))
print("best parameters =" + str(grid_search.best_params_)) 
classifier.fit(x_train1, y_train1.ravel())
score_train = classifier.score(x_train1, y_train1.ravel())
print("score second split", score_train)

x_train1 = x_train
y_train1 = y_train
print( x_train1.shape,y_train1.shape)
grid_search.fit(x_train1, y_train1.ravel())
classifier = SVC(C=100, kernel = 'rbf', gamma = 0.1 , random_state = 0)
#printing best parameters 
classifier.fit(x_train1, y_train1.ravel())
print("Best Accurancy =" +str( grid_search.best_score_))
print("best parameters =" + str(grid_search.best_params_)) 
score_train = classifier.score(x_train1, y_train1.ravel())
print("score third split", score_train)

import matplotlib.pyplot as plt
numbers = [729,1458,2187]

svm_acc=[ 0.6982167352537723,0.7283950617283951 ,0.7343392775491541]
lstm_acc = [0.8642 ,0.9259, 0.9291]
plt.plot(numbers, svm_acc, 'b', label='SVM Accuracy')
plt.plot(numbers, lstm_acc, 'r', label='LSTM accuracy')
plt.scatter(numbers,svm_acc, s=100)
plt.scatter(numbers,lstm_acc, s=100)
plt.xlabel("number of URLs")
plt.ylabel("Training Accuracy")
plt.legend()
plt.title('Training Accuracy Comparison')

plt.show()