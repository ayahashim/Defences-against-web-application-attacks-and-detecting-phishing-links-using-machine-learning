# -*- coding: utf-8 -*-

#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC
import new_phishing1
import pickle
import csv
#import features_extraction
#from sklearn.externals import joblib

#importing the dataset
dataset = pd.read_csv("Training_Dataset.csv")
dataset = dataset.drop('id', 1) #removing unwanted column
x = dataset.iloc[: , :-1].values
x = x[:, [0, 1, 2, 3, 4, 5, 6, 8, 9, 11, 12, 13, 14, 15, 16, 17,22, 23, 24, 25, 27, 29]]
y = dataset.iloc[:, -1:].values
y= y.ravel()
print( x.shape,y.shape)
#spliting the dataset into training set and test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.25, random_state =0 )

#applying grid search to find best performing parameters 
from sklearn.model_selection import GridSearchCV
#parameters = [{'C':[1, 10, 100, 1000], 'gamma': [ 0.1, 0.2,0.3, 0.5]}]
#grid_search = GridSearchCV(SVC(kernel='rbf' ),  parameters,cv =4, n_jobs= -1)
#grid_search.fit(x_train, y_train.ravel())

#printing best parameters 
#print("Best Accurancy =" +str( grid_search.best_score_))
#print("best parameters =" + str(grid_search.best_params_)) 

#fitting kernel SVM  with best parameters calculated 

classifier = SVC(C=10, kernel = 'rbf', gamma = 0.2 , random_state = 0)
classifier.fit(x_train, y_train.ravel())

#with open('phish_svm.pickle','wb') as f:
#    pickle.dump(classifier, f)
 
#pickle_in = open('phish_svm.pickle','rb')

#classifier = pickle.load(pickle_in)
score_train = classifier.score(x_train, y_train)
print("score", score_train)
print(x_test.shape ,y_test.shape )
score = classifier.score(x_test, y_test)
print("score for test", score)
#predicting the tests set result
y_pred = classifier.predict(x_test)
#confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
test_url="http://convalidainfo.com"
dataset = pd.read_csv("verified_online.csv")
dataset = dataset.drop('phish_id', 1) #removing unwanted column
dataset = dataset.drop(['phish_detail_url','submission_time','verified','verification_time','online','target'],axis =1)
#print(dataset.head)
phish = 0
test ="http://sampeeeebasahhhhhh.000webhostapp.com/login.php"
#ftest = new_phishing1.main(test)
#print(ftest)
x = dataset.iloc[: , :].values
f=[]
label=[]
with open ("C:/Users/Aya/Desktop/features1.csv", 'r' ,newline="") as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        f.append(row[1:-1])
        label.append(row[-1])
for i in range(len(f)):
    for j in range(22):
        if f[i][j] =='1':
            f[i][j] = 1
        elif f[i][j] =='0':
            f[i][j] = 0
        else:
            f[i][j] = -1
            
f = np.array(f).reshape(len(f),22).astype(float)
f= f[0:1000]
label= np.array(label).reshape(len(label),1).astype(float)
label = label[0:1000]
label= label.ravel()
print(f.shape, label.shape)
print (f[0,0])
score = classifier.score(f, label)
print(score)


