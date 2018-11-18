#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess
import datetime

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
t0 = datetime.datetime.now()
features_train, features_test, labels_train, labels_test = preprocess()
print "data loaded after :", datetime.datetime.now()-t0



#########################################################
### your code goes here ###
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

t1 = datetime.datetime.now()
gnb = GaussianNB()
gnbc = gnb.fit(features_train, labels_train)
print "Training time :", datetime.datetime.now() - t1

t2 = datetime.datetime.now()
pred = gnbc.predict(features_test)

print "Accuracy :", accuracy_score(pred, labels_test),
print "Time for accuracy: ", datetime.datetime.now()-t2

#########################################################


