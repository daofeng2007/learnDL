# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 22:24:22 2016

@author: Tao Liu
"""
from sklearn import svm
from numpy import genfromtxt
import numpy as np
from sklearn.metrics import accuracy_score
testD = genfromtxt('D:\\Study\\eaglabay\\deepLData\\testD.csv', delimiter=',')
trainD = genfromtxt('D:\\Study\\eaglabay\\deepLData\\trainD.csv', delimiter=',')
validD = genfromtxt('D:\\Study\\eaglabay\\deepLData\\validD.csv', delimiter=',')
traintX=trainD[:,0:3072]
traintY=trainD[:,3072]
testX=testD[:,0:3072]
testY=testD[:,3072]
clf=svm.SVC()
clf.fit(traintX,traintY)
result=clf.predict(testX)
accuracyResult=accuracy_score(testY,result)
