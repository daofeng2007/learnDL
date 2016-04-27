# -*- coding: utf-8 -*-
"""
Created on Fri Apr 01 13:01:19 2016

@author: Tao Liu
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 18:36:03 2016

@author: Tao
"""
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import mnist_loader
import numpy as np
from numpy import genfromtxt
import network3
from network3 import Network
from network3 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer
testD = genfromtxt('D:\\Study\\eaglabay\\DeelLData2\\testD.csv', delimiter=',')
trainD = genfromtxt('D:\\Study\\eaglabay\\DeelLData2\\trainD.csv', delimiter=',')
validD = genfromtxt('D:\\Study\\eaglabay\\DeelLData2\\validD.csv', delimiter=',')
#testD = genfromtxt('D:\\Study\\eaglabay\\DeelLData2\\testD.csv', delimiter=',')
#trainD = genfromtxt('D:\\Study\\eaglabay\\DeelLData2\\trainD.csv', delimiter=',')
#validD = genfromtxt('D:\\Study\\eaglabay\\DeelLData2\\validD.csv', delimiter=',')
testt=(testD[:,0:3072],testD[:,3072])
traint=(trainD[:,0:3072],trainD[:,3072])
#np.random.shuffle(traint)
validt=(validD[:,0:3072],validD[:,3072])
train_d,valid_d,test_d=network3.load_data_shared2(traint,validt,testt)
#train_d,valid_d,test_d=network3.load_data_shared()
mini_batch_size=50
net = Network([
        ConvPoolLayer(image_shape=(mini_batch_size, 3, 32, 32), 
                      filter_shape=(20, 3, 5, 5), 
                      poolsize=(2, 2)),
        ConvPoolLayer(image_shape=(mini_batch_size, 20, 14, 14), 
                      filter_shape=(30, 20, 5, 5), 
                      poolsize=(2, 2)),
        FullyConnectedLayer(n_in=30*5*5, n_out=200),
        SoftmaxLayer(n_in=200, n_out=10)], mini_batch_size)
net.SGD(train_d,100, mini_batch_size, 0.005, 
            valid_d, test_d)
            
a=net.testResult(test_d,50)
b=np.array([a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7]]).reshape((400,1))
c=b[range(0,373)]
d=testD[:,3072]
d1=d[range(0,373)]
d2=confusion_matrix(d1, c)
accuracy_score(d1,c)
np.savetxt("result.csv",c , delimiter=",")
np.savetxt("true.csv",d1 , delimiter=",")
np.savetxt("confusion.csv",d2 , delimiter=",")
#net = Network([
#        ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28), 
#                      filter_shape=(20, 1, 5, 5), 
#                      poolsize=(2, 2)),
#        ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12), 
#                      filter_shape=(20, 20, 5, 5), 
#                      poolsize=(2, 2)),
#        FullyConnectedLayer(n_in=20*4*4, n_out=100),
#        SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
#net.SGD(train_d, 60, mini_batch_size, 0.1, 
#            valid_d, test_d)               
        
#training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
#import cPickle, gzip, numpy
#
## Load the dataset
#f = gzip.open('mnist.pkl.gz', 'rb')
#train_set, valid_set, test_set = cPickle.load(f)
#f.close()


#f = gzip.open('mnist.pkl.gz', 'rb')
#training_data, validation_data, test_data = cPickle.load(f)
#f.close()