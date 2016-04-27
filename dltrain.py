# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 18:36:03 2016

@author: Tao
"""

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
        FullyConnectedLayer(n_in=20*14*14, n_out=100),
        SoftmaxLayer(n_in=100, n_out=10)], mini_batch_size)
net.SGD(train_d, 5, mini_batch_size, 0.005, 
            valid_d, test_d)
            
a=net.testResult(test_d,50)
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