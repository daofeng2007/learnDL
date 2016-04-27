# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 18:56:38 2016

@author: Tao
"""

import numpy as np
a=np.array([[2,3,6,1],[4,15,9,0]])
b=np.argmax(a,axis=0)
#a = np.arange(3 * 4 * 5).reshape(3, 4, 5).astype('float32')
#b = np.arange(3 * 5).reshape(5, 3).astype('float32')
#
#result = a.dot(b)