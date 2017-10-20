# -*- coding: utf-8 -*-
"""
Created on Fri Aug 26 20:45:10 2016

@author: DIP
"""

import numpy as np
from sklearn.decomposition import TruncatedSVD, NMF, PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

vec1 = np.random.randn(1,1000)
vec2 = np.random.randn(1,1000)

X = np.zeros((40,1000))
kerroin = np.zeros((2,20))
kerroin[0,0:15]=1
kerroin[1,15:30]=1

for i in range(0,20):
    X[i,:] = kerroin[0,i]*vec1 + kerroin[1,i]*vec2 + 0.05*np.random.randn(1,1000)

scaler = StandardScaler()
X = scaler.fit_transform(X)
model = TruncatedSVD(n_components = 42)
X_red = model.fit_transform(X)

plt.figure()
plt.plot(X)
