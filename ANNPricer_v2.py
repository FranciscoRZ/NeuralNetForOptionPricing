#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 13:47:02 2018

@author: Cisco
"""

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# @@@@@@@@@@@ Option pricing using MLP Regressor from ScikitLearn @@@@@@@@@@@@
# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

import os
os.chdir("/Users/Cisco/Desktop/M1 EIF/S2/Mémoire")

# @@@@@@@@@@@ Importing relevant libraries @@@@@@@@@@@

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# @@@@@@@@@@@@@@@ Importing the data @@@@@@@@@@@@@@@@@

data = pd.read_csv('DataCAC.csv', sep = ';', decimal = ',').dropna(axis = 0, how = 'all')
X = data.iloc[:, :-1].values
y = data.iloc[:, 5].values

# @@@@@@@@@@@@@@@@ Splitting the data @@@@@@@@@@@@@@@@

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
y_test = y_test.reshape((94,1))
y_train = y_train.reshape((372, 1))

# @@@@@@@@@@@@@@@ Standardizing data @@@@@@@@@@@@@@@@@

#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#X_train = sc.fit_transform(X_train)
#X_test = sc.transform(X_test)

# @@@@@@@@@@@@@@@ Normalizing the data @@@@@@@@@@@@@@@

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# @@@@@@@@@@@@@@@@@ Building the model @@@@@@@@@@@@@@@

from sklearn.neural_network import MLPRegressor
pricer = MLPRegressor(hidden_layer_sizes = (25,25,), learning_rate = 'adaptive', alpha = 0.001, activation = 'relu', solver = 'lbfgs')
pricer.fit(X_train, y_train)

# @@@@@@ Prediciting the values of the test set @@@@@@

y_pred = pricer.predict(X_test)

plt.plot(y_test, color = 'red')
plt.plot(y_pred, color = 'blue')
plt.title('MLP Regression for option pricing')
plt.show

pricer.score(X_train, y_train)
pricer.score(X_test, y_test)

# Testing model
Predictors = sc.fit_transform(X)
PredictCall = pricer.predict(Predictors)

plt.clf
plt.plot(PredictCall, color = 'red')
plt.plot(y, color = 'blue')
plt.show


