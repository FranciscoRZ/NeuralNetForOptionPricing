#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 17:13:23 2018

@author: Cisco
"""

import os
os.chdir("/Users/Cisco/Desktop/M1 EIF/S2/Mémoire/DataCACPaul")

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import data
data = pd.read_csv('DataK5200.csv', sep = ';', decimal = ',').dropna(axis = 0, how = 'all')
data = data.reindex(index = data.index[::-1]).dropna(axis = 1, how = 'all')
X = data.iloc[:, [1,2]].values
#y_call = data.iloc[:, 3].values
y_put = data.iloc[:, 4].values

# Splitting into Training and Test sets
from sklearn.model_selection import train_test_split
#X_train, X_test, y_call_train, y_call_test = train_test_split(X, y_call, test_size = 0.2)
X_train, X_test, y_put_train, y_put_test = train_test_split(X, y_put, test_size = 0.2)

# Feature scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# @@@@@@@ Building the ANN Model @@@@@@@@@@
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initializing the ANN
pricer = Sequential()

# Adding a first hidden layer and the input layer
pricer.add(Dense(output_dim = 3, init = 'uniform', activation = 'relu', input_dim = 2))

# Adding a second input layer
pricer.add(Dense(output_dim = 3, init = 'uniform', activation = 'relu'))

# Adding the output layer
pricer.add(Dense(output_dim = 1, init = 'uniform', activation = 'linear'))

# compiling
pricer.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the ANN to the training set

#pricer.fit(X_train, y_call_train, batch_size = 10, epochs = 100)
pricer.fit(X_train, y_put_train, batch_size = 10, epochs = 100)


# @@@@@@@ Making the predictions and evaluating the model @@@@@@@@
y_pred = pricer.predict(X_test)

plt.clf
#plt.plot(y_call_test, color = 'red')
plt.plot(y_put_test, color = 'red')
plt.plot(y_pred, color = 'blue')

#pricer.evaluate(X_test, y_call_test)
pricer.evaluate(X_test, y_put_test)

# Predicting the entire set 
Predictors = scaler.fit_transform(X)
PredictCall = pricer.predict(Predictors)

plt.clf
plt.plot(PredictCall, color = 'blue', label = 'Predicted put prices')
#plt.plot(y_call, color = 'red', label = 'Market call prices')
plt.plot(y_put, color = 'red', label = 'Market put prices')
plt.legend()
#plt.title('Predicted call prices vs. observed call prices')
plt.title('Predicted call prices vs. observed call prices')
plt.savefig(fname = '/Users/Cisco/Desktop/M1 EIF/S2/Mémoire/Graphiques/Model25200_put.jpeg')

#pricer.evaluate(Predictors, y_call)
pricer.evaluate(Predictors, y_put)

output = pd.DataFrame(PredictCall)
#output.to_csv("/Users/Cisco/Desktop/Mémoire/PredictionCallModel2", sep = ",", index = False)

# Saving results
import csv
csvfile = '/Users/Cisco/Desktop/M1 EIF/S2/Mémoire/Predictions/Model2_put52Predictions'

PredictCall = PredictCall[::-1]

with open(csvfile, 'w') as output:
    writer = csv.writer(output, lineterminator = '\n')
    writer.writerows(PredictCall)


