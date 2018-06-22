# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 18:18:59 2018

@author: MILAN
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pb
 
dataset = pb.read_csv('Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2:3].values

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
X = sc_x.fit_transform(X)
y = sc_y.fit_transform(y)

from sklearn.svm import SVR
regressor = SVR()
regressor.fit(X,y)
y_pred = sc_y.inverse_transform(regressor.predict(sc_x.transform(np.array([[6.5]]))))

plt.scatter(X,y,color = 'red')
plt.plot(x,regressor.predict(x),color = 'green')
plt.legend('training data','prediction')
￼