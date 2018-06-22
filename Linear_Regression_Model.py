# -*- coding: utf-8 -*-
"""
Created on Sun Jun  3 15:14:56 2018

@author: MILAN
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pb
#import data
dataset = pb.read_csv('Salary_Data.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values

#spilit data
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 1/3, random_state = 0)

#model simple linear regression
from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()
regressor.fit(x_train,y_train)

#predict
y_pred = regressor.predict(x_test)
 
#plot 
plt.scatter(x_train,y_train, color = 'green')
plt.plot(x_train, regressor.predict(x_train))
plt.scatter(x_test,y_test,color = 'yellow')
plt.plot(x_train, regressor.predict(x_train))