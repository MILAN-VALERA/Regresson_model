# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 15:16:00 2018

@author: MILAN
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pb

dataset = pb.read_csv('Position_Salaries.csv')
x = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2:3].values

#model evelution
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(random_state= 0,n_estimators= 1000000)
regressor.fit(x,y)
y1 = regressor.predict(6.5)


#plot data
plt.scatter(x,y,marker = '+',color = 'purple')
plt.plot(x,regressor.predict(x),color = 'brown' )
plt.legends()

