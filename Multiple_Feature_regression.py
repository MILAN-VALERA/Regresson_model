# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 14:42:44 2018

@author: MILAN
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pb

#import data
dataset = pb.read_csv('50_Startups.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values

#label data encode
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
x[:,3] = labelencoder.fit_transform(x[:,3])
onehot = OneHotEncoder(categorical_features = [3])
x = onehot.fit_transform(x).toarray()

#dummy variable trap 
x = x[:,1:]

from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,random_state = 0, test_size = 0.2)

from sklearn.linear_model import LinearRegression
regresor = LinearRegression()
regresor.fit(x_train,y_train)

y_pred = regresor.predict(x_test)

#backward elimation
import statsmodels.formula.api as sp
x = np.append(arr = np.ones((50,1)).astype(int), values = x, axis = 1)
x_opt = x[:,[0,1,2,3,4,5]]
regressor_ols = sp.OLS(endog=y, exog = x_opt).fit()
regressor_ols.summary()

x_opt = x[:,[0,1,3,4,5]]
regressor_ols = sp.OLS(endog=y, exog = x_opt).fit()
regressor_ols.summary()

x_opt = x[:,[0,3,4,5]]
regressor_ols = sp.OLS(endog=y, exog = x_opt).fit()
regressor_ols.summary()

x_opt = x[:,[0,3,5]]
regressor_ols = sp.OLS(endog=y, exog = x_opt).fit()
regressor_ols.summary()