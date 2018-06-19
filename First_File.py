# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pb
#import data
dataset = pb.read_csv('Data.csv')
#now make matrix
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values 
#doingwork for missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = "NaN", strategy = "mean" , axis = 0)
imputer = imputer.fit(x[:,1:3])
x[:,1:3] = imputer.transform(x[:,1:3])
#catagrise the data 
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_x = LabelEncoder()

x[:,0] = labelencoder_x.fit_transform(x[:,0])
onehotencoder_x = OneHotEncoder(categorical_features = [0])
x = onehotencoder_x.fit_transform(x).toarray()
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
#spiliting data
from sklearn.cross_validation import train_test_split
x_train, x_test, y_test, y_train = train_test_split(x,y,test_size = 0.2, random_state = 0)  
#feature scaling
from sklearn.preprocessing import StandardScaler  
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)