# -*- coding: utf-8 -*-
"""
Created on Wed May 18 09:43:34 2022

@author: DELL
"""

import os
import pandas as pd
import numpy as np
import pickle

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score 

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Input

from tensorflow.keras.utils import plot_model

#%% classes

class ExploratoryDataAnalysis():
    def _init_(self):
        pass
    
    def fill_null(self, data):
         imputer = SimpleImputer(strategy = 'most_frequent')
         data =  imputer.fit_transform(data)
         return data
     
    def label_encoder(self,data):
        obj_list = data.select_dtypes(include = 'object').columns
        le = LabelEncoder()
        
        for feat in obj_list:
            data[feat] = le.fit_transform(data[feat].astype(str))
        return data
    
    def one_hot_encoder(self,label):
        ohe = OneHotEncoder(sparse = False)
        y_train = ohe.fit_transform(label)
        return y_train
    
    def feature_scalling(self, data):
        mms = MinMaxScaler()
        x_scaled = mms.fit_transform(data)
        return x_scaled

class ModelCreation():
    def model_create(self, input_shape,output=4, nodes=300, activation = 'relu', dropout= (0.2)):
        model = Sequential()
        
        model.add(Input(shape = (input_shape)))
        model.add(Dense(nodes, activation))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))
        model.add(Dense(output, activation = 'softmax'))
        
        model.summary()
        plot_model(model)
        return model

class ModelEvaluate():
    def report(self, y_true, y_pred):
        print(accuracy_score(y_true, y_pred))
        print(confusion_matrix(y_true,y_pred))
        print(classification_report(y_true, y_pred))     
        

