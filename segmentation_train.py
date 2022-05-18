# -*- coding: utf-8 -*-
"""
Created on Wed May 18 11:36:55 2022

@author: DELL
"""
# import all libraries
import os 
import pandas as pd
import numpy as np
import pickle
import datetime

from segmentation_class_module import ExploratoryDataAnalysis, ModelCreation, ModelEvaluate 
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split
#%% data loading   
PATH = os.path.join(os.getcwd(),'dataset','train.csv') # dataset path
PATH_LOG = os.path.join(os.getcwd(),'log') #log path
MODEL_SAVE_PATH = os.path.join(os.getcwd(), 'model_saved','model.h5') # model path

df = pd.read_csv(PATH)

#%% Data inspection
df.head()
df.info()
df.describe()
df.columns

# checking sum of the nan value
df.isna().sum()

# visualize the nan value
import missingno as msno
msno.matrix(df)

# split the data 
x = df.drop('Segmentation', axis = 1) # features
y = pd.DataFrame(df['Segmentation']) # target

features_names= x.columns # assign column names


#%% eda 
# data cleaning

eda = ExploratoryDataAnalysis() # calling eda class function


x = eda.fill_null(x)

# data preprocessing
# model cannot read categorical data
# categorical data need to convert into numerical
x_train = eda.label_encoder(pd.DataFrame(x,columns = features_names)) 

#feature scaling 
x_scaled = eda.feature_scalling(x_train)

# convert label data into one hot encoder
y_train = eda.one_hot_encoder(y)

#train test split
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_train,
                                                    test_size = 0.3,
                                                    random_state = 123)

#%% model creation

nb_inputs = x_train.shape[1] # assign number of features inputs 
mc = ModelCreation() # calling ModelCreation class for modeul building

# model creation 
model = mc.model_create(input_shape= 10, nodes = 64) 
plot_model(model) # visualize architecture of the model
#%% model compile?evaluat

# log folder
# assign a name for the train and validation loss, accuracy
log_dir = os.path.join(PATH_LOG,
                       datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))

# tensorboard callbacks
tensorboard_callback = TensorBoard(log_dir= log_dir)

# early stopping
# avoid overfit
early_stopping = EarlyStopping(monitor ='loss',patience = 15)

#%% model compile
model.compile(optimizer = 'adam',
              loss =  'categorical_crossentropy',
              metrics = 'acc')


#%% fit the model
model.fit(x_train, y_train, epochs = 100,
                 validation_data = (x_test, y_test),
                 batch_size = 132,
                 callbacks= [early_stopping, tensorboard_callback])

# import matplotlib.pyplot as plt
# plt.figure()
# plt.plot(hist.history['acc'])
# plt.plot(hist.history['val_acc'])
# plt.xlabel('epochs')
# plt.ylabel('accuracy')
# plt.legend(['acc', 'val_acc'])
# plt.show()

# plt.figure()
# plt.plot(hist.history['loss'])
# plt.plot(hist.history['val_loss'])
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.legend(['loss', 'val_loss'])
# plt.show()
#%% model evaluate

y_pred = model.predict(x_test)
y_true = y_test

y_pred = np.argmax(y_pred, axis = 1)
y_true = np.argmax(y_true, axis = -1)

me = ModelEvaluate()
me.report(y_pred, y_true)

#%% saved model 
model.save(MODEL_SAVE_PATH)

