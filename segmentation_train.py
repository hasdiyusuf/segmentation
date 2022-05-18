# -*- coding: utf-8 -*-
"""
Created on Wed May 18 11:36:55 2022

@author: DELL
"""
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
PATH = os.path.join(os.getcwd(),'dataset','train.csv')
PATH_LOG = os.path.join(os.getcwd(),'log')
MODEL_SAVE_PATH = os.path.join(os.getcwd(), 'model_saved','model.h5')

df = pd.read_csv(PATH)


# check nan value
df.isna().sum()

# visualize nan value
import missingno as msno
msno.matrix(df)

# #split the data 
x = df.drop('Segmentation', axis = 1)
y = pd.DataFrame(df['Segmentation'])

features_names= x.columns


#%% eda 
# data cleaning

eda = ExploratoryDataAnalysis()


x = eda.fill_null(x)

# data preprocessing
x_train = eda.label_encoder(pd.DataFrame(x,columns = features_names))
x_scaled = eda.feature_scalling(x_train)

y_train = eda.one_hot_encoder(y)

#train test split
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y_train,
                                                    test_size = 0.3,
                                                    random_state = 123)

#%% model creation

nb_inputs = x_train.shape[1]
mc = ModelCreation()
model = mc.model_create(input_shape= 10, nodes = 64)
plot_model(model)
#%% model compile?evaluat

log_dir = os.path.join(PATH_LOG,
                       datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))

tensorboard_callback = TensorBoard(log_dir= log_dir)

early_stopping = EarlyStopping(monitor ='loss',patience = 15)

model.compile(optimizer = 'adam',
              loss =  'categorical_crossentropy',
              metrics = 'acc')


hist = model.fit(x_train, y_train, epochs = 100,
                 validation_data = (x_test, y_test),
                 batch_size = 132,
                 callbacks= [early_stopping, tensorboard_callback])

import matplotlib.pyplot as plt
plt.figure()
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend(['acc', 'val_acc'])
plt.show()

plt.figure()
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(['loss', 'val_loss'])
plt.show()
#%% model evaluate

y_pred = model.predict(x_test)
y_true = y_test

y_pred = np.argmax(y_pred, axis = 1)
y_true = np.argmax(y_true, axis = -1)

me = ModelEvaluate()
me.report(y_pred, y_true)

#%% saved model and scaler
model.save(MODEL_SAVE_PATH)

