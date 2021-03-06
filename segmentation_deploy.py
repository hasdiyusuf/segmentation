# import library
import pandas as pd
import numpy as np
import pickle
import os
from tensorflow.keras.models import load_model
from segmentation_class_module import ExploratoryDataAnalysis, ModelCreation, ModelEvaluate 

# saved model path
MODEL_SAVE_PATH = os.path.join(os.getcwd(), 'model_saved','model.h5')

# path for the test dataset
PATH = os.path.join(os.getcwd(),'dataset', 'new_customers.csv')
# load dataset
df = pd.read_csv(PATH)
#%% load model 

model = load_model(MODEL_SAVE_PATH)
model.summary()


#%% new customer

# check nan value
df.isna().sum()

# visualize nan value
import missingno as msno
msno.matrix(df)


eda = ExploratoryDataAnalysis()
df = eda.fill_null(df)
df = eda.label_encoder(pd.DataFrame(df))


df = eda.feature_scalling(df)

#%%

outcome = model.predict(df)

segmentation_dict = {0:'A', 1 :'B', 2:'C', 3:'D'}

predicted = []
for index, segment in enumerate(outcome):
    predicted.append(segmentation_dict[np.argmax(segment)])
    