# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 01:13:04 2022

@author: zhang
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import h5py
import tensorflow as tf
import os
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, TimeDistributed, Layer, Bidirectional, GRU, Activation, Masking
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.activations import relu, softmax
from contextlib import redirect_stdout
from keras.losses import mean_squared_error
import time
from keras.losses import mean_squared_error, categorical_crossentropy
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve

print(tf.config.list_physical_devices('GPU'))

n_top_genes = 1000
drop_out_list = [0.3]
num_layer_list = [1] #, 2
model_type_list = ["LSTM"] # , "GRU"

LOG_DATA = [True]

SCALE_DATA = [False]

model_hyperparameters_list = [[500, 100, 75, 0.3, 0.2]]
# model_hyperparameters_list = [[4000, 2000, 1000, 0.3, 0.5], [3000, 1000, 400, 0.3, 0.2]]

cols = ['Day6', 'Day9', 'Day12', 'Day15', 'Day21', 'Day28']

RESULT_LIST = []
RESULT_COLUMNS = np.loadtxt('Reprogramming_{0}_Genes_Logged_True_Scaled_False_Column_Names.txt'.format(n_top_genes), dtype=str)

with h5py.File('{0}_Genes_Reprogramming_Two_Classes_Data_Time_Series_With_Class.h5'.format(n_top_genes), 'a') as f: 
    print("Keys: %s" % f.keys())
    
    # X_train = list(f['X_train'])
    # X_val = list(f['X_val'])
    X_test = list(f['X_test'])
    # y_train = list(f['y_train'])
    # y_val = list(f['y_val'])
    y_test = list(f['y_test'])

# X_train = np.array(X_train)
# X_val = np.array(X_val)
X_test = np.array(X_test)
# y_train = np.array(y_train)
# y_val = np.array(y_val)
y_test = np.array(y_test)

# X_train_mod = X_train.copy()
# X_val_mod = X_val.copy()
X_test_mod = X_test.copy()

# X_train_mod[:, 2:, :] = 0.0
# X_val_mod[:, 2:, :] = 0.0
X_test_mod[:, 2:, :] = 0.0

# print(X_train.shape)
# print(X_val.shape)
print(X_test.shape)
# print(y_train.shape)
# print(y_val.shape)
print(y_test.shape)

TIMESTEPS = X_test.shape[1]
FEATURES = X_test.shape[2]
NUM_CLASSES = y_test.shape[1]

for DROP_OUT in drop_out_list: 
    for MODEL_TYPE in model_type_list: 
        for NUM_LAYER in num_layer_list: 
            folder = "Classification_On_Encoded_Reprogramming_Data/{4}_Genes_Not_Encoded_Two_Classes/Features_{0}_Dropout_{1}_NumLayer_{2}_ModelType_{3}/".format(FEATURES, DROP_OUT, NUM_LAYER, MODEL_TYPE, n_top_genes)
            filepath = "model"
            
            model = load_model(folder+filepath)
            # y_train_hat = model.predict(X_train)
            # y_val_hat = model.predict(X_val)
            y_test_hat = model.predict(X_test_mod)
            y_test_hat = np.array(y_test_hat)
            y_test_hat_mods = []
            
            for i in range(n_top_genes): 
                X_test_mod_i = X_test_mod.copy()
                X_test_mod_i[:, :, i] = 0.0
                y_test_hat_mod_i = model.predict(X_test_mod_i)
                y_test_hat_mod_i = np.array(y_test_hat_mod_i)
                delta_p = y_test_hat_mod_i[:, 1] - y_test_hat[:, 1]
                RESULT_LIST.append(delta_p)
            
            RESULT_LIST = np.array(RESULT_LIST)
            RESULT_LIST = np.swapaxes(RESULT_LIST, 0, 1)
            
            RESULT_DF = pd.DataFrame(RESULT_LIST, columns=RESULT_COLUMNS)
            RESULT_DF.to_csv("Perturb_Reprogramming_{0}_Genes_Not_Encoded_Change_In_Reprog_Result.csv".format(n_top_genes))
