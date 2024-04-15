# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 22:20:35 2023

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
from keras.activations import relu
from contextlib import redirect_stdout
from keras.losses import mean_squared_error
import time

print(tf.config.list_physical_devices('GPU'))

Predicted_nth_Day = 3

class SelectKthOutput(Layer):
    def __init__(self, k=0):
        super(SelectKthOutput, self).__init__()
        self.k = k

    def call(self, inputs):
        return inputs[:, self.k, :]

# n_top_genes_list = [1000] 
n_top_genes = 1000
drop_out_list = [0.25]
num_layer_list = [2, 4]
model_type_list = ["LSTM", "GRU"]

LOG_DATA = [True]

SCALE_DATA = [False]

model_hyperparameters_list = [[500, 100, 75, 0.3, 0.2]]
# model_hyperparameters_list = [[4000, 2000, 1000, 0.3, 0.5], [3000, 1000, 400, 0.3, 0.2]]

RESULT_LIST = []
RESULT_COLUMNS = ['Num_genes', 'Reg_Model_Type', 'Num_Layers_BRNN', 
                  'Reg_Dropout', 'Reg_P_Corr', 'Reg_Sp_Corr', 'mse']

with h5py.File('{0}_Genes_Reprogramming_Two_Classes_Data_Time_Series_With_Class.h5'.format(n_top_genes), 'a') as f: 
    print("Keys: %s" % f.keys())
    
    X_train = list(f['X_train'])
    X_val = list(f['X_val'])
    X_test = list(f['X_test'])
    y_train = list(f['y_train'])
    y_val = list(f['y_val'])
    y_test = list(f['y_test'])

X_train = np.array(X_train)
X_val = np.array(X_val)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_val = np.array(y_val)
y_test = np.array(y_test)

TIMESTEPS = X_train.shape[1]
FEATURES = X_train.shape[2]

y_train = X_train[:, Predicted_nth_Day, : ].copy()
X_train[:, Predicted_nth_Day, : ] = np.full([X_train.shape[0], FEATURES], 0)
# X_train = X_train[:, [0, 1, 2, 4, 5], : ]

y_val = X_val[:, Predicted_nth_Day, : ].copy()
X_val[:, Predicted_nth_Day, : ] = np.full([X_val.shape[0], FEATURES], 0)
# X_val = X_val[:, [0, 1, 2, 4, 5], : ]

y_test = X_test[:, Predicted_nth_Day, : ].copy()
X_test[:, Predicted_nth_Day, : ] = np.full([X_test.shape[0], FEATURES], 0)
# X_test = X_test[:, [0, 1, 2, 4, 5], : ]

print(X_train.shape)
print(X_val.shape)
print(X_test.shape)
print(y_train.shape)
print(y_val.shape)
print(y_test.shape)

                
for DROP_OUT in drop_out_list: 
    for MODEL_TYPE in model_type_list: 
        for NUM_LAYER in num_layer_list: 
            reg_folder = "Regression_On_Reprogramming_Data/{4}_Genes_Two_Classes_Day15_With_All_Other_Days/Features_{0}_Dropout_{1}_NumLayer{2}_{3}/".format(FEATURES, DROP_OUT, NUM_LAYER, MODEL_TYPE, n_top_genes)
            filepath = "model"
            os.makedirs(reg_folder)
            
            model = Sequential()
            model.add(Masking(mask_value=0.0, input_shape=(TIMESTEPS, FEATURES)))
            for i in range(NUM_LAYER): 
                if MODEL_TYPE == "LSTM": 
                    model.add(Bidirectional(LSTM(FEATURES, return_sequences=True, dropout=DROP_OUT), 
                                            input_shape=(TIMESTEPS, FEATURES), merge_mode='ave'))
                elif MODEL_TYPE == "GRU": 
                    model.add(Bidirectional(GRU(FEATURES, return_sequences=True, dropout=DROP_OUT), 
                                            input_shape=(TIMESTEPS, FEATURES), merge_mode='ave'))
            model.add(SelectKthOutput(Predicted_nth_Day))
            model.add(Dense(FEATURES))
            model.compile(loss='mse', optimizer='adam', metrics=['mse'])
            
            with open(reg_folder+'model_summary.txt', 'w') as f:
                with redirect_stdout(f):
                    model.summary()
            
            model.summary()
            
            es = EarlyStopping(monitor='val_loss', patience=100, min_delta=5e-5)
            
            # store best model in case overtrained
            cp = ModelCheckpoint(filepath=reg_folder+filepath,
                                 save_best_only=True,
                                 verbose=0,
                                 mode="auto")
            start = time.time()
            history = model.fit(X_train, y_train, batch_size=1024, epochs=100000, shuffle=True, 
                      validation_data=(X_val, y_val), 
                      callbacks=[es, cp])
            end = time.time()
            
            with open(reg_folder+'training_time.txt', 'a') as f:
                with redirect_stdout(f):
                    print(end - start)
            
            plt.figure(figsize=(10, 8))
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.savefig(reg_folder + 'history.png')
            
            model = load_model(reg_folder + filepath)
            
            y_train_hat = model.predict(X_train)
            y_val_hat = model.predict(X_val)
            y_test_hat = model.predict(X_test)
            
            y_train_df = pd.DataFrame(y_train)
            y_train_hat_df = pd.DataFrame(y_train_hat)
            spcorr_train = y_train_hat_df.corrwith(y_train_df, axis=1, method='spearman')
            pcorr_train = y_train_hat_df.corrwith(y_train_df, axis=1, method='pearson')
            spcorr_train.to_csv(reg_folder+"spearman_correlation_train_data.csv")
            pcorr_train.to_csv(reg_folder+"pearson_correlation_train_data.csv")
            
            avg_pcorr_train = np.mean(pcorr_train)
            avg_spcorr_train = np.mean(spcorr_train)
            plt.figure(figsize=(10, 8))
            plt.ylim((0,10))
            plt.hist(spcorr_train, bins=20, range=(0, 1), density=True)
            plt.title("Average Spearman Correlation = {0}".format(round(avg_spcorr_train, 4)))
            plt.show()
            plt.savefig(reg_folder+"spearman_correlation_train_data_hist.png")
            plt.figure(figsize=(10, 8))
            plt.ylim((0,10))
            plt.hist(pcorr_train, bins=20, range=(0, 1), density=True)
            plt.title("Average Pearson Correlation = {0}".format(round(avg_pcorr_train, 4)))
            plt.show()
            plt.savefig(reg_folder+"pearson_correlation_train_data_hist.png")
            
            
            y_val_df = pd.DataFrame(y_val)
            y_val_hat_df = pd.DataFrame(y_val_hat)
            spcorr_val = y_val_hat_df.corrwith(y_val_df, method='spearman', axis=1)
            pcorr_val = y_val_hat_df.corrwith(y_val_df, method='pearson', axis=1)
            spcorr_train.to_csv(reg_folder+"spearman_correlation_val_data.csv")
            pcorr_train.to_csv(reg_folder+"pearson_correlation_val_data.csv")
            y_val_hat_df.to_csv(reg_folder+"y_val_predicted.csv")
            
            avg_pcorr_val = np.mean(pcorr_val)
            avg_spcorr_val = np.mean(spcorr_val)
            plt.figure(figsize=(10, 8))
            plt.ylim((0,10))
            plt.hist(spcorr_val, bins=20, range=(0, 1), density=True)
            plt.title("Average Spearman Correlation = {0}".format(round(avg_spcorr_val, 4)))
            plt.show()
            plt.savefig(reg_folder+"spearman_correlation_val_data_hist.png")
            plt.figure(figsize=(10, 8))
            plt.ylim((0,10))
            plt.hist(pcorr_val, bins=20, range=(0, 1), density=True)
            plt.title("Average Pearson Correlation = {0}".format(round(avg_pcorr_val, 4)))
            plt.show()
            plt.savefig(reg_folder+"pearson_correlation_val_data_hist.png")
            
            
            y_test_df = pd.DataFrame(y_test)
            y_test_hat_df = pd.DataFrame(y_test_hat)
            spcorr_test = y_test_hat_df.corrwith(y_test_df, method='spearman', axis=1)
            pcorr_test = y_test_hat_df.corrwith(y_test_df, method='pearson', axis=1)
            spcorr_train.to_csv(reg_folder+"spearman_correlation_test_data.csv")
            pcorr_train.to_csv(reg_folder+"pearson_correlation_test_data.csv")
            y_test_hat_df.to_csv(reg_folder+"y_test_predicted.csv")
            
            avg_pcorr_test = np.mean(pcorr_test)
            avg_spcorr_test = np.mean(spcorr_test)
            plt.figure(figsize=(10, 8))
            plt.ylim((0,10))
            plt.hist(spcorr_test, bins=20, range=(0, 1), density=True)
            plt.title("Average Spearman Correlation = {0}".format(round(avg_spcorr_test, 4)))
            plt.show()
            plt.savefig(reg_folder+"spearman_correlation_test_data_hist.png")
            plt.figure(figsize=(10, 8))
            plt.ylim((0,10))
            plt.hist(pcorr_test, bins=20, range=(0, 1), density=True)
            plt.title("Average Pearson Correlation = {0}".format(round(avg_pcorr_test, 4)))
            plt.show()
            plt.savefig(reg_folder+"pearson_correlation_test_data_hist.png")
            
            MSE = np.mean(mean_squared_error(y_test, y_test_hat))
            with open(reg_folder+'mse.txt', 'w') as FOB:
                with redirect_stdout(FOB):
                    print(MSE)

            result = [n_top_genes, MODEL_TYPE, NUM_LAYER, 
                      DROP_OUT, avg_pcorr_test, avg_spcorr_test, MSE]
    
            RESULT_LIST.append(result)
            
RESULT_DF = pd.DataFrame(RESULT_LIST, columns=RESULT_COLUMNS)
RESULT_DF.to_csv("Regression_On_Reprogramming_Day15_With_All_Other_Days_{0}_Genes_Two_Classes_Result.csv".format(n_top_genes))
