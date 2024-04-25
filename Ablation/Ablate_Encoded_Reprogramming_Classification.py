# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 23:39:09 2024

@author: Allen Zhang
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import h5py
import tensorflow as tf
import os
import copy
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
drop_out_list = [0.25, 0.5]
num_layer_list = [2, 4]
model_type_list = ["LSTM", "GRU"]

LOG_DATA = [True]

SCALE_DATA = [False]

model_hyperparameters_list = [[500, 100, 75, 0.3, 0.2]]
# model_hyperparameters_list = [[4000, 2000, 1000, 0.3, 0.5], [3000, 1000, 400, 0.3, 0.2]]

cols = ['Day6', 'Day9', 'Day12', 'Day15', 'Day21', 'Day28']

RESULT_LIST = []
RESULT_COLUMNS = ['Num_Genes', 'Num_Features', 'Classification_Model_Type', 
                  'Num_Layers', 'Classification_Dropout', 'Accuracy', 
                  'Categorical_Crossentropy']

def myfunc(row):
    return row[1] == 0

def swap_days(col_inds, X_test, y_test):
    bool_arr_test = np.array([myfunc(row) for row in y_test])

    X_test_failed = X_test[bool_arr_test, :, :]
    y_test_failed = y_test[bool_arr_test, :]
    X_test_reprogrammed = X_test[~bool_arr_test, :, :]
    y_test_reprogrammed = y_test[~bool_arr_test]
    
    for i in col_inds: 
        tmp = copy.deepcopy(X_test_failed[:, i, :])
        X_test_failed[:, i, :] = X_test_reprogrammed[:, i, :]
        X_test_reprogrammed[:, i, :] = tmp
        
    return np.concatenate([X_test_failed, X_test_reprogrammed], axis=0), np.concatenate([y_test_failed, y_test_reprogrammed], axis=0)
    
    

for model_hyperparameters in model_hyperparameters_list: 
    dim1 = model_hyperparameters[0]
    dim2 = model_hyperparameters[1]
    dim3 = model_hyperparameters[2]
    alpha = model_hyperparameters[3]
    dropout = model_hyperparameters[4]
    for log in LOG_DATA: 
        for scale in SCALE_DATA: 
            folder = "Dimension_Reduction/Reprogramming/{7}_Genes_Logged_{8}_Scaled_{9}_Data/E1_{0}_E2_{1}_BN_{2}_D1_{3}_D2_{4}_alpha_{5}_dropout_{6}/".format(dim1, dim2, dim3, dim2, dim1, alpha, dropout, n_top_genes, log, scale)
            

            with h5py.File(folder+'{0}_Genes_Data_Encoded_Random_Time_Series_With_Class_Two_Classes.h5'.format(n_top_genes), 'r') as f:
                # Print all root level object names (aka keys) 
                # these can be group or dataset names 
                print("Keys: %s" % f.keys())
                
                X_test = list(f['X_test'])
                y_test = list(f['y_test'])

            X_test = np.array(X_test)
            y_test = np.array(y_test)
            
            print(X_test.shape)
            print(y_test.shape)
            
            TIMESTEPS = X_test.shape[1]
            FEATURES = X_test.shape[2]
            NUM_CLASSES = y_test.shape[1]
            
            # X_test_ablated, y_test_ablated = swap_days([1, 3, 5], X_test, y_test)
            X_test_ablated, y_test_ablated = X_test, y_test
            
            print(X_test_ablated.shape)
            print(y_test_ablated.shape)
            X_test_ablated[:, 5, :] = 0.0

            for DROP_OUT in drop_out_list: 
                for MODEL_TYPE in model_type_list: 
                    for NUM_LAYER in num_layer_list: 
                        folder = "Classification_On_Encoded_Reprogramming_Data/{4}_Encoded_Genes_Two_Classes_Without_28/Features_{0}_Dropout_{1}_NumLayer_{2}_ModelType_{3}/".format(FEATURES, DROP_OUT, NUM_LAYER, MODEL_TYPE, n_top_genes)
                        filepath = "model"
                        # os.makedirs(folder)
                        
                        model = load_model(folder+filepath)
                        y_test_ablated_hat = model.predict(X_test_ablated)
                        
                        
                        y_test_ablated_df = pd.DataFrame(y_test_ablated)
                        y_test_ablated_hat_df = pd.DataFrame(y_test_ablated_hat)
                        spcorr_test = y_test_ablated_hat_df.corrwith(y_test_ablated_df, method='spearman', axis=1)
                        pcorr_test = y_test_ablated_hat_df.corrwith(y_test_ablated_df, method='pearson', axis=1)
                        spcorr_test.to_csv(folder+"spearman_correlation_y_test_ablated_Random.csv")
                        pcorr_test.to_csv(folder+"pearson_correlation_y_test_ablated_Random.csv")
                        y_test_ablated_df.to_csv(folder+"y_test_ablated_hat_predicted_Random.csv")
                        
                        avg_pcorr_test = np.mean(pcorr_test)
                        avg_spcorr_test = np.mean(spcorr_test)
                        plt.figure(figsize=(10, 8))
                        plt.ylim((0,10))
                        plt.hist(spcorr_test, bins=20, range=(0, 1), density=True)
                        plt.title("Average Spearman Correlation = {0}".format(round(avg_spcorr_test, 4)))
                        plt.show()
                        plt.savefig(folder+"spearman_correlation_test_data_ablated_hist_Random.png")
                        plt.figure(figsize=(10, 8))
                        plt.ylim((0,10))
                        plt.hist(pcorr_test, bins=20, range=(0, 1), density=True)
                        plt.title("Average Pearson Correlation = {0}".format(round(avg_pcorr_test, 4)))
                        plt.show()
                        plt.savefig(folder+"pearson_correlation_test_data_ablated_hist_Random.png")
                        
                        y_test_ablated = y_test_ablated_df.to_numpy()
                        y_test_ablated_hat = y_test_ablated_hat_df.to_numpy()
                        
                        y_test_ablated_classes = np.argmax(y_test_ablated, axis=1)
                        y_test_ablated_hat_classes = np.argmax(y_test_ablated_hat, axis=1)
                        
                        cf_mat = confusion_matrix(y_test_ablated_classes, y_test_ablated_hat_classes)
                        acc_score = accuracy_score(y_test_ablated_classes, y_test_ablated_hat_classes)
                        test_loss = np.average(categorical_crossentropy(y_test_ablated, y_test_ablated_hat))
                        
                        with open(folder+'accuracy_ablated_Random.txt', 'a') as f:
                            with redirect_stdout(f):
                                print(f'accuracy score: {acc_score}')
                                f.close()
                        
                        with open(folder+'confusion_matrix_ablated_Random.txt', 'a') as f:
                            with redirect_stdout(f):        
                                print('Confusion matrix')
                                print(cf_mat)
                                f.close()
                        
                        RESULT_LIST.append([n_top_genes, FEATURES, MODEL_TYPE, 
                                            NUM_LAYER, DROP_OUT, acc_score, test_loss])
                        
RESULT_DF = pd.DataFrame(RESULT_LIST, columns=RESULT_COLUMNS)
RESULT_DF.to_csv("Classification_Reprogramming_Two_Classes_{0}_Genes_Ablated_Without_28_Random.csv".format(n_top_genes))

