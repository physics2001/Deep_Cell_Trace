# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 00:09:17 2022

@author: zhang
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import h5py
import tensorflow as tf
import os
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, TimeDistributed, Layer, Bidirectional, GRU, Activation
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.activations import relu, softmax
from contextlib import redirect_stdout
from keras.losses import mean_squared_error
from keras.metrics import SparseCategoricalAccuracy
import time
from keras.losses import mean_squared_error, categorical_crossentropy
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve


print(tf.config.list_physical_devices('GPU'))

n_top_genes_list = [1000] #100, 500, 5000, 2000, 
drop_out_list = [0.25, 0.5]
num_layer_list = [1, 2, 4]
model_type_list = ["LSTM", "GRU"]

RESULT_LIST = []
RESULT_COLUMNS = ['Num_Genes', 'Num_Features', 'Classification_Model_Type', 
                  'Num_Layers', 'Classification_Dropout', 'Accuracy', 'Categorical_Crossentropy']


for n_top_genes in n_top_genes_list: 

    with h5py.File('{0}_Encoded_Genes_Scaled_Random_Time_Series_With_Monocyte_Neutrophil_with_orig_data.h5'.format(n_top_genes), 'r') as f:
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
    TRAIN_TEST_SPLIT = 0.2
    VAL_TEST_SPLIT = 0.5
    NUM_CLASSES = y_test.shape[1]
    
    X_test[:, 2, :] = 0.0
    
    for DROP_OUT in drop_out_list: 
        for MODEL_TYPE in model_type_list: 
            for NUM_LAYER in num_layer_list: 
                folder = "Classification_On_Encoded_Data_Mon_Neu_Two_Classes_Without_Day6/Features_{0}_Dropout_{1}_NumLayer_{2}_ModelType_{3}/".format(FEATURES, DROP_OUT, NUM_LAYER, MODEL_TYPE)
                filepath = "model"
                # os.makedirs(folder)
                
                model = load_model(folder+filepath)
                y_test_hat = model.predict(X_test)
                
                y_test_df = pd.DataFrame(y_test)
                y_test_hat_df = pd.DataFrame(y_test_hat)
                spcorr_test = y_test_hat_df.corrwith(y_test_df, method='spearman', axis=1)
                pcorr_test = y_test_hat_df.corrwith(y_test_df, method='pearson', axis=1)
                spcorr_test.to_csv(folder+"spearman_correlation_test_data_ablated.csv")
                pcorr_test.to_csv(folder+"pearson_correlation_test_data_ablated.csv")
                y_test_hat_df.to_csv(folder+"y_test_predicted_ablated.csv")
                
                avg_pcorr_test = np.mean(pcorr_test)
                avg_spcorr_test = np.mean(spcorr_test)
                plt.figure(figsize=(10, 8))
                plt.ylim((0,10))
                plt.hist(spcorr_test, bins=20, range=(0, 1), density=True)
                plt.title("Average Spearman Correlation = {0}".format(round(avg_spcorr_test, 4)))
                plt.show()
                plt.savefig(folder+"spearman_correlation_test_data_hist_ablated.png")
                plt.figure(figsize=(10, 8))
                plt.ylim((0,10))
                plt.hist(pcorr_test, bins=20, range=(0, 1), density=True)
                plt.title("Average Pearson Correlation = {0}".format(round(avg_pcorr_test, 4)))
                plt.show()
                plt.savefig(folder+"pearson_correlation_test_data_hist_ablated.png")
                
                y_test = y_test_df.to_numpy()
                y_test_hat = y_test_hat_df.to_numpy()
                
                y_test_classes = np.argmax(y_test, axis=1)
                y_test_hat_classes = np.argmax(y_test_hat, axis=1)
                
                cf_mat = confusion_matrix(y_test_classes, y_test_hat_classes)
                acc_score = accuracy_score(y_test_classes, y_test_hat_classes)
                test_loss = np.average(categorical_crossentropy(y_test, y_test_hat))
                
                with open(folder+'accuracy_ablated.txt', 'a') as f:
                    with redirect_stdout(f):
                        print(f'accuracy score: {acc_score}')
                        f.close()
                
                with open(folder+'confusion_matrix_ablated.txt', 'a') as f:
                    with redirect_stdout(f):        
                        print('Confusion matrix')
                        print(cf_mat)
                        f.close()
                
                RESULT_LIST.append([n_top_genes, FEATURES, MODEL_TYPE, 
                                    NUM_LAYER, DROP_OUT, acc_score, test_loss])
                
                RESULT_DF = pd.DataFrame(RESULT_LIST, columns=RESULT_COLUMNS)
                RESULT_DF.to_csv("Classification_Mon_Neu_Result_Two_Classes_Without_Day6_Random_Ablated.csv")
