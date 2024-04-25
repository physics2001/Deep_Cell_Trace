# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 19:15:42 2023

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
import seaborn as sns

print(tf.config.list_physical_devices('GPU'))

n_top_genes_list = [1000] #100, 500, 5000, 2000, 
drop_out_list = [0.3]
num_layer_list = [2, 4]
model_type_list = ["LSTM", "GRU"]

LOG_DATA = [True]

SCALE_DATA = [False]

model_hyperparameters_list = [[500, 100, 75, 0.3, 0.2]]
# model_hyperparameters_list = [[4000, 2000, 1000, 0.3, 0.5], [3000, 1000, 400, 0.3, 0.2]]

RESULT_LIST = []
RESULT_COLUMNS = ['Num_genes', 'Logged', 'Scaled', 'AE_dim1', 'AE_dim2', 'AE_latent', 
                  'Alpha', 'Reg_Model_Type', 'Num_Layers_BRNN', 
                  'Reg_Dropout', 'Reg_P_Corr', 'Reg_Sp_Corr']

storage = "Plots/Regression_Correlation_Results/"

def Draw_Correlation_Graph(y_test, y_test_hat, storage, name): 
    y_test_df = pd.DataFrame(y_test)
    y_test_hat_df = pd.DataFrame(y_test_hat)
    spcorr_test = y_test_hat_df.corrwith(y_test_df, method='spearman', axis=1)
    pcorr_test = y_test_hat_df.corrwith(y_test_df, method='pearson', axis=1)
    spcorr_test.to_csv(storage+"spearman_correlation_test_data_{0}.csv".format(name))
    pcorr_test.to_csv(storage+"pearson_correlation_test_data_{0}.csv".format(name))
    y_test_hat_df.to_csv(storage+"y_test_predicted_{0}.csv".format(name))
    
    avg_pcorr_test = np.mean(pcorr_test)
    avg_spcorr_test = np.mean(spcorr_test)
    plt.figure(figsize=(10, 8))
    plt.ylim((0,10))
    plt.hist(spcorr_test, bins=20, range=(0, 1), density=True)
    plt.title("Average Spearman Correlation = {0}".format(round(avg_spcorr_test, 4)))
    plt.show()
    plt.savefig(storage+"Reprogramming_Day28_Spearman_Correlation_Hist_{0}.png".format(name))
    plt.savefig(storage+"pdfs/Reprogramming_Day28_Spearman_Correlation_Hist_{0}.pdf".format(name))
    plt.clf()
    
    plt.figure(figsize=(10, 8))
    plt.ylim((0,10))
    plt.hist(pcorr_test, bins=20, range=(0, 1), density=True)
    plt.title("Average Pearson Correlation = {0}".format(round(avg_pcorr_test, 4)))
    plt.show()
    plt.savefig(storage+"Reprogramming_Day28_Pearson_Correlation_Hist_{0}.png".format(name))
    plt.savefig(storage+"pdfs/Reprogramming_Day28_Pearson_Correlation_Hist_{0}.pdf".format(name))
    plt.clf()
    
    return (avg_pcorr_test, avg_spcorr_test)

for model_hyperparameters in model_hyperparameters_list: 
    dim1 = model_hyperparameters[0]
    dim2 = model_hyperparameters[1]
    dim3 = model_hyperparameters[2]
    alpha = model_hyperparameters[3]
    dropout = model_hyperparameters[4]
    for log in LOG_DATA: 
        for scale in SCALE_DATA: 
            for n_top_genes in n_top_genes_list: 
                folder = "Dimension_Reduction/Reprogramming/{7}_Genes_Logged_{8}_Scaled_{9}_Data/E1_{0}_E2_{1}_BN_{2}_D1_{3}_D2_{4}_alpha_{5}_dropout_{6}/".format(dim1, dim2, dim3, dim2, dim1, alpha, dropout, n_top_genes, log, scale)
                
                with h5py.File(folder+'{0}_Genes_Data_Encoded_Time_Series_With_Class_Two_Classes.h5'.format(n_top_genes), 'r') as f:
                    # Print all root level object names (aka keys) 
                    # these can be group or dataset names 
                    print("Keys: %s" % f.keys())
                    
                    X_test = list(f['X_test'])
                    test_classes = list(f['y_test'])
                
                X_test = np.array(X_test)
                test_classes = np.array(test_classes)
                
                y_test = X_test[:, 5, : ].copy()
                X_test = X_test[:, 0:5, : ]
                
                print(X_test.shape)
                print(y_test.shape)
                print(test_classes.shape)
                
                test_classes = test_classes[:,1]
                print(test_classes.shape)
                
                TIMESTEPS = X_test.shape[1]
                FEATURES = X_test.shape[2]
                
                for DROP_OUT in drop_out_list: 
                    for MODEL_TYPE in model_type_list: 
                        for NUM_LAYER in num_layer_list: 
                            reg_folder = "Regression_On_Encoded_Reprogramming_Data/{6}_Genes_Two_Classes_Day28/{6}_Genes_Logged_{4}_Scaled_{5}_Data/Features_{0}_Dropout_{1}_NumLayer{2}_{3}/".format(FEATURES, DROP_OUT, NUM_LAYER, MODEL_TYPE, log, scale, n_top_genes)
                            filepath = "model"
                            
                            model = load_model(reg_folder + filepath)
                            
                            y_test_hat = model.predict(X_test)
                            
                            reprogrammed_ind = np.argwhere(test_classes == 1).flatten()
                            failed_ind = np.argwhere(test_classes == 0).flatten()
                            print(reprogrammed_ind.shape)
                            print(failed_ind.shape)
                            y_test_reprogrammed = y_test[reprogrammed_ind, :]
                            y_test_hat_reprogrammed = y_test_hat[reprogrammed_ind, :]
                            y_test_failed = y_test[failed_ind]
                            y_test_hat_failed = y_test_hat[failed_ind]
                            
                            avg_pcorr_test_reprogrammed, avg_spcorr_test_reprogrammed = Draw_Correlation_Graph(y_test_reprogrammed, y_test_hat_reprogrammed, storage, "reprogrammed")
                            avg_pcorr_test_failed, avg_spcorr_test_failed = Draw_Correlation_Graph(y_test_failed, y_test_hat_failed, storage, "failed")
                            avg_pcorr_test_all, avg_spcorr_test_all = Draw_Correlation_Graph(y_test, y_test_hat, storage, "all")
                            
                            d = {"Cell Type": ["reprogrammed", "failed", "both"], 
                                 "Average Pearson Correlation": [avg_pcorr_test_reprogrammed, 
                                                                 avg_pcorr_test_failed, 
                                                                 avg_pcorr_test_all]}
                            df = pd.DataFrame(data = d)
                            
                            fig, ax = plt.subplots(figsize=(10, 8))
                            ax = sns.barplot(data=df, x="Cell Type", y="Average Pearson Correlation")
                            ax.bar_label(ax.containers[0])
                            plt.savefig(storage+"Reprogramming_Day28_Average_Pearson_Correlation_Hist_Comparison.png")
                            plt.savefig(storage+"pdfs/Reprogramming_Day28_Average_Pearson_Correlation_Hist_Comparison.pdf")

                            