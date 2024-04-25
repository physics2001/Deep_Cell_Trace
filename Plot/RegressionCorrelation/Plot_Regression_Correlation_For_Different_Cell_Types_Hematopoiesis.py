# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 22:02:51 2023

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

LOG_DATA = [True, False]

SCALE_DATA = [True, False]

model_hyperparameters_list = [[500, 100, 75, 0.3, 0.2], 
                              [500, 200, 50, 0.1, 0.3]]

RESULT_LIST = []
RESULT_COLUMNS = ['Num_genes', 'Reg_Model_Type', 'Num_Layers_BRNN', 
                  'Reg_Dropout', 'Reg_P_Corr', 'Reg_Sp_Corr', 'mse']

storage = "Plots/Regression_Correlation_Results_Hematopoiesis/"

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
    plt.savefig(storage+"Hematopoiesis_Day6_Spearman_Correlation_Hist_{0}.png".format(name))
    plt.savefig(storage+"pdfs/Hematopoiesis_Day6_Spearman_Correlation_Hist_{0}.pdf".format(name))
    plt.clf()
    
    plt.figure(figsize=(10, 8))
    plt.ylim((0,10))
    plt.hist(pcorr_test, bins=20, range=(0, 1), density=True)
    plt.title("Average Pearson Correlation = {0}".format(round(avg_pcorr_test, 4)))
    plt.savefig(storage+"Hematopoiesis_Day6_Pearson_Correlation_Hist_{0}.png".format(name))
    plt.savefig(storage+"pdfs/Hematopoiesis_Day6_Pearson_Correlation_Hist_{0}.pdf".format(name))
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
                
                with h5py.File('{0}_Genes_Logged_Time_Series_With_Monocyte_Neutrophil_Two_Classes.h5'.format(n_top_genes), 'r') as f:
                    print("Keys: %s" % f.keys())
                    
                    X_test = list(f['X_test'])
                    test_classes = list(f['y_test'])
                
                X_test = np.array(X_test)
                test_classes = np.array(test_classes)
                
                y_test = X_test[:, 2, : ].copy()
                X_test = X_test[:, 0:2, : ]
                
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
                            reg_folder = "Regression_On_Not_Encoded_Data_Neu_Mon_Two_Classes/{4}_Genes_Two_Classes_Day6/Features_{0}_Dropout_{1}_NumLayer{2}_{3}/".format(FEATURES, DROP_OUT, NUM_LAYER, MODEL_TYPE, n_top_genes)
                            filepath = "model"
                            
                            model = load_model(reg_folder + filepath)
                            
                            y_test_hat = model.predict(X_test)
                            
                            neu_ind = np.argwhere(test_classes == 1).flatten()
                            mon_ind = np.argwhere(test_classes == 0).flatten()
                            print(neu_ind.shape)
                            print(mon_ind.shape)
                            y_test_neu = y_test[neu_ind, :]
                            y_test_hat_neu = y_test_hat[neu_ind, :]
                            y_test_mon = y_test[mon_ind]
                            y_test_hat_mon = y_test_hat[mon_ind]
                            
                            avg_pcorr_test_neu, avg_spcorr_test_neu = Draw_Correlation_Graph(y_test_neu, y_test_hat_neu, storage, "neu")
                            avg_pcorr_test_mon, avg_spcorr_test_mon = Draw_Correlation_Graph(y_test_mon, y_test_hat_mon, storage, "mon")
                            avg_pcorr_test_all, avg_spcorr_test_all = Draw_Correlation_Graph(y_test, y_test_hat, storage, "all")
                            
                            d = {"Cell Type": ["Neutrophil", "Monocyte", "both"], 
                                 "Average Pearson Correlation": [avg_pcorr_test_neu, 
                                                                 avg_pcorr_test_mon, 
                                                                 avg_pcorr_test_all]}
                            df = pd.DataFrame(data = d)
                            
                            fig, ax = plt.subplots(figsize=(10, 8))
                            ax = sns.barplot(data=df, x="Cell Type", y="Average Pearson Correlation")
                            ax.bar_label(ax.containers[0])
                            plt.savefig(storage+"Hematopoiesis_Day6_Average_Pearson_Correlation_Hist_Comparison.png")
                            plt.savefig(storage+"pdfs/Hematopoiesis_Day6_Average_Pearson_Correlation_Hist_Comparison.pdf")

                            