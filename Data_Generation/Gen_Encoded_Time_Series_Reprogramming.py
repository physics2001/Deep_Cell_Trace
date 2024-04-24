# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 00:26:01 2022

@author: zhang
"""

import pandas as pd
import numpy as np
import h5py
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import scanpy as sc
import cospar as cs
import joblib

# n_top_genes = 5000
n_top_genes = 1000

LOG_DATA = [True]

SCALE_DATA = [False]

model_hyperparameters_list = [[500, 100, 75, 0.3, 0.2]]
# model_hyperparameters_list = [[4000, 2000, 1000, 0.3, 0.5], [3000, 1000, 400, 0.3, 0.2]]

cols = ['Day6', 'Day9', 'Day12', 'Day15', 'Day21', 'Day28']

for model_hyperparameters in model_hyperparameters_list: 
    dim1 = model_hyperparameters[0]
    dim2 = model_hyperparameters[1]
    dim3 = model_hyperparameters[2]
    alpha = model_hyperparameters[3]
    dropout = model_hyperparameters[4]
    for log in LOG_DATA: 
        for scale in SCALE_DATA: 
            folder = "Dimension_Reduction/Reprogramming/{7}_Genes_Logged_{8}_Scaled_{9}_Data/E1_{0}_E2_{1}_BN_{2}_D1_{3}_D2_{4}_alpha_{5}_dropout_{6}/".format(dim1, dim2, dim3, dim2, dim1, alpha, dropout, n_top_genes, log, scale)
            
            with h5py.File(folder+'{0}_Genes_Data_Encoded.h5'.format(n_top_genes), 'a') as q:
                print("Keys: %s" % q.keys())
                X = list(q[folder+'{0}_Genes_Data_Encoded'.format(n_top_genes)])
                q.close()
            
            X = np.array(X, dtype=np.float32)
            
            df = pd.DataFrame(X)
            
            train_inds_df = pd.read_csv(folder+'Reprogramming_Train_Inds_Two_Classes.csv', index_col=0)
            val_inds_df = pd.read_csv(folder+'Reprogramming_Train_Inds_Two_Classes.csv', index_col=0)
            test_inds_df = pd.read_csv(folder+'Reprogramming_Train_Inds_Two_Classes.csv', index_col=0)
            
            # time_series_indices_list = []
            X_list = [[], [], []]
            y_list = [[], [], []]
            
            inds_dfs = [train_inds_df, val_inds_df, test_inds_df]
            for k in range(len(inds_dfs)): 
                for i, row in inds_dfs[k].iterrows(): 
                    cells = []
                    for j in cols: 
                        ind = row[j]
                        if ind == -1: 
                            cells.append(np.zeros(df.shape[1]))
                        else: 
                            cell = df.iloc[row[j]].to_numpy()
                            cells.append(cell)
                    X_list[k].append(cells)
                    y_list[k].append(row['state_info'])
            
            X_train = X_list[0]
            X_val = X_list[1]
            X_test = X_list[2]
            y_train = y_list[0]
            y_val = y_list[1]
            y_test = y_list[2]
            
            X_train = np.array(X_train, dtype=np.float32)
            X_val = np.array(X_val, dtype=np.float32)
            X_test = np.array(X_test, dtype=np.float32)
            ohenc = OneHotEncoder(sparse=False)
            y_train = np.array([y_train], dtype=str)
            y_train = y_train.swapaxes(0, 1)
            y_train = ohenc.fit_transform(y_train)
            y_val = np.array([y_val], dtype=str)
            y_val = y_val.swapaxes(0, 1)
            y_val = ohenc.fit_transform(y_val)
            y_test = np.array([y_test], dtype=str)
            y_test = y_test.swapaxes(0, 1)
            y_test = ohenc.fit_transform(y_test)
            print(X_train.shape)
            print(X_val.shape)
            print(X_test.shape)
            print(y_train.shape)
            print(y_val.shape)
            print(y_test.shape)
            
            ohenc_filename = "ohenc_reprogramming_two_classes.save"
            joblib.dump(ohenc, ohenc_filename) 
            
            with h5py.File(folder+'{0}_Genes_Data_Encoded_Time_Series_With_Class_Two_Classes.h5'.format(n_top_genes), 'a') as FOB: 
                FOB.create_dataset("X_train", data=X_train, dtype='f')
                FOB.create_dataset("X_val", data=X_val, dtype='f')
                FOB.create_dataset("X_test", data=X_test, dtype='f')
                FOB.create_dataset("y_train", data=y_train, dtype='f')
                FOB.create_dataset("y_val", data=y_val, dtype='f')
                FOB.create_dataset("y_test", data=y_test, dtype='f')
