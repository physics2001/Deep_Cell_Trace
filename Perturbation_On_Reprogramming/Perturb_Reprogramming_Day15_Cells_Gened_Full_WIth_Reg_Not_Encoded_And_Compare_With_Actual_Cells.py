# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 14:27:07 2023

@author: zhang
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import h5py
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import seaborn as sns
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, TimeDistributed, Layer, Bidirectional, GRU, Activation, Masking
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.activations import relu, softmax
from contextlib import redirect_stdout
from keras.losses import mean_squared_error
import time
from keras.losses import mean_squared_error, categorical_crossentropy
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
import gc
import cospar as cs
import scanpy as sc
from sklearn.metrics import r2_score

print(tf.config.list_physical_devices('GPU'))

n_top_genes = 1000
drop_out_list = [0.3]
num_layer_list = [1] #, 2
model_type_list = ["LSTM"] # , "GRU"

with h5py.File('{0}_Genes_Reprogramming_Time_Series_Day_6_9_12_15_Two_Classes.h5'.format(n_top_genes), 'a') as FOB: 
    print("Keys: %s" % FOB.keys())
    X = list(FOB['X'])
    FOB.close()
    
with h5py.File('{0}_Genes_Reprogramming_Time_Series_Gened_Day_21_28_Two_Classes.h5'.format(n_top_genes), 'a') as FOB: 
    print("Keys: %s" % FOB.keys())
    X_Gened = list(FOB['X'])
    FOB.close()

X = np.array(X)
X_Gened = np.array(X_Gened)

TIMESTEPS = X_Gened.shape[1]
FEATURES = X.shape[2]

df = pd.read_csv("Perturb_Reprogramming_{0}_Genes_Not_Encoded_Day_21_28_Gened_Change_In_Reprog_Result_With_Init_Class_Info.csv".format(n_top_genes))

max_list = df.loc[:,df.columns[~df.columns.isin(['class_info'])]].max().to_frame().T

max_list = max_list.to_numpy()
max_list = max_list.flatten()

top_100_inds = max_list.argsort()[-100:]

Reg_Day21_Model_Path = 'Regression_On_Reprogramming_Data/1000_Genes_Two_Classes_Day21/Features_1000_Dropout_0.25_NumLayer2_LSTM/model'
Reg_Day21_Model = load_model(Reg_Day21_Model_Path)
Reg_Day28_Model_Path = 'Regression_On_Reprogramming_Data/1000_Genes_Two_Classes_Day28/Features_1000_Dropout_0.25_NumLayer2_LSTM/model'
Reg_Day28_Model = load_model(Reg_Day28_Model_Path)

for DROP_OUT in drop_out_list: 
    for MODEL_TYPE in model_type_list: 
        for NUM_LAYER in num_layer_list: 
            folder = "Classification_On_Encoded_Reprogramming_Data/{4}_Genes_Not_Encoded_Two_Classes/Features_{0}_Dropout_{1}_NumLayer_{2}_ModelType_{3}/".format(FEATURES, DROP_OUT, NUM_LAYER, MODEL_TYPE, n_top_genes)
            filepath = "model"
            
            model = load_model(folder+filepath)
            y_Gened = model.predict(X_Gened)
            y_Gened = np.array(y_Gened)
            y_Gened_classes = np.argmax(y_Gened, axis=1)
            # y_Gened_classes = pd.DataFrame(y_Gened_classes, columns=('class_info'))
            
            RESULT_LIST = []
            X_mod_i = X.copy()
            
            for i in top_100_inds: 
                X_mod_i[:, :, i] = X_mod_i[:, :, i] * 0.0
            Predicted_Day21 = Reg_Day21_Model.predict(X_mod_i)
            Predicted_Day21 = np.array([Predicted_Day21], dtype=np.float32)
            Predicted_Day21 = np.swapaxes(Predicted_Day21, 0, 1)
            X_mod_i = np.concatenate((X_mod_i, Predicted_Day21), axis=1, dtype=np.float32)
            
            Predicted_Day28 = Reg_Day28_Model.predict(X_mod_i)
            Predicted_Day28 = np.array([Predicted_Day28], dtype=np.float32)
            Predicted_Day28 = np.swapaxes(Predicted_Day28, 0, 1)
            X_mod_i = np.concatenate((X_mod_i, Predicted_Day28), axis=1, dtype=np.float32)
            
            y_Gened_mod_i = model.predict(X_mod_i)
            y_Gened_mod_i = np.array(y_Gened_mod_i)
            
            print(y_Gened_mod_i.shape)
            print(y_Gened.shape)
            
            perturb_failed_ind = np.argwhere(y_Gened_mod_i[:, 1] <= 0.5)
            perturb_reprogramming_ind = np.argwhere(y_Gened_mod_i[:, 1] > 0.5)
            gened_failed = np.argwhere(y_Gened_classes <= 0.5)
            failed_to_reprogrammed_ind = np.intersect1d(gened_failed, perturb_reprogramming_ind)
            
            failed_to_reprogrammed = Predicted_Day28[failed_to_reprogrammed_ind, 0, :]
            
            print(perturb_failed_ind.shape)
            print(perturb_reprogramming_ind.shape)
            print(gened_failed.shape)
            print(failed_to_reprogrammed_ind.shape)
            print(failed_to_reprogrammed.shape)
            
            adata_orig=cs.datasets.reprogramming()
            
            sc.pp.filter_genes(adata_orig, min_cells=100)
            
            sc.pp.filter_genes(adata_orig, min_counts=1)
            
            sc.pp.filter_cells(adata_orig, min_genes=1000)
            
            sc.pp.log1p(adata_orig)
            sc.pp.highly_variable_genes(adata_orig, n_top_genes=n_top_genes)
            
            adata_orig = adata_orig[:, adata_orig.var['highly_variable']]
            df = adata_orig.to_df()
            column_names = df.columns.to_numpy()
            
            day28_reprogrammed_cells = adata_orig[(adata_orig.obs["time_info"]=="Day28") & (adata_orig.obs["state_info"]=="Reprogrammed")]
            day28_failed_cells = adata_orig[(adata_orig.obs["time_info"]=="Day28") & (adata_orig.obs["state_info"]=="Failed")]
            
            day28_reprogrammed_cells_avg = np.average(day28_reprogrammed_cells.to_df().to_numpy(), axis=0)
            day28_failed_cells_avg = np.average(day28_failed_cells.to_df().to_numpy(), axis=0)
            
            failed_to_reprogrammed_avg = np.average(failed_to_reprogrammed, axis=0)
            print(day28_reprogrammed_cells_avg.shape)
            print(day28_failed_cells_avg.shape)
            print(failed_to_reprogrammed_avg.shape)
            
            reprogrammed_vs_failed_to_reprogrammed = r2_score(day28_reprogrammed_cells_avg, failed_to_reprogrammed_avg)
            failed_vs_failed_to_reprogrammed = r2_score(day28_failed_cells_avg, failed_to_reprogrammed_avg)
            reprogrammed_vs_failed = r2_score(day28_reprogrammed_cells_avg, day28_failed_cells_avg)
            
            fig, ax = plt.subplots(figsize = (8, 6))
 
            # creating the bar plot
            ax.bar(["reprogrammed and perturbed", "failed and perturbed", "reprogrammed and failed"], 
                    [reprogrammed_vs_failed_to_reprogrammed, failed_vs_failed_to_reprogrammed, reprogrammed_vs_failed],
                    color=['red', 'orange', 'blue'])
            for container in ax.containers:
                ax.bar_label(container)
            plt.ylabel("Correlation")
            plt.title("Comparison of Correlation between reprogrammed, failed, and perturbed cells")
            plt.tight_layout()
            plt.savefig("Plots/Correlation_between_reprogrammed_failed_perturbed_cells.png")
            plt.savefig("Plots/pdfs/Correlation_between_reprogrammed_failed_perturbed_cells.pdf")
            
            # delta_p = y_Gened_mod_i[:, 1] - y_Gened[:, 1]
            # RESULT_LIST.append(delta_p)
            
            # RESULT_LIST = np.array(RESULT_LIST)
            # RESULT_LIST = np.swapaxes(RESULT_LIST, 0, 1)
            
            # RESULT_DF = pd.DataFrame(RESULT_LIST)
            # RESULT_DF = pd.concat((RESULT_DF, y_Gened_classes), axis=1)
            # RESULT_DF.to_csv("Perturb_Reprogramming_{0}_Genes_Not_Encoded_Day_21_28_Gened_Change_In_Reprog_Result_With_Init_Class_Info_Top_100.csv".format(n_top_genes), index=False)
