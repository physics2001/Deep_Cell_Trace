# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 22:27:22 2023

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

with h5py.File('1000_Genes_Data_Not_Encoded_Time_Series_Day_2_4_Two_Classes.h5', 'a') as FOB: 
    print("Keys: %s" % FOB.keys())
    X = list(FOB['X'])
    FOB.close()
    
with h5py.File('1000_Genes_Hemato_Time_Series_Gened_Day_6_Two_Classes.h5', 'r') as f:
    print("Keys: %s" % f.keys())
    X_Gened = list(f['X'])
    f.close()

X = np.array(X)
X_Gened = np.array(X_Gened)

print(X.shape)
print(X_Gened.shape)

TIMESTEPS = X_Gened.shape[1]
FEATURES = X.shape[2]

all_hemato_1000_gene_names = np.loadtxt("Hemato_1000_Genes_Unique_Column_Names.txt", dtype=str).tolist()
print(all_hemato_1000_gene_names)

# df = pd.read_csv("Perturb_Reprogramming_{0}_Genes_Not_Encoded_Day_21_28_Gened_Change_In_Reprog_Result_With_Init_Class_Info.csv".format(n_top_genes))

# max_list = df.loc[:,df.columns[~df.columns.isin(['class_info'])]].max().to_frame().T

# max_list = max_list.to_numpy()
# max_list = max_list.flatten()

# top_100_inds = max_list.argsort()[-100:]

df = pd.read_csv("Avg_Abs_Shap_Values/Hemo_Avg_Abs_Shap_Values_Sorted_Neutrophil_Day4.csv")
top_neu_100_shap_names = df['Gene_Name'][:20].tolist()
print(top_neu_100_shap_names)

top_neu_100_shap_inds = [all_hemato_1000_gene_names.index(name) for name in top_neu_100_shap_names]
print(top_neu_100_shap_inds)

Reg_Day6_Model_Path = 'Regression_On_Not_Encoded_Data_Neu_Mon_Two_Classes/1000_Genes_Two_Classes_Day6/Features_1000_Dropout_0.3_NumLayer2_LSTM/model'
Reg_Day6_Model = load_model(Reg_Day6_Model_Path)

for DROP_OUT in drop_out_list: 
    for MODEL_TYPE in model_type_list: 
        for NUM_LAYER in num_layer_list: 
            folder = "Classification_On_Not_Encoded_Data_Mon_Neu_Two_Classes/Features_1000_Dropout_0.3_NumLayer_1_ModelType_LSTM/"
            filepath = "model"
            
            model = load_model(folder+filepath)
            y_Gened = model.predict(X_Gened)
            y_Gened = np.array(y_Gened)
            y_Gened_classes = np.argmax(y_Gened, axis=1)
            # y_Gened_classes = pd.DataFrame(y_Gened_classes, columns=('class_info'))
            
            RESULT_LIST = []
            X_mod_i = X.copy()
            
            print(X_mod_i.shape)
            
            for i in top_neu_100_shap_inds: 
                X_mod_i[:, :, i] = X_mod_i[:, :, i] * 0.0
            Predicted_Day6 = Reg_Day6_Model.predict(X_mod_i)
            Predicted_Day6 = np.array([Predicted_Day6], dtype=np.float32)
            Predicted_Day6 = np.swapaxes(Predicted_Day6, 0, 1)
            X_mod_i = np.concatenate((X_mod_i, Predicted_Day6), axis=1, dtype=np.float32)
            
            y_Gened_mod_i = model.predict(X_mod_i)
            y_Gened_mod_i = np.array(y_Gened_mod_i)
            
            print(y_Gened_mod_i.shape)
            print(y_Gened.shape)
            
            perturb_mon_ind = np.argwhere(y_Gened_mod_i[:, 1] <= 0.5)
            perturb_neu_ind = np.argwhere(y_Gened_mod_i[:, 1] > 0.5)
            gened_neu = np.argwhere(y_Gened_classes > 0.5)
            neu_to_mon_ind = np.intersect1d(gened_neu, perturb_mon_ind)
            
            neu_to_mon = Predicted_Day6[neu_to_mon_ind, 0, :]
            
            print(perturb_mon_ind.shape)
            print(perturb_neu_ind.shape)
            print(gened_neu.shape)
            print(neu_to_mon_ind.shape)
            print(neu_to_mon.shape)
            print(neu_to_mon)
            print(" ")
            
            adata_orig=cs.datasets.hematopoiesis()
    
            sc.pp.filter_genes(adata_orig, min_cells=100)
            
            sc.pp.filter_genes(adata_orig, min_counts=1)
            
            sc.pp.filter_cells(adata_orig, min_genes=1000)
            
            sc.pp.log1p(adata_orig)
            
            sc.pp.highly_variable_genes(adata_orig, n_top_genes=1000)
            
            adata_orig = adata_orig[:, adata_orig.var['highly_variable']]
            
            df = adata_orig.to_df()
            column_names = df.columns.to_numpy()
            
            state_info = adata_orig.obs["state_info"]
            
            day6_neu_cells = adata_orig[(adata_orig.obs["time_info"]=="6") & (adata_orig.obs["state_info"]=="Neutrophil")]
            day6_mon_cells = adata_orig[(adata_orig.obs["time_info"]=="6") & (adata_orig.obs["state_info"]=="Monocyte")]
            
            day6_neu_cells_avg = np.average(day6_neu_cells.to_df().to_numpy(), axis=0)
            day6_mon_cells_avg = np.average(day6_mon_cells.to_df().to_numpy(), axis=0)
            
            neu_to_mon_avg = np.average(neu_to_mon, axis=0)
            print(day6_neu_cells_avg)
            print(" ")
            print(day6_mon_cells_avg)
            print(" ")
            print(neu_to_mon_avg)
            print(" ")
            print(day6_neu_cells_avg.shape)
            print(day6_mon_cells_avg.shape)
            print(neu_to_mon_avg.shape)
            
            neu_vs_neu_to_mon = r2_score(day6_neu_cells_avg, neu_to_mon_avg)
            mon_vs_neu_to_mon = r2_score(day6_mon_cells_avg, neu_to_mon_avg)
            neu_vs_mon = r2_score(day6_neu_cells_avg, day6_mon_cells_avg)
            
            fig, ax = plt.subplots(figsize = (8, 6))
 
            # creating the bar plot
            ax.bar(["monocyte and perturbed", "neutrophil and perturbed", "neutrophil and monocyte"], 
                    [mon_vs_neu_to_mon, neu_vs_neu_to_mon, neu_vs_mon],
                    color=['red', 'orange', 'blue'])
            for container in ax.containers:
                ax.bar_label(container)
            plt.ylabel("Correlation")
            plt.title("Comparison of Correlation between neutrophil, monocyte, and perturbed cells")
            plt.tight_layout()
            plt.savefig("Plots/Correlation_between_hemato_original_cells_and_perturbed_cells_that_changed_from_neu_to_mon_20_genes.png")
            plt.savefig("Plots/pdfs/Correlation_between_hemato_original_cells_and_perturbed_cells_that_changed_from_neu_to_mon_20_genes.pdf")