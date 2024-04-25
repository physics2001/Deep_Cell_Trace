# -*- coding: utf-8 -*-
"""
Created on Wed Jul 26 19:36:31 2023

@author: zhang
"""

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import h5py
# import tensorflow as tf
# import os
# from keras.models import Sequential, load_model
# from keras.layers import LSTM, Dense, TimeDistributed, Layer, Bidirectional, GRU, Activation, Masking
# from keras.callbacks import ModelCheckpoint, EarlyStopping
# from keras.activations import relu
# from contextlib import redirect_stdout
# from keras.losses import mean_squared_error
# import time
# import seaborn as sns

# print(tf.config.list_physical_devices('GPU'))

# n_top_genes_list = [1000] #100, 500, 5000, 2000, 
# drop_out_list = [0.3]
# num_layer_list = [2, 4]
# model_type_list = ["LSTM", "GRU"]

# LOG_DATA = [True, False]

# SCALE_DATA = [True, False]

# model_hyperparameters_list = [[500, 100, 75, 0.3, 0.2], 
#                               [500, 200, 50, 0.1, 0.3]]

# RESULT_LIST = []
# RESULT_COLUMNS = ['Num_genes', 'Reg_Model_Type', 'Num_Layers_BRNN', 
#                   'Reg_Dropout', 'Reg_P_Corr', 'Reg_Sp_Corr', 'mse']

# gene_names = np.loadtxt('1000_Genes_Logged_True_Scaled_False_Column_Names.txt', dtype=str)

# storage = "Plots/Regression_Correlation_Results_Hematopoiesis/"

# def Draw_Correlation_Graph(y_test, y_test_hat, storage, name): 
#     y_test_df = pd.DataFrame(y_test)
#     y_test_hat_df = pd.DataFrame(y_test_hat)
#     spcorr_test = y_test_hat_df.corrwith(y_test_df, method='spearman', axis=1)
#     pcorr_test = y_test_hat_df.corrwith(y_test_df, method='pearson', axis=1)
#     spcorr_test.to_csv(storage+"spearman_correlation_test_data_{0}.csv".format(name))
#     pcorr_test.to_csv(storage+"pearson_correlation_test_data_{0}.csv".format(name))
#     y_test_hat_df.to_csv(storage+"y_test_predicted_{0}.csv".format(name))
    
#     avg_pcorr_test = np.mean(pcorr_test)
#     avg_spcorr_test = np.mean(spcorr_test)
#     plt.figure(figsize=(10, 8))
#     plt.ylim((0,10))
#     plt.hist(spcorr_test, bins=20, range=(0, 1), density=True)
#     plt.title("Average Spearman Correlation = {0}".format(round(avg_spcorr_test, 4)))
#     plt.savefig(storage+"Hematopoiesis_Day6_Spearman_Correlation_Hist_{0}.png".format(name))
#     plt.savefig(storage+"pdfs/Hematopoiesis_Day6_Spearman_Correlation_Hist_{0}.pdf".format(name))
#     plt.clf()
    
#     plt.figure(figsize=(10, 8))
#     plt.ylim((0,10))
#     plt.hist(pcorr_test, bins=20, range=(0, 1), density=True)
#     plt.title("Average Pearson Correlation = {0}".format(round(avg_pcorr_test, 4)))
#     plt.savefig(storage+"Hematopoiesis_Day6_Pearson_Correlation_Hist_{0}.png".format(name))
#     plt.savefig(storage+"pdfs/Hematopoiesis_Day6_Pearson_Correlation_Hist_{0}.pdf".format(name))
#     plt.clf()
    
#     return (avg_pcorr_test, avg_spcorr_test)

# for model_hyperparameters in model_hyperparameters_list: 
#     dim1 = model_hyperparameters[0]
#     dim2 = model_hyperparameters[1]
#     dim3 = model_hyperparameters[2]
#     alpha = model_hyperparameters[3]
#     dropout = model_hyperparameters[4]
#     for log in LOG_DATA: 
#         for scale in SCALE_DATA: 
#             for n_top_genes in n_top_genes_list: 
#                 folder = "Dimension_Reduction/Reprogramming/{7}_Genes_Logged_{8}_Scaled_{9}_Data/E1_{0}_E2_{1}_BN_{2}_D1_{3}_D2_{4}_alpha_{5}_dropout_{6}/".format(dim1, dim2, dim3, dim2, dim1, alpha, dropout, n_top_genes, log, scale)
                
#                 with h5py.File('{0}_Genes_Logged_Time_Series_With_Monocyte_Neutrophil_Two_Classes.h5'.format(n_top_genes), 'r') as f:
#                     print("Keys: %s" % f.keys())
                    
#                     X_test = list(f['X_test'])
#                     test_classes = list(f['y_test'])
                
#                 X_test = np.array(X_test)
#                 test_classes = np.array(test_classes)
                
#                 y_test = X_test[:, 2, : ].copy()
#                 X_test = X_test[:, 0:2, : ]
                
#                 print(X_test.shape)
#                 print(y_test.shape)
#                 print(test_classes.shape)
                
#                 test_classes = test_classes[:,1]
#                 print(test_classes.shape)
                
#                 TIMESTEPS = X_test.shape[1]
#                 FEATURES = X_test.shape[2]
                
#                 for DROP_OUT in drop_out_list: 
#                     for MODEL_TYPE in model_type_list: 
#                         for NUM_LAYER in num_layer_list: 
#                             reg_folder = "Regression_On_Not_Encoded_Data_Neu_Mon_Two_Classes/{4}_Genes_Two_Classes_Day6/Features_{0}_Dropout_{1}_NumLayer{2}_{3}/".format(FEATURES, DROP_OUT, NUM_LAYER, MODEL_TYPE, n_top_genes)
#                             filepath = "model"
                            
#                             model = load_model(reg_folder + filepath)
                            
#                             y_test_hat = model.predict(X_test)
                            
#                             neu_ind = np.argwhere(test_classes == 1).flatten()
#                             mon_ind = np.argwhere(test_classes == 0).flatten()
#                             print(neu_ind.shape)
#                             print(mon_ind.shape)
#                             y_test_neu = y_test[neu_ind, :]
#                             y_test_hat_neu = y_test_hat[neu_ind, :]
#                             y_test_mon = y_test[mon_ind]
#                             y_test_hat_mon = y_test_hat[mon_ind]
                            
#                             avg_pcorr_test_neu, avg_spcorr_test_neu = Draw_Correlation_Graph(y_test_neu, y_test_hat_neu, storage, "neu")
#                             avg_pcorr_test_mon, avg_spcorr_test_mon = Draw_Correlation_Graph(y_test_mon, y_test_hat_mon, storage, "mon")
#                             avg_pcorr_test_all, avg_spcorr_test_all = Draw_Correlation_Graph(y_test, y_test_hat, storage, "all")
                            
#                             d = {"Cell Type": ["Neutrophil", "Monocyte", "both"], 
#                                  "Average Pearson Correlation": [avg_pcorr_test_neu, 
#                                                                  avg_pcorr_test_mon, 
#                                                                  avg_pcorr_test_all]}
#                             df = pd.DataFrame(data = d)
                            
#                             fig, ax = plt.subplots(figsize=(10, 8))
#                             ax = sns.barplot(data=df, x="Cell Type", y="Average Pearson Correlation")
#                             ax.bar_label(ax.containers[0])
#                             plt.savefig(storage+"Hematopoiesis_Day6_Average_Pearson_Correlation_Hist_Comparison.png")
#                             plt.savefig(storage+"pdfs/Hematopoiesis_Day6_Average_Pearson_Correlation_Hist_Comparison.pdf")


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

df = pd.read_csv("Avg_Abs_Shap_Values/Hemo_Avg_Abs_Shap_Values_Sorted_Monocyte_Day4.csv")
top_mono_100_shap_names = df['Gene_Name'][:100].tolist()
print(top_mono_100_shap_names)

top_mono_100_shap_inds = [all_hemato_1000_gene_names.index(name) for name in top_mono_100_shap_names]
print(top_mono_100_shap_inds)

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
            
            for i in top_mono_100_shap_inds: 
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
            gened_mon = np.argwhere(y_Gened_classes <= 0.5)
            mon_to_neu_ind = np.intersect1d(gened_mon, perturb_neu_ind)
            
            mon_to_neu = Predicted_Day6[mon_to_neu_ind, 0, :]
            
            print(perturb_mon_ind.shape)
            print(perturb_neu_ind.shape)
            print(gened_mon.shape)
            print(mon_to_neu_ind.shape)
            print(mon_to_neu.shape)
            
            adata_orig=cs.datasets.hematopoiesis()
    
            sc.pp.filter_genes(adata_orig, min_cells=100)
            
            sc.pp.filter_genes(adata_orig, min_counts=1)
            
            sc.pp.filter_cells(adata_orig, min_genes=1000)
            
            sc.pp.log1p(adata_orig)
            
            sc.pp.highly_variable_genes(adata_orig, n_top_genes=1000)
            
            adata_orig = adata_orig[:, adata_orig.var['highly_variable']]
            
            df = adata_orig.to_df()
            column_names = df.columns.to_numpy()
            
            day6_neu_cells = adata_orig[(adata_orig.obs["time_info"]=="Day6") & (adata_orig.obs["state_info"]=="Neutrophil")]
            day6_mon_cells = adata_orig[(adata_orig.obs["time_info"]=="Day6") & (adata_orig.obs["state_info"]=="Monocyte")]
            
            day6_neu_cells_avg = np.average(day6_neu_cells.to_df().to_numpy(), axis=0)
            day6_mon_cells_avg = np.average(day6_mon_cells.to_df().to_numpy(), axis=0)
            
            mon_to_neu_avg = np.average(mon_to_neu, axis=0)
            print(day6_neu_cells_avg.shape)
            print(day6_mon_cells_avg.shape)
            print(mon_to_neu_avg.shape)
            
            neu_vs_mon_to_neu = r2_score(day6_neu_cells_avg, mon_to_neu_avg)
            mon_vs_mon_to_neu = r2_score(day6_mon_cells_avg, mon_to_neu_avg)
            neu_vs_mon = r2_score(day6_neu_cells_avg, day6_mon_cells_avg)
            
            fig, ax = plt.subplots(figsize = (8, 6))
 
            # creating the bar plot
            ax.bar(["neutrophil and perturbed", "monocyte and perturbed", "neutrophil and monocyte"], 
                    [neu_vs_mon_to_neu, mon_vs_mon_to_neu, neu_vs_mon],
                    color=['red', 'orange', 'blue'])
            for container in ax.containers:
                ax.bar_label(container)
            plt.ylabel("Correlation")
            plt.title("Comparison of Correlation between neutrophil, monocyte, and perturbed cells")
            plt.tight_layout()
            plt.savefig("Plots/Correlation_between_hemato_monocyte_perturbed_cells.png")
            plt.savefig("Plots/pdfs/Correlation_between_hemato_monocyte_perturbed_cells.pdf")
            
            # delta_p = y_Gened_mod_i[:, 1] - y_Gened[:, 1]
            # RESULT_LIST.append(delta_p)
            
            # RESULT_LIST = np.array(RESULT_LIST)
            # RESULT_LIST = np.swapaxes(RESULT_LIST, 0, 1)
            
            # RESULT_DF = pd.DataFrame(RESULT_LIST)
            # RESULT_DF = pd.concat((RESULT_DF, y_Gened_classes), axis=1)
            # RESULT_DF.to_csv("Perturb_Reprogramming_{0}_Genes_Not_Encoded_Day_21_28_Gened_Change_In_Reprog_Result_With_Init_Class_Info_Top_100.csv".format(n_top_genes), index=False)
