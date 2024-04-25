# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 21:00:48 2022

@author: zhang
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import h5py
import tensorflow as tf
import os
from keras.models import load_model
import seaborn as sns
from sklearn.metrics import r2_score

print(tf.config.list_physical_devices('GPU'))

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

with h5py.File('{0}_Genes_Logged_Time_Series_With_Monocyte_Neutrophil_Two_Classes.h5'.format(n_top_genes), 'a') as f: 
    print("Keys: %s" % f.keys())
    
    X_test = list(f['X_test'])
    class_info = list(f['y_test'])


X_test = np.array(X_test)
class_info = np.array(class_info)
y_test = X_test[:, 2, : ].copy()

print(X_test.shape)
print(class_info.shape)
print(y_test.shape)

class_info = class_info[:, 1]

mon_indices = np.where(class_info == 0)[0]
neu_indices = np.where(class_info == 1)[0]
print(mon_indices.shape)
print(neu_indices.shape)

reg_folder = "Regression_On_Not_Encoded_Data_Neu_Mon_Two_Classes/1000_Genes_Two_Classes_Day6/Features_1000_Dropout_0.3_NumLayer2_LSTM/"

y_test_pred = pd.read_csv(reg_folder+"y_test_predicted.csv", index_col=0)

y_test_pred = y_test_pred.to_numpy()

y_test_mon = y_test[mon_indices, :]
y_test_neu = y_test[neu_indices, :]
print(y_test_mon.shape)
print(y_test_neu.shape)

y_test_mon_pred = y_test_pred[mon_indices, :]
y_test_neu_pred = y_test_pred[neu_indices, :]
print(y_test_mon_pred.shape)
print(y_test_neu_pred.shape)

y_test_mon_avgs = np.average(y_test_mon, axis=0)
y_test_neu_avgs = np.average(y_test_neu, axis=0)
print(y_test_mon_avgs.shape)
print(y_test_neu_avgs.shape)

y_test_mon_pred_avgs = np.average(y_test_mon_pred, axis=0)
y_test_neu_pred_avgs = np.average(y_test_neu_pred, axis=0)
print(y_test_mon_pred_avgs.shape)
print(y_test_neu_pred_avgs.shape)

real_vs_predicted_mon = np.array([y_test_mon_avgs, y_test_mon_pred_avgs])
real_vs_predicted_mon = np.swapaxes(real_vs_predicted_mon, 0, 1)

real_vs_predicted_neu = np.array([y_test_neu_avgs, y_test_neu_pred_avgs])
real_vs_predicted_neu = np.swapaxes(real_vs_predicted_neu, 0, 1)

gene_names = np.loadtxt('1000_Genes_Logged_True_Scaled_False_Column_Names.txt', dtype=str)

mon_df = pd.DataFrame(real_vs_predicted_mon, columns=['Real', 'Predicted'], index=gene_names)
neu_df = pd.DataFrame(real_vs_predicted_neu, columns=['Real', 'Predicted'], index=gene_names)

import seaborn.objects as so

Interested_Genes = ['mt-Atp6',
                    'mt-Co2',
                    'Ppia',
                    'Itgb2',
                    'Cd33',
                    'Arg2',]
                    

mon_df_selected = mon_df[mon_df.index.isin(Interested_Genes)]
neu_df_selected = neu_df[neu_df.index.isin(Interested_Genes)]
mon_df_selected['Gene_Name'] = mon_df_selected.index
neu_df_selected['Gene_Name'] = neu_df_selected.index

plt.rcParams.update({'font.size': 11})
fig, ax = plt.subplots(figsize=(6, 6))
p1 = sns.scatterplot(data=mon_df, x='Predicted', y='Real', ax=ax)
p2 = so.Plot(mon_df_selected,x='Predicted',y='Real', text='Gene_Name').add(so.Dot(color='red')).add(so.Text(halign='left', valign='bottom')).on(ax).plot()
ax.plot([0, 1], [0, 1], transform=ax.transAxes)
ax.annotate("r-squared = {:.3f}".format(r2_score(mon_df['Real'], mon_df['Predicted'])), (0.5, 4.5))
plt.xlim((0, 5))
plt.ylim((0, 5))
plt.title('Monocyte Cell Real vs Predicted in Hematopoiesis')
plt.tight_layout()
plt.savefig('Plots/Real_vs_Predicted_Avg_Gene_Expressions_For_Monocyte_Type_Hematopoiesis.png')
plt.savefig('Plots/pdfs/Real_vs_Predicted_Avg_Gene_Expressions_For_Monocyte_Type_Hematopoiesis.pdf')
plt.close()

plt.clf()
plt.rcParams.update({'font.size': 11})
fig, ax = plt.subplots(figsize=(6, 6))
p1 = sns.scatterplot(data=neu_df, x='Predicted', y='Real', ax=ax)
p2 = so.Plot(neu_df_selected,x='Predicted',y='Real', text='Gene_Name').add(so.Dot(color='red')).add(so.Text(halign='left', valign='bottom')).on(ax).plot()
ax.plot([0, 1], [0, 1], transform=ax.transAxes)
ax.annotate("r-squared = {:.3f}".format(r2_score(neu_df['Real'], neu_df['Predicted'])), (0.5, 4.5))
plt.xlim((0, 5))
plt.ylim((0, 5))
plt.title('Neutrophol Cell Real vs Predicted in Hematopoiesis')
plt.tight_layout()
plt.savefig('Plots/Real_vs_Predicted_Avg_Gene_Expressions_For_Neutrophil_Type_Hematopoiesis.png')
plt.savefig('Plots/pdfs/Real_vs_Predicted_Avg_Gene_Expressions_For_Neutrophil_Type_Hematopoiesis.pdf')
plt.close()
