# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 21:10:11 2022

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

with h5py.File('{0}_Genes_Reprogramming_Two_Classes_Data_Time_Series_With_Class.h5'.format(n_top_genes), 'a') as f: 
    print("Keys: %s" % f.keys())
    
    X_test = list(f['X_test'])
    class_info = list(f['y_test'])

X_test = np.array(X_test)
class_info = np.array(class_info)
y_test = X_test[:, 5, : ].copy()

print(X_test.shape)
print(class_info.shape)
print(y_test.shape)

failed_indices = np.where(class_info == 0)[0]
reprogrammed_indices = np.where(class_info == 1)[0]
print(failed_indices.shape)
print(reprogrammed_indices.shape)

reg_folder = "Regression_On_Reprogramming_Data/1000_Genes_Two_Classes_Day28/Features_1000_Dropout_0.25_NumLayer2_LSTM/"

y_test_pred = pd.read_csv(reg_folder+"y_test_predicted.csv", index_col=0)

y_test_pred = y_test_pred.to_numpy()

y_test_failed = y_test[15000:, :]
y_test_reprogrammed = y_test[:15000, :]
print(y_test_failed.shape)
print(y_test_reprogrammed.shape)

y_test_failed_pred = y_test_pred[15000:, :]
y_test_reprogrammed_pred = y_test_pred[:15000, :]
print(y_test_failed_pred.shape)
print(y_test_reprogrammed_pred.shape)

y_test_failed_avgs = np.average(y_test_failed, axis=0)
y_test_reprogrammed_avgs = np.average(y_test_reprogrammed, axis=0)
print(y_test_failed_avgs.shape)
print(y_test_reprogrammed_avgs.shape)

y_test_failed_pred_avgs = np.average(y_test_failed_pred, axis=0)
y_test_reprogrammed_pred_avgs = np.average(y_test_reprogrammed_pred, axis=0)
print(y_test_failed_pred_avgs.shape)
print(y_test_reprogrammed_pred_avgs.shape)

real_vs_predicted_failed = np.array([y_test_failed_avgs, y_test_failed_pred_avgs])
real_vs_predicted_failed = np.swapaxes(real_vs_predicted_failed, 0, 1)

real_vs_predicted_reprogrammed = np.array([y_test_reprogrammed_avgs, y_test_reprogrammed_pred_avgs])
real_vs_predicted_reprogrammed = np.swapaxes(real_vs_predicted_reprogrammed, 0, 1)

gene_names = np.loadtxt('Reprogramming_1000_Genes_Logged_True_Scaled_False_Column_Names.txt', dtype=str)

failed_df = pd.DataFrame(real_vs_predicted_failed, columns=['Real', 'Predicted'], index=gene_names)
reprogrammed_df = pd.DataFrame(real_vs_predicted_reprogrammed, columns=['Real', 'Predicted'], index=gene_names)

import seaborn.objects as so

Interested_Genes = ['Capn6',
                    'Steap2',
                    'Trf',
                    'Wnt5a', ]

failed_df_selected = failed_df[failed_df.index.isin(Interested_Genes)]
reprogrammed_df_selected = reprogrammed_df[reprogrammed_df.index.isin(Interested_Genes)]
failed_df_selected['Gene_Name'] = failed_df_selected.index
reprogrammed_df_selected['Gene_Name'] = reprogrammed_df_selected.index

plt.rcParams.update({'font.size': 11})
fig, ax = plt.subplots(figsize=(6, 6))
p1 = sns.scatterplot(data=failed_df, x='Predicted', y='Real', ax=ax)
p2 = so.Plot(failed_df_selected,x='Predicted',y='Real', text='Gene_Name').add(so.Dot(color='red')).add(so.Text(halign='left', valign='bottom')).on(ax).plot()
ax.plot([0, 1], [0, 1], transform=ax.transAxes)
ax.annotate("r-squared = {:.3f}".format(r2_score(failed_df['Real'], failed_df['Predicted'])), (0.5, 4.5))
plt.xlim((0, 5))
plt.ylim((0, 5))
plt.title('Failed Cell Real vs Predicted in Reprogramming')
plt.tight_layout()
plt.savefig('Plots/Real_vs_Predicted_Avg_Gene_Expressions_For_Failed_Type_Reprogramming.png')
plt.savefig('Plots/pdfs/Real_vs_Predicted_Avg_Gene_Expressions_For_Failed_Type_Reprogramming.pdf')
plt.close()

plt.clf()
plt.rcParams.update({'font.size': 11})
fig, ax = plt.subplots(figsize=(6, 6))
p1 = sns.scatterplot(data=reprogrammed_df, x='Predicted', y='Real', ax=ax)
p2 = so.Plot(reprogrammed_df_selected,x='Predicted',y='Real', text='Gene_Name').add(so.Dot(color='red')).add(so.Text(halign='left', valign='bottom')).on(ax).plot()
ax.plot([0, 1], [0, 1], transform=ax.transAxes)
ax.annotate("r-squared = {:.3f}".format(r2_score(reprogrammed_df['Real'], reprogrammed_df['Predicted'])), (0.5, 4.5))
plt.xlim((0, 5))
plt.ylim((0, 5))
plt.title('Regrogrammed Cell Real vs Predicted in Reprogramming')
plt.tight_layout()
plt.savefig('Plots/Real_vs_Predicted_Avg_Gene_Expressions_For_Reprogrammed_Type_Reprogramming.png')
plt.savefig('Plots/pdfs/Real_vs_Predicted_Avg_Gene_Expressions_For_Reprogrammed_Type_Reprogramming.pdf')
plt.close()

