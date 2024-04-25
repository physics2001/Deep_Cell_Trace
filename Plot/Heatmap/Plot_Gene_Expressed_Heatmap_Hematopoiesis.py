# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 23:31:29 2022

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

# calc percentage expressed
y_test_mon_percentage = np.count_nonzero(y_test_mon > 0.08, axis=0) / y_test_mon.shape[0]
y_test_neu_percentage = np.count_nonzero(y_test_neu > 0.08, axis=0) / y_test_neu.shape[0]
y_test_mon_pred_percentage = np.count_nonzero(y_test_mon_pred > 0.08, axis=0) / y_test_mon_pred.shape[0]
y_test_neu_pred_percentage = np.count_nonzero(y_test_neu_pred > 0.08, axis=0) / y_test_neu_pred.shape[0]

# calc average
y_test_mon_avgs = np.average(y_test_mon, axis=0)
y_test_neu_avgs = np.average(y_test_neu, axis=0)
print(y_test_mon_avgs.shape)
print(y_test_neu_avgs.shape)

y_test_mon_pred_avgs = np.average(y_test_mon_pred, axis=0)
y_test_neu_pred_avgs = np.average(y_test_neu_pred, axis=0)
print(y_test_mon_pred_avgs.shape)
print(y_test_neu_pred_avgs.shape)

real_mon = np.array([y_test_mon_avgs])
real_mon = np.swapaxes(real_mon, 0, 1)

predicted_mon = np.array([y_test_mon_pred_avgs])
predicted_mon = np.swapaxes(predicted_mon, 0, 1)

real_neu = np.array([y_test_neu_avgs])
real_neu = np.swapaxes(real_neu, 0, 1)

predicted_neu = np.array([y_test_neu_pred_avgs])
predicted_neu = np.swapaxes(predicted_neu, 0, 1)

print(real_mon.shape)
print(predicted_mon.shape)
print(real_neu.shape)
print(predicted_neu.shape)

gene_names = np.loadtxt('1000_Genes_Logged_True_Scaled_False_Column_Names.txt', dtype=str)

mon_real_df = pd.DataFrame(real_mon, columns=['Expression_Level'], index=gene_names)
neu_real_df = pd.DataFrame(real_neu, columns=['Expression_Level'], index=gene_names)

mon_predicted_df = pd.DataFrame(predicted_mon, columns=['Expression_Level'], index=gene_names)
neu_predicted_df = pd.DataFrame(predicted_neu, columns=['Expression_Level'], index=gene_names)

mon_real_df = mon_real_df.assign(Gene_Names=gene_names)
neu_real_df = neu_real_df.assign(Gene_Names=gene_names)
mon_predicted_df = mon_predicted_df.assign(Gene_Names=gene_names)
neu_predicted_df = neu_predicted_df.assign(Gene_Names=gene_names)

mon_real_df = mon_real_df.assign(Percentage_Expressing=y_test_mon_percentage)
neu_real_df = neu_real_df.assign(Percentage_Expressing=y_test_neu_percentage)
mon_predicted_df = mon_predicted_df.assign(Percentage_Expressing=y_test_mon_pred_percentage)
neu_predicted_df = neu_predicted_df.assign(Percentage_Expressing=y_test_neu_pred_percentage)

print(mon_real_df)
print(neu_real_df)
print(mon_predicted_df)
print(neu_predicted_df)

Interested_Genes = ['Adgre1', 'Atp6v0d2', 'Camp', 'Casp6', 'Ccl6', 
                    'Cd74','Cd300c2', 'Clec4d', 'Clec4n', 'Dpep2',
                    'Dusp3', 'Gpnmb', 'H2-Aa', 'Lyz2', 'Mgl2', 
                    'Mmp12', 'Prss57', 'Rnh1', 'Spp1', 'Vcan']

mon_real_df = mon_real_df[mon_real_df['Gene_Names'].isin(Interested_Genes)]
neu_real_df = neu_real_df[neu_real_df['Gene_Names'].isin(Interested_Genes)]
mon_predicted_df = mon_predicted_df[mon_predicted_df['Gene_Names'].isin(Interested_Genes)]
neu_predicted_df = neu_predicted_df[neu_predicted_df['Gene_Names'].isin(Interested_Genes)]

print(mon_real_df)
print(neu_real_df)
print(mon_predicted_df)
print(neu_predicted_df)

mon_real_df = mon_real_df.assign(Type=['Real_Monocyte_Expressions']*mon_real_df.shape[0])
neu_real_df = neu_real_df.assign(Type=['Real_Neutrophil_Expressions']*neu_real_df.shape[0])
mon_predicted_df = mon_predicted_df.assign(Type=['Predicted_Monocyte_Expressions']*mon_predicted_df.shape[0])
neu_predicted_df = neu_predicted_df.assign(Type=['Predicted_Neutrophil_Expressions']*neu_predicted_df.shape[0])

print(mon_real_df)
print(neu_real_df)
print(mon_predicted_df)
print(neu_predicted_df)

all_df = pd.concat([mon_real_df, mon_predicted_df, neu_real_df, neu_predicted_df], axis=0)
print(all_df.shape)
all_df.sort_values(by=['Gene_Names'])

# Draw each cell as a scatter point with varying size and color
sns.set(rc={'figure.figsize':(10,8)})
sns.set(font_scale=2)
sns.set_theme(style="whitegrid", font_scale=2.2)
g = sns.relplot(
    data=all_df,
    x="Gene_Names", y="Type", hue="Expression_Level", size="Percentage_Expressing",
    palette="vlag", hue_norm=(-1, 1), edgecolor=".7",
    height=10, sizes=(100, 300), size_norm=(-.2, .8), aspect=2.5
)

# Tweak the figure to finalize
# g.set(xlabel="", ylabel="", aspect="equal")
g.despine(left=True, bottom=True)
g.ax.margins(.02)
for label in g.ax.get_xticklabels():
    label.set_rotation(90)
for artist in g.legend.legendHandles:
    artist.set_edgecolor(".7")

g.legend.set(loc="center left", bbox_to_anchor=(0.81, 0.5), in_layout=False)
plt.tight_layout(rect=[0, 0, 0.8, 1])
plt.savefig("Plots/Heatmap_Hemotapoiesis_New.png")
plt.savefig("Plots/pdfs/Heatmap_Hemotapoiesis_New.pdf")

# sns.set_theme(style="whitegrid")

# # Load the brain networks dataset, select subset, and collapse the multi-index
# df = sns.load_dataset("brain_networks", header=[0, 1, 2], index_col=0)

# used_networks = [1, 5, 6, 7, 8, 12, 13, 17]
# used_columns = (df.columns
#                   .get_level_values("network")
#                   .astype(int)
#                   .isin(used_networks))
# df = df.loc[:, used_columns]

# df.columns = df.columns.map("-".join)

# # Compute a correlation matrix and convert to long-form
# corr_mat = df.corr().stack().reset_index(name="correlation")

# # Draw each cell as a scatter point with varying size and color
# g = sns.relplot(
#     data=corr_mat,
#     x="level_0", y="level_1", hue="correlation", size="correlation",
#     palette="vlag", hue_norm=(-1, 1), edgecolor=".7",
#     height=10, sizes=(50, 250), size_norm=(-.2, .8),
# )

# # Tweak the figure to finalize
# g.set(xlabel="", ylabel="", aspect="equal")
# g.despine(left=True, bottom=True)
# g.ax.margins(.02)
# for label in g.ax.get_xticklabels():
#     label.set_rotation(90)
# for artist in g.legend.legendHandles:
#     artist.set_edgecolor(".7")
