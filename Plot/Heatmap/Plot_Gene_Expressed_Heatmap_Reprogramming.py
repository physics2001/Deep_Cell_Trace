# -*- coding: utf-8 -*-
"""
Created on Wed Dec 14 20:56:38 2022

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

# calc percentage expressed
y_test_failed_percentage = np.count_nonzero(y_test_failed > 0.08, axis=0) / 15000
y_test_reprogrammed_percentage = np.count_nonzero(y_test_reprogrammed > 0.08, axis=0) / 15000
y_test_failed_pred_percentage = np.count_nonzero(y_test_failed_pred > 0.08, axis=0) / 15000
y_test_reprogrammed_pred_percentage = np.count_nonzero(y_test_reprogrammed_pred > 0.08, axis=0) / 15000

# calc average
y_test_failed_avgs = np.average(y_test_failed, axis=0)
y_test_reprogrammed_avgs = np.average(y_test_reprogrammed, axis=0)
print(y_test_failed_avgs.shape)
print(y_test_reprogrammed_avgs.shape)

y_test_failed_pred_avgs = np.average(y_test_failed_pred, axis=0)
y_test_reprogrammed_pred_avgs = np.average(y_test_reprogrammed_pred, axis=0)
print(y_test_failed_pred_avgs.shape)
print(y_test_reprogrammed_pred_avgs.shape)

real_failed = np.array([y_test_failed_avgs])
real_failed = np.swapaxes(real_failed, 0, 1)

predicted_failed = np.array([y_test_failed_pred_avgs])
predicted_failed = np.swapaxes(predicted_failed, 0, 1)

real_reprogrammed = np.array([y_test_reprogrammed_avgs])
real_reprogrammed = np.swapaxes(real_reprogrammed, 0, 1)

predicted_reprogrammed = np.array([y_test_reprogrammed_pred_avgs])
predicted_reprogrammed = np.swapaxes(predicted_reprogrammed, 0, 1)

print(real_failed.shape)
print(predicted_failed.shape)
print(real_reprogrammed.shape)
print(predicted_reprogrammed.shape)

gene_names = np.loadtxt('Reprogramming_1000_Genes_Logged_True_Scaled_False_Column_Names.txt', dtype=str)

failed_real_df = pd.DataFrame(real_failed, columns=['Expression_Level'], index=gene_names)
reprogrammed_real_df = pd.DataFrame(real_reprogrammed, columns=['Expression_Level'], index=gene_names)

failed_predicted_df = pd.DataFrame(predicted_failed, columns=['Expression_Level'], index=gene_names)
reprogrammed_predicted_df = pd.DataFrame(predicted_reprogrammed, columns=['Expression_Level'], index=gene_names)

failed_real_df = failed_real_df.assign(Gene_Names=gene_names)
reprogrammed_real_df = reprogrammed_real_df.assign(Gene_Names=gene_names)
failed_predicted_df = failed_predicted_df.assign(Gene_Names=gene_names)
reprogrammed_predicted_df = reprogrammed_predicted_df.assign(Gene_Names=gene_names)

failed_real_df = failed_real_df.assign(Percentage_Expressing=y_test_failed_percentage)
reprogrammed_real_df = reprogrammed_real_df.assign(Percentage_Expressing=y_test_reprogrammed_percentage)
failed_predicted_df = failed_predicted_df.assign(Percentage_Expressing=y_test_failed_pred_percentage)
reprogrammed_predicted_df = reprogrammed_predicted_df.assign(Percentage_Expressing=y_test_reprogrammed_pred_percentage)

print(failed_real_df)
print(reprogrammed_real_df)
print(failed_predicted_df)
print(reprogrammed_predicted_df)

Interested_Genes = ['Aspn',
                    'Capn6',
                    'Csrp2',
                    'Cyp26b1',
                    'Islr',
                    'Ism1',
                    'Lgr6',
                    'Lsp1',
                    'Meg3',
                    'Pdp1',
                    'Plxdc2',
                    'Prox1',
                    'Ptx3',
                    'Runx1t1',
                    'Sord',
                    'Steap2',
                    'Tfdp2',
                    'Tgfbr3',
                    'Trf',
                    'Wnt5a']

failed_real_df = failed_real_df[failed_real_df['Gene_Names'].isin(Interested_Genes)]
reprogrammed_real_df = reprogrammed_real_df[reprogrammed_real_df['Gene_Names'].isin(Interested_Genes)]
failed_predicted_df = failed_predicted_df[failed_predicted_df['Gene_Names'].isin(Interested_Genes)]
reprogrammed_predicted_df = reprogrammed_predicted_df[reprogrammed_predicted_df['Gene_Names'].isin(Interested_Genes)]

print(failed_real_df)
print(reprogrammed_real_df)
print(failed_predicted_df)
print(reprogrammed_predicted_df)

failed_real_df = failed_real_df.assign(Type=['Real_Failed_Expressions']*failed_real_df.shape[0])
reprogrammed_real_df = reprogrammed_real_df.assign(Type=['Real_Reprogrammed_Expressions']*reprogrammed_real_df.shape[0])
failed_predicted_df = failed_predicted_df.assign(Type=['Predicted_Failed_Expressions']*failed_predicted_df.shape[0])
reprogrammed_predicted_df = reprogrammed_predicted_df.assign(Type=['Predicted_Reprogrammed_Expressions']*reprogrammed_predicted_df.shape[0])

print(failed_real_df)
print(reprogrammed_real_df)
print(failed_predicted_df)
print(reprogrammed_predicted_df)

all_df = pd.concat([failed_real_df, failed_predicted_df, reprogrammed_real_df, reprogrammed_predicted_df], axis=0)
print(all_df.shape)
all_df.sort_values(by=['Gene_Names'])

# Draw each cell as a scatter point with varying size and color
sns.set(rc={'figure.figsize':(10,8)})
# fig, ax = plt.subplots(figsize=(20,12), constrained_layout=True)
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
plt.savefig("Plots/Heatmap_Reprogramming_New.png")
plt.savefig("Plots/pdfs/Heatmap_Reprogramming_New.pdf")

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
