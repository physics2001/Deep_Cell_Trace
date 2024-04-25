# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 21:58:26 2023

@author: zhang
"""

# import seaborn as sns
# import matplotlib.pyplot as plt
# sns.set(style="whitegrid", font_scale=1.5)

# tips = sns.load_dataset("tips")
# # here you add a new column with the two categorical data you want
# tips["sex_time"] = tips[["sex", "time"]].apply(lambda x: "_".join(x), axis=1)

# fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8, 10), 
#                          sharex=True, gridspec_kw=dict(height_ratios=(1, 3), hspace=0))

# # select the order you want:
# order=["Female_Lunch", "Male_Lunch", "Female_Dinner", "Male_Dinner"]

# sns.countplot(
#     data=tips, x="sex_time", hue="smoker", 
#     order=order,
#     ax=axes[0]
# )

# sns.violinplot(
#     x="sex_time", y="total_bill", hue="smoker", data=tips, 
#     split=True, scale="count", scale_hue=False, inner="stick",
#     order=order,
#     ax=axes[1]
# )
# axes[1].set_xticklabels(["Lunch (Female)", "Lunch (Male)", "Dinner (Female)", "Dinner (Male)"])
# axes[1].set_xlabel("Time (Sex)")
# axes[1].legend("")

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

with h5py.File('{0}_Genes_Reprogramming_Two_Classes_Data_Time_Series_With_Class.h5'.format(n_top_genes), 'a') as f: 
    print("Keys: %s" % f.keys())
    
    X_test = list(f['X_test'])
    class_info = list(f['y_test'])

X_test = np.array(X_test)
class_info = np.array(class_info)
y_test = X_test[:, 5, : ].copy()

# print(X_test.shape)
# print(class_info.shape)
# print(y_test.shape)

failed_indices = np.where(class_info == 0)[0]
reprogrammed_indices = np.where(class_info == 1)[0]
# print(failed_indices.shape)
# print(reprogrammed_indices.shape)

reg_folder = "Regression_On_Reprogramming_Data/1000_Genes_Two_Classes_Day28/Features_1000_Dropout_0.25_NumLayer2_LSTM/"

y_test_pred = pd.read_csv(reg_folder+"y_test_predicted.csv", index_col=0)

y_test_pred = y_test_pred.to_numpy()

y_test_failed = y_test[15000:, :]
y_test_reprogrammed = y_test[:15000, :]
# print(y_test_failed.shape)
# print(y_test_reprogrammed.shape)

y_test_failed_pred = y_test_pred[15000:, :]
y_test_reprogrammed_pred = y_test_pred[:15000, :]
# print(y_test_failed_pred.shape)
# print(y_test_reprogrammed_pred.shape)

gene_names = np.loadtxt('Reprogramming_1000_Genes_Logged_True_Scaled_False_Column_Names.txt', dtype=str)

failed_df = pd.DataFrame(y_test_failed, columns=gene_names)
reprogrammed_df = pd.DataFrame(y_test_reprogrammed, columns=gene_names)
failed_pred_df = pd.DataFrame(y_test_failed_pred, columns=gene_names)
reprogrammed_pred_df = pd.DataFrame(y_test_reprogrammed_pred, columns=gene_names)
print(failed_df.shape)
print(reprogrammed_df.shape)
print(failed_pred_df.shape)
print(reprogrammed_pred_df.shape)
real_col = ['real'] * 15000
predicted_col = ['predicted'] * 15000
data_type_col = real_col + predicted_col
data_type_col = np.array([data_type_col])
data_type_col = np.swapaxes(data_type_col, 0, 1)
data_type_df = pd.DataFrame(data_type_col, columns=['Data_Type'])

print(data_type_df.shape)

failed_all_df = pd.concat([failed_df, failed_pred_df], axis=0)
reprogrammed_all_df = pd.concat([reprogrammed_df, reprogrammed_pred_df], axis=0)

print(failed_all_df.shape)
print(reprogrammed_all_df.shape)

failed_all_df = failed_all_df.reset_index(drop=True)
reprogrammed_all_df = reprogrammed_all_df.reset_index(drop=True)

print(failed_all_df.shape)
print(reprogrammed_all_df.shape)

failed_all_df = pd.concat([failed_all_df, data_type_df], axis=1)
reprogrammed_all_df = pd.concat([reprogrammed_all_df, data_type_df], axis=1)

print(failed_all_df.shape)
print(reprogrammed_all_df.shape)

# genes = ['Cd55', 'Cryab', 'Spp1']

Interested_Genes_1 = ['Aspn',
                    'Capn6',
                    'Csrp2',
                    'Cyp26b1',
                    'Islr',
                    'Ism1',
                    'Lgr6',
                    'Lsp1',
                    'Meg3',
                    'Pdp1',
                    ]

Interested_Genes_2 = ['Plxdc2',
                    'Prox1',
                    'Ptx3',
                    'Runx1t1',
                    'Sord',
                    'Steap2',
                    'Tfdp2',
                    'Tgfbr3',
                    'Trf',
                    'Wnt5a']

Interested_Failed_DF_List_1 = []
for gene in Interested_Genes_1: 
    interested_gene_df = failed_all_df[["Data_Type", gene]]
    interested_gene_df.columns = ["Data_Type", "Expression"]
    interested_gene_df = interested_gene_df.assign(Genes = [gene] * interested_gene_df.shape[0])
    Interested_Failed_DF_List_1.append(interested_gene_df)
    
Interested_Failed_DF_List_2 = []
for gene in Interested_Genes_2: 
    interested_gene_df = failed_all_df[["Data_Type", gene]]
    interested_gene_df.columns = ["Data_Type", "Expression"]
    interested_gene_df = interested_gene_df.assign(Genes = [gene] * interested_gene_df.shape[0])
    Interested_Failed_DF_List_2.append(interested_gene_df)

All_Failed_df_1 = pd.concat(Interested_Failed_DF_List_1, axis=0)
All_Failed_df_2 = pd.concat(Interested_Failed_DF_List_2, axis=0)

sns.set(rc={'figure.figsize':(12,12)})
sns.set(font_scale=1.3)
fig, axes = plt.subplots(nrows=4, ncols=1, sharex=False, sharey=True)
sns.violinplot(
    x="Genes", y="Expression", hue="Data_Type", data=All_Failed_df_1, split=False, 
    inner=None, ax=axes[0]
    # scale="count", scale_hue=True, inner="stick", order=order, ax=axes[1]
)
# plt.tight_layout(rect=[0, 0, 0.8, 1])

sns.violinplot(
    x="Genes", y="Expression", hue="Data_Type", data=All_Failed_df_2, split=False, 
    inner=None, ax=axes[2]
    # scale="count", scale_hue=True, inner="stick", order=order, ax=axes[1]
)
# plt.tight_layout(rect=[0, 0, 0.8, 1])

Interested_Reprogrammed_DF_List_1 = []
for gene in Interested_Genes_1: 
    interested_gene_df = reprogrammed_all_df[["Data_Type", gene]]
    interested_gene_df.columns = ["Data_Type", "Expression"]
    interested_gene_df = interested_gene_df.assign(Genes = [gene] * interested_gene_df.shape[0])
    Interested_Reprogrammed_DF_List_1.append(interested_gene_df)
    
Interested_Reprogrammed_DF_List_2 = []
for gene in Interested_Genes_2: 
    interested_gene_df = reprogrammed_all_df[["Data_Type", gene]]
    interested_gene_df.columns = ["Data_Type", "Expression"]
    interested_gene_df = interested_gene_df.assign(Genes = [gene] * interested_gene_df.shape[0])
    Interested_Reprogrammed_DF_List_2.append(interested_gene_df)

All_Reprogrammed_df_1 = pd.concat(Interested_Reprogrammed_DF_List_1, axis=0)
All_Reprogrammed_df_2 = pd.concat(Interested_Reprogrammed_DF_List_2, axis=0)

# sns.set(rc={'figure.figsize':(40,6)})
# sns.set(rc={'figure.figsize':(16,16)})
# sns.set(font_scale=2)
sns.violinplot(
    x="Genes", y="Expression", hue="Data_Type", data=All_Reprogrammed_df_1, split=False, 
    inner=None, ax=axes[1]
    # scale="count", scale_hue=True, inner="stick", order=order, ax=axes[1]
)
# plt.tight_layout(rect=[0, 0, 0.8, 1])

sns.violinplot(
    x="Genes", y="Expression", hue="Data_Type", data=All_Reprogrammed_df_2, split=False, 
    inner=None, ax=axes[3]
    # scale="count", scale_hue=True, inner="stick", order=order, ax=axes[1]
)

axes[0].set_title('Failed Cells')
axes[2].set_title('Failed Cells')
axes[1].set_title('Reprogrammed Cells')
axes[3].set_title('Reprogrammed Cells')


fig.tight_layout(rect=[0, 0, 0.8, 1])
for ax in axes: 
    ax.legend(loc="center left", bbox_to_anchor=(1.04, 0.5))

# plt.tight_layout()

plt.savefig('Plots/Interesting_Expression_Distribution_Reprogramming_New.png')
plt.savefig('Plots/pdfs/Interesting_Expression_Distribution_Reprogramming_New.pdf')
plt.close()

# Interested_Genes = ['Stard8',
#                     'Stra6',
#                     'Trf',
#                     'Tspan8',
#                     'Tubb6',
#                     '2700094K13Rik',
#                     'Abi3bp',
#                     'Ankrd44',
#                     'Aspn',
#                     'Bmp3',
#                     'Capn6',
#                     'Cdca8',
#                     'Col4a2',
#                     'Csrp2',
#                     'Cyp26b1',
#                     'Epha7',
#                     'Hspa1b',
#                     'Islr',
#                     'Ism1',
#                     'Ifitm1',
#                     'Kcnj8',
#                     'Lgr6',
#                     'Lsp1',
#                     'Meg3',
#                     'Nr4a1',
#                     'Pcdh9',
#                     'Pdp1',
#                     'Plxdc2',
#                     'Prox1',
#                     'Ptx3',
#                     'Ramp3',
#                     'Runx1t1',
#                     'Sema5a',
#                     'Snhg18',
#                     'Sord',
#                     'Steap2',
#                     'Tfdp2',
#                     'Tgfbr3',
#                     'Tnnt3',
#                     'Trib2',
#                     'Wnt5a',
#                     'Zim1']
# Interested_Failed_DF_List = []
# for gene in Interested_Genes: 
#     interested_gene_df = failed_all_df[["Data_Type", gene]]
#     interested_gene_df.columns = ["Data_Type", "Expression"]
#     interested_gene_df = interested_gene_df.assign(Genes = [gene] * interested_gene_df.shape[0])
#     Interested_Failed_DF_List.append(interested_gene_df)

# All_Failed_df = pd.concat(Interested_Failed_DF_List, axis=0)

# print(All_Failed_df.shape)
# print(All_Failed_df.head())
# print(All_Failed_df.columns)

# sns.set(rc={'figure.figsize':(50,6)})
# sns.violinplot(
#     x="Genes", y="Expression", hue="Data_Type", data=All_Failed_df, split=True, 
#     inner=None
#     # scale="count", scale_hue=True, inner="stick", order=order, ax=axes[1]
# )

# plt.savefig('Plots/Failed_Interesting_Expression_Distribution_Reprogramming.png')
# plt.savefig('Plots/pdfs/Failed_Interesting_Expression_Distribution_Reprogramming.pdf')
# plt.close()

# Interested_Reprogrammed_DF_List = []
# for gene in Interested_Genes: 
#     interested_gene_df = reprogrammed_all_df[["Data_Type", gene]]
#     interested_gene_df.columns = ["Data_Type", "Expression"]
#     interested_gene_df = interested_gene_df.assign(Genes = [gene] * interested_gene_df.shape[0])
#     Interested_Reprogrammed_DF_List.append(interested_gene_df)

# All_Reprogrammed_df = pd.concat(Interested_Reprogrammed_DF_List, axis=0)

# print(All_Reprogrammed_df.shape)
# print(All_Reprogrammed_df.head())
# print(All_Reprogrammed_df.columns)

# sns.set(rc={'figure.figsize':(50,6)})
# sns.violinplot(
#     x="Genes", y="Expression", hue="Data_Type", data=All_Reprogrammed_df, split=True, 
#     inner=None
#     # scale="count", scale_hue=True, inner="stick", order=order, ax=axes[1]
# )

# plt.savefig('Plots/Reprogrammed_Interesting_Expression_Distribution_Reprogramming.png')
# plt.savefig('Plots/pdfs/Reprogrammed_Interesting_Expression_Distribution_Reprogramming.pdf')
# plt.close()


# for i in range(len(gene_names)): 
#     sns.set_context("talk", font_scale=1.1)
#     plt.figure(figsize=(12,8))
#     sns.violinplot(y=gene_names[i], 
#                    x='Data_Type', 
#                    data=failed_all_df)
#     sns.stripplot(y=gene_names[i], 
#                   x='Data_Type', 
#                   data=failed_all_df,
#                   color="black", edgecolor="gray")
#     plt.savefig('Plots/Violin_Plots/Failed_'+gene_names[i]+'_Expression_Distribution_Reprogramming.png')
#     plt.close()
    
# for i in range(len(gene_names)): 
#     sns.set_context("talk", font_scale=1.1)
#     plt.figure(figsize=(12,8))
#     sns.violinplot(y=gene_names[i], 
#                    x='Data_Type', 
#                    data=reprogrammed_all_df)
#     sns.stripplot(y=gene_names[i], 
#                   x='Data_Type', 
#                   data=reprogrammed_all_df,
#                   color="black", edgecolor="gray")
#     plt.savefig('Plots/Violin_Plots/Reprogrammed_'+gene_names[i]+'_Expression_Distribution_Reprogramming.png')
#     plt.close()
