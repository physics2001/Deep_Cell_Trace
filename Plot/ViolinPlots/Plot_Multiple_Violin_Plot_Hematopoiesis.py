# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 21:27:37 2023

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
# print(y_test_failed_pred.shape)
# print(y_test_reprogrammed_pred.shape)

gene_names = np.loadtxt('1000_Genes_Logged_True_Scaled_False_Column_Names.txt', dtype=str)

mon_df = pd.DataFrame(y_test_mon, columns=gene_names)
neu_df = pd.DataFrame(y_test_neu, columns=gene_names)
mon_pred_df = pd.DataFrame(y_test_mon_pred, columns=gene_names)
neu_pred_df = pd.DataFrame(y_test_neu_pred, columns=gene_names)
print(mon_df.shape)
print(neu_df.shape)
print(mon_pred_df.shape)
print(neu_pred_df.shape)
real_col = ['real'] * mon_df.shape[0]
predicted_col = ['predicted'] * mon_pred_df.shape[0]
data_type_col = real_col + predicted_col
data_type_col = np.array([data_type_col])
data_type_col = np.swapaxes(data_type_col, 0, 1)
data_type_df = pd.DataFrame(data_type_col, columns=['Data_Type'])

print(data_type_df.shape)

mon_all_df = pd.concat([mon_df, mon_pred_df], axis=0)
neu_all_df = pd.concat([neu_df, neu_pred_df], axis=0)

print(mon_all_df.shape)
print(neu_all_df.shape)

mon_all_df = mon_all_df.reset_index(drop=True)
neu_all_df = neu_all_df.reset_index(drop=True)

print(mon_all_df.shape)
print(neu_all_df.shape)

mon_all_df = pd.concat([mon_all_df, data_type_df], axis=1)
neu_all_df = pd.concat([neu_all_df, data_type_df], axis=1)

print(mon_all_df.shape)
print(neu_all_df.shape)

# genes = ['Cd55', 'Cryab', 'Spp1']

Interested_Genes_1 = ['Adgre1',
                        'Anpep',
                        'Arg2',
                        'Atp6v0d2',
                        'Bst1',
                        'Casp6',
                        'Ccl2',
                        'Ccl6',
                        'Ccr2',
                        'Cd33',
                        'Cd74' ,
                        'Cd84',
                        'Cd300c2',
                        'Clec4d',
                        'Clec4n',
                        'Ctsc',
                        'Ctsl',
                        'Dpep2',
                        'Dusp3',
                        'Fabp4',
                        'Fam20c',
                        'Fcer1g',
                        'Fth1' ,
                        'Gns' ,
                        'Gpnmb',
                        'H2-Aa',
                        'Itgb2',
                        'Lgals3',
                        'Lgmn',
                        'Lipa',
                        'Lrpap1' ,
                        'Ly6a',
                        'Lyz2',
                        'Mitf',
                        'Mmp12',
                        'Mrc1' ,
                        'Ms4a6d',
                        'mt-Atp6',
                        'mt-Co2',
                        'Myof',
                        'Nabp1',
                        'Pdxk',
                        'Pla2g7',
                        'Plxna1',
                        'Ppia',
                        'Prtn3',
                        'Rab7b',
                        'Ralgds',
                        'Rnf149',
                        'Rps27rt',
                        'S100a4',
                        'S100a6',
                        'S100a8',
                        'Sell',
                        'Sirpa',
                        'Socs3',
                        'Srgn',
                        'Tecpr1',
                        'Timp2',
                        'Tmem154',
                        'Tnfrsf1b',
                        'Trem2',
                        'Tubb2a',
                        'Vcan',
                        'Vim',
                        'Wfdc21',]

Interested_Genes_2 = ['AA467197',
                        'Adpgk',
                        'Alox5' ,
                        'Atp6v0a1',
                        'C5ar1',
                        'Camp',
                        'Casp6',
                        'Ccdc62',
                        'Ccl6',
                        'Ccl9',
                        'Ccnd2',
                        'Cd177',
                        'Cd300c2',
                        'Cdkn1a',
                        'Chil1',
                        'Chil3',
                        'Csf2rb2',
                        'Ctsg',
                        'Cybb',
                        'Cyp11a1',
                        'Ddhd1',
                        'Dmkn',
                        'Elane',
                        'Fabp5',
                        'Fam105a',
                        'Fcnb',
                        'Fth1',
                        'Gfi1',
                        'Gns',
                        'Gstm1',
                        'Hacd4',
                        'Hmox1',
                        'Irs2',
                        'Itgam',
                        'Itgb2',
                        'Klhdc4',
                        'Lcn2',
                        'Lgals3',
                        'Lipg',
                        'Lrpap1',
                        'Lsp1',
                        'Ltf',
                        'Ly6a',
                        'Lyz2',
                        'Mgl2',
                        'Mmp8',
                        'Mpo',
                        'Mt1',
                        'mt-Atp6',
                        'Myc',
                        'Ndst1',
                        'Nfkbia',
                        'Ngp',
                        'Nrp1',
                        'Pbx1',
                        'Pde2a',
                        'Pdxk',
                        'Pirb',
                        'Pde1c',
                        'Ppia',
                        'Prss57',
                        'Prtn3',
                        'Psap',
                        'Rnh1',
                        'Rps27rt',
                        'S100a8',
                        'S100a9',
                        'Serpinb10',
                        'Sgms2',
                        'Slpi',
                        'Spp1',
                        'Srgn',
                        'Syne1',
                        'Tecpr1',
                        'Tmem216',
                        'Tnfrsf1b',
                        'Trim24',
                        'Trim30a',
                        'Tusc1',
                        'Vim',
                        'Wfdc21', 
                        'Zfp36',]

Interested_Mon_DF_List_1 = []
for gene in Interested_Genes_1: 
    interested_gene_df = mon_all_df[["Data_Type", gene]]
    interested_gene_df.columns = ["Data_Type", "Expression"]
    interested_gene_df = interested_gene_df.assign(Genes = [gene] * interested_gene_df.shape[0])
    Interested_Mon_DF_List_1.append(interested_gene_df)

All_Mon_df_list = []

All_Mon_df_list.append(pd.concat(Interested_Mon_DF_List_1[0:17], axis=0))
All_Mon_df_list.append(pd.concat(Interested_Mon_DF_List_1[17:34], axis=0))
All_Mon_df_list.append(pd.concat(Interested_Mon_DF_List_1[34:50], axis=0))
All_Mon_df_list.append(pd.concat(Interested_Mon_DF_List_1[50:66], axis=0))

sns.set(rc={'figure.figsize':(16,16)})
sns.set(font_scale=1.3)
fig, axes = plt.subplots(nrows=4, ncols=1, sharex=False, sharey=True)
for i in list(range(4)): 
    sns.violinplot(
        x="Genes", y="Expression", hue="Data_Type", data=All_Mon_df_list[i], split=False, 
        inner=None, ax=axes[i]
        # scale="count", scale_hue=True, inner="stick", order=order, ax=axes[1]
    )
    axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=90, ha="right")
    
fig.tight_layout(rect=[0, 0, 0.8, 1])
for ax in axes: 
    ax.legend(loc="center left", bbox_to_anchor=(1.04, 0.5))

plt.savefig('Plots/Interesting_Expression_Distribution_Hematopoiesis_Mon_New.png')
plt.savefig('Plots/pdfs/Interesting_Expression_Distribution_Hematopoiesis_Mon_New.pdf')
plt.close()

Interested_Neu_DF_List_2 = []
for gene in Interested_Genes_2: 
    interested_gene_df = neu_all_df[["Data_Type", gene]]
    interested_gene_df.columns = ["Data_Type", "Expression"]
    interested_gene_df = interested_gene_df.assign(Genes = [gene] * interested_gene_df.shape[0])
    Interested_Neu_DF_List_2.append(interested_gene_df)

All_Neu_df_list = []

All_Neu_df_list.append(pd.concat(Interested_Neu_DF_List_2[0:21], axis=0))
All_Neu_df_list.append(pd.concat(Interested_Neu_DF_List_2[21:42], axis=0))
All_Neu_df_list.append(pd.concat(Interested_Neu_DF_List_2[42:62], axis=0))
All_Neu_df_list.append(pd.concat(Interested_Neu_DF_List_2[62:82], axis=0))

# sns.set(rc={'figure.figsize':(16,16)})
# sns.set(font_scale=2)
fig, axes = plt.subplots(nrows=4, ncols=1, sharex=False, sharey=True)
for i in list(range(4)): 
    sns.violinplot(
        x="Genes", y="Expression", hue="Data_Type", data=All_Neu_df_list[i], split=False, 
        inner=None, ax=axes[i]
        # scale="count", scale_hue=True, inner="stick", order=order, ax=axes[1]
    )
    axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=90, ha="right")
    

fig.tight_layout(rect=[0, 0, 0.8, 1])
for ax in axes: 
    ax.legend(loc="center left", bbox_to_anchor=(1.04, 0.5))

plt.savefig('Plots/Interesting_Expression_Distribution_Hematopoiesis_Neu_New.png')
plt.savefig('Plots/pdfs/Interesting_Expression_Distribution_Hematopoiesis_Neu_New.pdf')
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
