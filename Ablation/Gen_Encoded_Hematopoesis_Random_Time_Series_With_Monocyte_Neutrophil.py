# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 10:30:26 2024

@author: Allen Zhang
"""

import pandas as pd
import numpy as np
import h5py
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import scanpy as sc
import cospar as cs
import joblib

adata_orig=cs.datasets.hematopoiesis()
    
sc.pp.filter_genes(adata_orig, min_cells=100)

sc.pp.filter_genes(adata_orig, min_counts=1)

sc.pp.filter_cells(adata_orig, min_genes=1000)

sc.pp.log1p(adata_orig)

sc.pp.highly_variable_genes(adata_orig, n_top_genes=1000)

column_names = adata_orig.var['highly_variable']

adata_orig = adata_orig[:, adata_orig.var['highly_variable']]

clone=pd.DataFrame(adata_orig.obsm["X_clone"].indices)
clone = clone.rename(columns={0: 'clone'})
time=pd.DataFrame(adata_orig.obs['time_info'])

cell_state=pd.DataFrame(adata_orig.obs['state_info'], columns=['state_info'])

print(time.shape)
print(time.columns)
print(clone.shape)
print(clone.columns)

with h5py.File('1000_Encoded_Genes_Scaled_Data.h5', 'a') as q:
    print("Keys: %s" % q.keys())
    X = list(q['1000_Encoded_Genes_Scaled_Data'])
    q.close()

X = np.array(X, dtype=np.float32)
df = pd.DataFrame(X)

df = pd.concat([df.reset_index(drop=True), time.reset_index(drop=True)], axis=1)

df = pd.concat([df.reset_index(drop=True), clone.reset_index(drop=True)], axis=1)

df = pd.concat([df.reset_index(drop=True), cell_state.reset_index(drop=True)], axis=1)

# time_series_indices_list = []
training_data_list_X = []
list_y = []

clones = df
clones_day2 = clones.loc[clones['time_info']=='2']
clones_day4 = clones.loc[clones['time_info']=='4']
clones_day6 = clones.loc[(clones['time_info']=='6') & (clones['state_info'].isin(['Monocyte', 'Neutrophil']))]

np.random.seed(23)

for j in range(3000): 
    i2 = np.random.choice(clones_day2.index.astype(int))
    i4 = np.random.choice(clones_day4.index.astype(int))
    i6 = np.random.choice(clones_day6.index.astype(int))
    time_serie = [df.iloc[i2, ~df.columns.isin(['clone', 'time_info', 'state_info'])].to_numpy(), 
                  df.iloc[i4, ~df.columns.isin(['clone', 'time_info', 'state_info'])].to_numpy(), 
                  df.iloc[i6, ~df.columns.isin(['clone', 'time_info', 'state_info'])].to_numpy()]
    list_y.append(df.loc[i6, 'state_info'])
    training_data_list_X.append(time_serie)

X = np.array(training_data_list_X, dtype=np.float32)
print(X.shape)
print(X.dtype)

# print(list_y.shape)
ohenc = OneHotEncoder(sparse=False)
y = np.array([list_y], dtype=str)
y = y.swapaxes(0, 1)
y = ohenc.fit_transform(y)
print(y.shape)
print(y)
inverted = ohenc.inverse_transform(y)
print(inverted.shape)
ohenc_filename = "ohenc_hemo_random.save"
joblib.dump(ohenc, ohenc_filename) 

with h5py.File('1000_Encoded_Genes_Scaled_Random_Time_Series_With_Monocyte_Neutrophil_with_orig_data.h5', 'a') as FOB: 
    FOB.create_dataset("X_test", data=X, dtype='f')
    FOB.create_dataset("y_test", data=y, dtype='f')

