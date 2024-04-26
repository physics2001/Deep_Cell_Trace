import pandas as pd
import numpy as np
import h5py
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import scanpy as sc
import cospar as cs
import joblib

# Load Hematopoesis Data and preprocess it
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

with h5py.File('1000_Genes_Logged_True_Scaled_False_Data.h5', 'a') as q:
    print("Keys: %s" % q.keys())
    X = list(q['1000_Genes_Logged_True_Scaled_False_Data'])
    q.close()

# Generate time series based on clone amd time information
X = np.array(X, dtype=np.float32)
df = pd.DataFrame(X)

df = pd.concat([df.reset_index(drop=True), time.reset_index(drop=True)], axis=1)

df = pd.concat([df.reset_index(drop=True), clone.reset_index(drop=True)], axis=1)

df = pd.concat([df.reset_index(drop=True), cell_state.reset_index(drop=True)], axis=1)

training_data_list_X = []
list_y = []
for i in range(5864): 
    clones = df.loc[df['clone'] == i]
    clones_day2 = clones.loc[clones['time_info']=='2']
    clones_day4 = clones.loc[clones['time_info']=='4']
    clones_day6 = clones.loc[(clones['time_info']=='6') & (clones['state_info'].isin(['Monocyte', 'Neutrophil']))]
    for i2, row2 in clones_day2.iterrows(): 
        for i4, row4 in clones_day4.iterrows():
            for i6, row6 in clones_day6.iterrows(): 
                time_serie = [df.iloc[i2, ~df.columns.isin(['clone', 'time_info', 'state_info'])].to_numpy(), 
                              df.iloc[i4, ~df.columns.isin(['clone', 'time_info', 'state_info'])].to_numpy(), 
                              df.iloc[i6, ~df.columns.isin(['clone', 'time_info', 'state_info'])].to_numpy()]
                list_y.append(df.loc[i6, 'state_info'])
                training_data_list_X.append(time_serie)

X = np.array(training_data_list_X, dtype=np.float32)
print(X.shape)
print(X.dtype)

# One Hote encode different classes
ohenc = OneHotEncoder(sparse=False)
y = np.array([list_y], dtype=str)
y = y.swapaxes(0, 1)
y = ohenc.fit_transform(y)
print(y.shape)
print(y)
inverted = ohenc.inverse_transform(y)
print(inverted.shape)
ohenc_filename = "ohenc_hemo_two_classes.save"
joblib.dump(ohenc, ohenc_filename) 

# Generate train, val, test data
TRAIN_TEST_SPLIT = 0.2
VAL_TEST_SPLIT = 0.5
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TRAIN_TEST_SPLIT, random_state=6482)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=VAL_TEST_SPLIT, random_state=6482)
print(X_train.shape)
print(X_val.shape)
print(X_test.shape)
print(y_train.shape)
print(y_val.shape)
print(y_test.shape)

# Save data in compressed h5 format
with h5py.File('1000_Genes_Logged_Time_Series_With_Monocyte_Neutrophil_Two_Classes.h5', 'a') as FOB: 
    FOB.create_dataset("X", data=X, dtype='f')
    FOB.create_dataset("y", data=y, dtype='f')
    FOB.create_dataset("X_train", data=X_train, dtype='f')
    FOB.create_dataset("X_val", data=X_val, dtype='f')
    FOB.create_dataset("X_test", data=X_test, dtype='f')
    FOB.create_dataset("y_train", data=y_train, dtype='f')
    FOB.create_dataset("y_val", data=y_val, dtype='f')
    FOB.create_dataset("y_test", data=y_test, dtype='f')

