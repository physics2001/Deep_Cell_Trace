import pandas as pd
import numpy as np
import h5py
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import scanpy as sc
import cospar as cs

n_top_genes=1000

LOG_DATA = [True, False]

SCALE_DATA = [True, False]

# Generate different preprocessing combination of data
for log in LOG_DATA: 
    for scale in SCALE_DATA: 
        # Load hemotapoesis data
        adata_orig=cs.datasets.hematopoiesis()

        # Preprocess data
        sc.pp.filter_genes(adata_orig, min_cells=100)
        sc.pp.filter_genes(adata_orig, min_counts=1)
        sc.pp.filter_cells(adata_orig, min_genes=1000)
        
        if log: 
            sc.pp.log1p(adata_orig)
            sc.pp.highly_variable_genes(adata_orig, n_top_genes=n_top_genes)
        else: 
            sc.pp.highly_variable_genes(adata_orig, n_top_genes=n_top_genes, flavor='seurat_v3')
        
        column_names = adata_orig.var['highly_variable']
        
        adata_orig = adata_orig[:, adata_orig.var['highly_variable']]
        
        df = adata_orig.to_df()
        
        if scale: 
            scaler = MinMaxScaler()
            df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
        
        # Save gene names
        column_names = df.columns.to_numpy()
        np.savetxt("{0}_Genes_Logged_{1}_Scaled_{2}_Column_Names.txt".format(n_top_genes, log, scale), column_names, fmt='%s')
        
        data = df.to_numpy()
        print(data.shape)
        
        NUM_EXAMPLES = data.shape[0]
        FEATURES = data.shape[1]

        # Split data in train, val, test set
        TRAIN_TEST_SPLIT = 0.2
        VAL_TEST_SPLIT = 0.5
        
        X_train, X_test, y_train, y_test = train_test_split(data, data, test_size=TRAIN_TEST_SPLIT, random_state=6482)
        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=VAL_TEST_SPLIT, random_state=6482)
        print(X_train.shape)
        print(X_val.shape)
        print(X_test.shape)
        print(y_train.shape)
        print(y_val.shape)
        print(y_test.shape)
        
        # Save data in compressed h5 format
        with h5py.File("{0}_Genes_Logged_{1}_Scaled_{2}_Data.h5".format(n_top_genes, log, scale), 'a') as FOB: 
            FOB.create_dataset("{0}_Genes_Logged_{1}_Scaled_{2}_Data".format(n_top_genes, log, scale), data=data, dtype='f')
            FOB.create_dataset("X_train", data=X_train, dtype='f')
            FOB.create_dataset("X_val", data=X_val, dtype='f')
            FOB.create_dataset("X_test", data=X_test, dtype='f')
            print("Keys: %s" % FOB.keys())
            FOB.close()
        
