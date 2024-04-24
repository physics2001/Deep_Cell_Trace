# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 18:45:00 2022

@author: zhang
"""

import numpy as np
import pandas as pd
import h5py
from keras.models import load_model
import matplotlib.pyplot as plt

n_top_genes = 1000
dim1 = 500
dim2 = 100
dim3 = 75
alpha = 0.3


LOG_DATA = [True, False]

SCALE_DATA = [True, False]

model_hyperparameters_list = [[500, 100, 75, 0.3, 0.2], 
                              [500, 200, 50, 0.1, 0.3]]
for model_hyperparameters in model_hyperparameters_list: 
    dim1 = model_hyperparameters[0]
    dim2 = model_hyperparameters[1]
    dim3 = model_hyperparameters[2]
    alpha = model_hyperparameters[3]
    dropout = model_hyperparameters[4]
    for log in LOG_DATA: 
        for scale in SCALE_DATA: 
            folder = "Dimension_Reduction/{7}_Genes_Logged_{8}_Scaled_{9}_Data/E1_{0}_E2_{1}_BN_{2}_D1_{3}_D2_{4}_alpha_{5}_dropout_{6}/".format(dim1, dim2, dim3, dim2, dim1, alpha, dropout, n_top_genes, log, scale)
            
            encoder = load_model(folder+"encoder")
            
            with h5py.File("{0}_Genes_Logged_{1}_Scaled_{2}_Data.h5".format(n_top_genes, log, scale), 'a') as f: 
                X = list(f["{0}_Genes_Logged_{1}_Scaled_{2}_Data".format(n_top_genes, log, scale)])
    
            X = np.array(X)
            print(X.shape)
    
            Dim_Reduced_Data = encoder.predict(X)
    
            with h5py.File(folder+'{0}_Genes_Data_Encoded.h5'.format(n_top_genes), 'a') as q:
                # Print all root level object names (aka keys) 
                # these can be group or dataset names 
                print("Keys: %s" % q.keys())
                q.create_dataset(folder+'{0}_Genes_Data_Encoded'.format(n_top_genes), data=Dim_Reduced_Data, dtype='f')
                q.close()
