# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 19:29:13 2022

@author: zhang
"""

import shap
import h5py
from keras.models import Sequential, load_model
import numpy as np
import matplotlib.pyplot as plt
import joblib

import tensorflow as tf    
tf.compat.v1.disable_v2_behavior()

n_top_genes = 1000
drop_out_list = [0.3]
num_layer_list = [1] #, 2
model_type_list = ["LSTM"]

with h5py.File('{0}_Genes_Reprogramming_Two_Classes_Data_Time_Series_With_Class.h5'.format(n_top_genes), 'a') as f: 
    print("Keys: %s" % f.keys())
    
    X_train = list(f['X_train'])
    X_test = list(f['X_test'])

X_train = np.array(X_train)
X_train = X_train[14500:15500, :, :]
X_test = np.array(X_test)
X_test = X_test[14500:15500, :, :]
print(X_train.shape)
print(X_test.shape)

TIMESTEPS = X_train.shape[1]
FEATURES = X_train.shape[2]

gene_names = np.loadtxt('Reprogramming_1000_Genes_Logged_True_Scaled_False_Column_Names.txt', dtype=str)

for DROP_OUT in drop_out_list: 
    for MODEL_TYPE in model_type_list: 
        for NUM_LAYER in num_layer_list: 
            folder = "Classification_On_Encoded_Reprogramming_Data/{4}_Genes_Not_Encoded_Two_Classes/Features_{0}_Dropout_{1}_NumLayer_{2}_ModelType_{3}/".format(FEATURES, DROP_OUT, NUM_LAYER, MODEL_TYPE, n_top_genes)
            filepath = "model"
            
            model = load_model(folder+filepath)

            # we use the first 100 training examples as our background dataset to integrate over
            explainer = shap.DeepExplainer(model, X_train)
            
            print('----------------------------------------------------')
            print('explainer ready')
            print('----------------------------------------------------')
            
            # explain the first 10 predictions
            # explaining each prediction requires 2 * background dataset size runs
            # shap_values = explainer.shap_values(X_test, )
            
            # shap_values = explainer.shap_values(X_test, check_additivity=False) # X_validate is 3d numpy.ndarray
            
            shape_values = explainer(X_test)
            
            explainer.save('Reprogramming_Shap_Explainer.save')
            joblib.dump(shape_values, 'shape_values_object.save')
            
            print('----------------------------------------------------')
            print('shap_values ready')
            print('----------------------------------------------------')
            
            # shap_values_save = np.array(shap_values)
            # np.save('reprogramming_shap_values_no_additivity.save', shap_values_save)
            # shap.initjs()
            # shap.summary_plot(
            #     shap_values[0], 
            #     X_test,
            #     feature_names=gene_names,
            #     max_display=50,
            #     plot_type='bar')
            
            # shap_values_save = np.array(shap_values)
            # np.save('reprogramming_shap_values.save', shap_values_save)
            
            # shap.initjs()
            # shap.summary_plot(
            #     shap_values[0], 
            #     X_test,
            #     feature_names=gene_names,
            #     max_display=50,
            #     plot_type='bar', show=False)
            # plt.savefig("Perturb_Reprogramming_Shap.png",dpi=150, bbox_inches='tight')
            # plt.clf()
            
            