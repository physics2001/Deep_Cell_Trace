# -*- coding: utf-8 -*-
"""
Created on Sat Nov 26 00:35:42 2022

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

with h5py.File('{0}_Genes_Logged_Time_Series_With_Monocyte_Neutrophil_Two_Classes.h5'.format(n_top_genes), 'a') as f: 
    print("Keys: %s" % f.keys())
    
    X_val = list(f['X_val'])
    X_test = list(f['X_test'])

X_val = np.array(X_val)
X_test = np.array(X_test)
print(X_val.shape)
print(X_test.shape)

days = ['Day2', 'Day4', 'Day6']

TIMESTEPS = X_test.shape[1]
FEATURES = X_test.shape[2]

gene_names = np.loadtxt('Reprogramming_1000_Genes_Logged_True_Scaled_False_Column_Names.txt', dtype=str)

for DROP_OUT in drop_out_list: 
    for MODEL_TYPE in model_type_list: 
        for NUM_LAYER in num_layer_list: 
            folder = "Classification_On_Not_Encoded_Data_Mon_Neu_Two_Classes/Features_{0}_Dropout_{1}_NumLayer_{2}_ModelType_{3}/".format(FEATURES, DROP_OUT, NUM_LAYER, MODEL_TYPE)
            filepath = "model"
            
            model = load_model(folder+filepath)

            # we use the first 100 training examples as our background dataset to integrate over
            explainer = shap.DeepExplainer(model, X_val)
            
            print('----------------------------------------------------')
            print('explainer ready')
            print('----------------------------------------------------')
            
            # explain the first 10 predictions
            # explaining each prediction requires 2 * background dataset size runs
            shap_values = explainer.shap_values(X_test)
            
            # shap_values = explainer.shap_values(X_test, check_additivity=False) # X_validate is 3d numpy.ndarray
            
            # shape_values = explainer(X_test)
            
            shap_values_save = np.array(shap_values)
            np.save('hemo_shap_values', shap_values_save)
            
            shap_values = shap_values_save
            # explainer.save('Reprogramming_Shap_Explainer.save')
            # joblib.dump(shape_values, 'shape_values_object.save')
            
            print('----------------------------------------------------')
            print('shap_values ready')
            print('----------------------------------------------------')
            
            for i in range(1, len(days)): 
                shap.summary_plot(
                    shap_values[0, :, i, :],
                    X_test[:, i, :],
                    feature_names=gene_names,
                    max_display=50,
                    plot_type='bar', show=False)
                # shap.save_html('Perturb_Reprogramming_Shap_Summary.html', p)
                plt.savefig("Perturb_Hemo_Shap_Summary_{0}.png".format(days[i]),dpi=150, bbox_inches='tight')
            
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
            
            