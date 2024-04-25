# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 13:04:12 2024

@author: Allen Zhang
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from keras.models import load_model
import pandas as pd
import h5py

n_top_genes=1000

plt.figure(figsize=(12, 7.5))
pred_folder = 'Classification_On_Encoded_Data_Mon_Neu_Two_Classes/Features_75_Dropout_0.5_NumLayer_4_ModelType_GRU/'
y_test_pred = pd.read_csv(pred_folder+'y_test_predicted.csv', index_col=0)
y_test_pred_0 = y_test_pred.to_numpy(dtype=np.float32)[:, 0]
y_test = pd.read_csv(pred_folder+'y_test.csv', index_col=0)
y_test_0 = y_test.to_numpy(dtype=np.float32)[:, 0]
fpr, tpr, _ = metrics.roc_curve(y_test_0, y_test_pred_0)
auc = round(metrics.roc_auc_score(y_test_0, y_test_pred_0), 4)
plt.plot(fpr,tpr,label="Until Day 6, AUC="+str(auc), alpha=0.7, linewidth=3)

pred_folder = 'Classification_On_Encoded_Data_Mon_Neu_Two_Classes_Without_Day6/Features_75_Dropout_0.5_NumLayer_4_ModelType_GRU/'
y_test_pred = pd.read_csv(pred_folder+'y_test_predicted.csv', index_col=0)
y_test_pred_0 = y_test_pred.to_numpy(dtype=np.float32)[:, 0]
y_test = pd.read_csv(pred_folder+'y_test.csv', index_col=0)
y_test_0 = y_test.to_numpy(dtype=np.float32)[:, 0]
fpr, tpr, _ = metrics.roc_curve(y_test_0, y_test_pred_0)
auc = round(metrics.roc_auc_score(y_test_0, y_test_pred_0), 4)
plt.plot(fpr,tpr,label="Until Day 4, AUC="+str(auc), alpha=0.7, linewidth=3)

pred_folder = 'Classification_On_Encoded_Data_Mon_Neu_Two_Classes_Without_Day4_And_Day6/Features_75_Dropout_0.5_NumLayer_4_ModelType_GRU/'
y_test_pred = pd.read_csv(pred_folder+'y_test_predicted.csv', index_col=0)
y_test_pred_0 = y_test_pred.to_numpy(dtype=np.float32)[:, 0]
y_test = pd.read_csv(pred_folder+'y_test.csv', index_col=0)
y_test_0 = y_test.to_numpy(dtype=np.float32)[:, 0]
fpr, tpr, _ = metrics.roc_curve(y_test_0, y_test_pred_0)
auc = round(metrics.roc_auc_score(y_test_0, y_test_pred_0), 4)
plt.plot(fpr,tpr,label="Until Day 2, AUC="+str(auc), alpha=0.7, linewidth=3)

plt.title("ROC_of_Different_Hematopoiesis_Time_Series_Length")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positve Rate")

#add legend
plt.legend()
plt.savefig('Plots/ROC_of_Different_Hematopoiesis_Time_Series_Length_3Days.png')
plt.savefig('Plots/pdfs/ROC_of_Different_Hematopoiesis_Time_Series_Length_3Days.pdf')

