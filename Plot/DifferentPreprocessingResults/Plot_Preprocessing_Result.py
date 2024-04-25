# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 21:36:12 2022

@author: zhang
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

different_preprocessing = pd.read_csv('Results/Regression_On_Encoded_Data_With_Different_Preprocessing_Result.csv', index_col=0)

different_preprocessing_different_RNN = different_preprocessing.loc[(different_preprocessing['AE_dim2'] == 100) & \
                                                          (different_preprocessing['Num_Layers_BRNN'] == 2)]

different_preprocessing_different_RNN['Preprocessing Methods'] = \
['logged and scaled', 'logged and scaled', 
 'logged and not scaled', 'logged and not scaled', 
 'not logged and scaled', 'not logged and scaled', 
 'not logged and not scaled', 'not logged and not scaled']

plt.figure(figsize=(16, 9))
plt.ylim((0, 1.1))
ax=sns.barplot(data=different_preprocessing_different_RNN, x='Preprocessing Methods', y="Reg_P_Corr", hue="Reg_Model_Type")
for container in ax.containers:
    ax.bar_label(container)
plt.ylabel('Pearson Correlation Between Encoded Data and Regression Predicted Data')
plt.xlabel('Preprocessing Methods Used Before Autoencoder')
plt.title('Effect of Different RNN Type and Different Preprocessing Methods On Regression Results')
# plt.show()
plt.savefig('Plots/Effect of Different RNN Type and Different Preprocessing Methods On Regression Results.png')
plt.savefig('Plots/pdfs/Effect of Different RNN Type and Different Preprocessing Methods On Regression Results.pdf')


different_preprocessing_different_AE = different_preprocessing.loc[(different_preprocessing['Reg_Model_Type'] == 'LSTM') & \
                                                          (different_preprocessing['Num_Layers_BRNN'] == 2)]

for i, row in different_preprocessing_different_AE.iterrows(): 
    val = 'dim1=' + str(row['AE_dim1'])
    val += ', dim2=' + str(row['AE_dim2'])
    val += ', dim3=' + str(row['AE_latent'])
    val += ', alpha=' + str(row['Alpha'])
    different_preprocessing_different_AE.loc[i, 'Different AE'] = val

different_preprocessing_different_AE['Preprocessing Methods'] = \
['logged and scaled',  'logged and not scaled', 'not logged and scaled', 'not logged and not scaled', 
 'logged and scaled', 'logged and not scaled', 'not logged and scaled', 'not logged and not scaled']

plt.figure(figsize=(16, 9))
plt.ylim((0, 1.1))
ax=sns.barplot(data=different_preprocessing_different_AE, x='Preprocessing Methods', y="Reg_P_Corr", hue="Different AE")
for container in ax.containers:
    ax.bar_label(container)
plt.ylabel('Pearson Correlation Between Encoded Data and Regression Predicted Data')
plt.xlabel('Preprocessing Methods Used Before Autoencoder')
plt.title('Effect of Different Autoencoder and Different Preprocessing Methods On Regression Results')
# plt.show()
plt.savefig('Plots/Effect of Different Autoencoder and Different Preprocessing Methods On Regression Results.png')
plt.savefig('Plots/pdfs/Effect of Different Autoencoder and Different Preprocessing Methods On Regression Results.pdf')


different_AE = pd.read_csv('Results/Autoencoder_Result_With_Different_Preprocessing.csv', index_col=0)

for i, row in different_AE.iterrows(): 
    val = 'dim1=' + str(row['AE_dim1'])
    val += ', dim2=' + str(row['AE_dim2'])
    val += ', dim3=' + str(row['AE_latent'])
    val += ', alpha=' + str(row['Alpha'])
    different_AE.loc[i, 'Different AE'] = val

different_AE['Preprocessing Methods'] = \
['logged and scaled',  'logged and not scaled', 'not logged and scaled', 'not logged and not scaled', 
 'logged and scaled', 'logged and not scaled', 'not logged and scaled', 'not logged and not scaled']

plt.figure(figsize=(16, 9))
plt.ylim((0, 1.1))
ax=sns.barplot(data=different_AE, x='Preprocessing Methods', y="AE_P_Corr", hue="Different AE")
for container in ax.containers:
    ax.bar_label(container)
plt.ylabel('Pearson Correlation Between Autoencoder Output and Actual Data')
plt.xlabel('Preprocessing Methods Used Before Autoencoder')
plt.title('Correlation of Different Autoencoder Output and Actual Data')
# plt.show()
plt.savefig('Plots/Correlation of Different Autoencoder Output and Actual Data.png')
plt.savefig('Plots/pdfs/Correlation of Different Autoencoder Output and Actual Data.pdf')
