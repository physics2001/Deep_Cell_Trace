# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 00:36:28 2022

@author: zhang
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import h5py
import tensorflow as tf
import os
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, TimeDistributed, Layer, Bidirectional, GRU, Activation, Masking
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.activations import relu, softmax
from contextlib import redirect_stdout
from keras.losses import mean_squared_error
import time
from keras.losses import mean_squared_error, categorical_crossentropy
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve

print(tf.config.list_physical_devices('GPU'))

n_top_genes = 1000
drop_out_list = [0.25, 0.5]
num_layer_list = [2, 4]
model_type_list = ["LSTM", "GRU"]

LOG_DATA = [True]

SCALE_DATA = [False]

model_hyperparameters_list = [[500, 100, 75, 0.3, 0.2]]
# model_hyperparameters_list = [[4000, 2000, 1000, 0.3, 0.5], [3000, 1000, 400, 0.3, 0.2]]

cols = ['Day6', 'Day9', 'Day12', 'Day15', 'Day21', 'Day28']

RESULT_LIST = []
RESULT_COLUMNS = ['Num_Genes', 'Num_Features', 'Classification_Model_Type', 
                  'Num_Layers', 'Classification_Dropout', 'Accuracy', 
                  'Categorical_Crossentropy']

for model_hyperparameters in model_hyperparameters_list: 
    dim1 = model_hyperparameters[0]
    dim2 = model_hyperparameters[1]
    dim3 = model_hyperparameters[2]
    alpha = model_hyperparameters[3]
    dropout = model_hyperparameters[4]
    for log in LOG_DATA: 
        for scale in SCALE_DATA: 
            folder = "Dimension_Reduction/Reprogramming/{7}_Genes_Logged_{8}_Scaled_{9}_Data/E1_{0}_E2_{1}_BN_{2}_D1_{3}_D2_{4}_alpha_{5}_dropout_{6}/".format(dim1, dim2, dim3, dim2, dim1, alpha, dropout, n_top_genes, log, scale)
            

            with h5py.File(folder+'{0}_Genes_Data_Encoded_Time_Series_With_Class_Two_Classes.h5'.format(n_top_genes), 'r') as f:
                # Print all root level object names (aka keys) 
                # these can be group or dataset names 
                print("Keys: %s" % f.keys())
                
                X_train = list(f['X_train'])
                X_val = list(f['X_val'])
                X_test = list(f['X_test'])
                y_train = list(f['y_train'])
                y_val = list(f['y_val'])
                y_test = list(f['y_test'])

            X_train = np.array(X_train)
            X_val = np.array(X_val)
            X_test = np.array(X_test)
            y_train = np.array(y_train)
            y_val = np.array(y_val)
            y_test = np.array(y_test)
            
            print(X_train.shape)
            print(X_val.shape)
            print(X_test.shape)
            print(y_train.shape)
            print(y_val.shape)
            print(y_test.shape)
            
            TIMESTEPS = X_train.shape[1]
            FEATURES = X_train.shape[2]
            NUM_CLASSES = y_test.shape[1]

            for DROP_OUT in drop_out_list: 
                for MODEL_TYPE in model_type_list: 
                    for NUM_LAYER in num_layer_list: 
                        folder = "Classification_On_Encoded_Reprogramming_Data/{4}_Encoded_Genes_Two_Classes/Features_{0}_Dropout_{1}_NumLayer_{2}_ModelType_{3}/".format(FEATURES, DROP_OUT, NUM_LAYER, MODEL_TYPE, n_top_genes)
                        filepath = "model"
                        os.makedirs(folder)
                        
                        model = Sequential()
                        model.add(Masking(mask_value=0.0, input_shape=(TIMESTEPS, FEATURES)))
                        for i in range(NUM_LAYER-1): 
                            if MODEL_TYPE == "LSTM": 
                                model.add(Bidirectional(LSTM(FEATURES, return_sequences=True, dropout=DROP_OUT), 
                                                        merge_mode='ave'))
                            elif MODEL_TYPE == "GRU": 
                                model.add(Bidirectional(GRU(FEATURES, return_sequences=True, dropout=DROP_OUT), 
                                                        merge_mode='ave'))
                        if MODEL_TYPE == "LSTM": 
                            model.add(Bidirectional(LSTM(FEATURES, return_sequences=False, dropout=DROP_OUT), 
                                                    merge_mode='ave'))
                        elif MODEL_TYPE == "GRU": 
                            model.add(Bidirectional(GRU(FEATURES, return_sequences=False, dropout=DROP_OUT), 
                                                    merge_mode='ave'))
                        model.add(Dense(NUM_CLASSES, activation=softmax))
                        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_crossentropy'])
                
                        with open(folder+'model_summary.txt', 'w') as f:
                            with redirect_stdout(f):
                                model.summary()
                        
                        model.summary()
                        
                        es = EarlyStopping(monitor='val_loss', patience=100, min_delta=5e-6)
                        
                        # store best model in case overtrained
                        cp = ModelCheckpoint(filepath=folder+filepath,
                                             save_best_only=True,
                                             verbose=0,
                                             mode="auto")
                        start = time.time()
                        history = model.fit(X_train, y_train, batch_size=1024, epochs=100000, shuffle=True, 
                                  validation_data=(X_val, y_val), 
                                  callbacks=[es, cp])
                        end = time.time()
                        
                        with open(folder+'training_time.txt', 'a') as f:
                            with redirect_stdout(f):
                                print(end - start)
                        
                        plt.figure(figsize=(10, 8))
                        plt.plot(history.history['loss'])
                        plt.plot(history.history['val_loss'])
                        plt.title('model loss')
                        plt.ylabel('loss')
                        plt.xlabel('epoch')
                        plt.legend(['train', 'test'], loc='upper left')
                        plt.savefig(folder + 'history.png')
                        
                        model = load_model(folder+filepath)
                        y_train_hat = model.predict(X_train)
                        y_val_hat = model.predict(X_val)
                        y_test_hat = model.predict(X_test)
                        
                        y_train_df = pd.DataFrame(y_train)
                        y_train_hat_df = pd.DataFrame(y_train_hat)
                        spcorr_train = y_train_hat_df.corrwith(y_train_df, axis=1, method='spearman')
                        pcorr_train = y_train_hat_df.corrwith(y_train_df, axis=1, method='pearson')
                        spcorr_train.to_csv(folder+"spearman_correlation_train_data.csv")
                        pcorr_train.to_csv(folder+"pearson_correlation_train_data.csv")
                        
                        avg_pcorr_train = np.mean(pcorr_train)
                        avg_spcorr_train = np.mean(spcorr_train)
                        plt.figure(figsize=(10, 8))
                        plt.ylim((0,10))
                        plt.hist(spcorr_train, bins=20, range=(0, 1), density=True)
                        plt.title("Average Spearman Correlation = {0}".format(round(avg_spcorr_train, 4)))
                        plt.show()
                        plt.savefig(folder+"spearman_correlation_train_data_hist.png")
                        plt.figure(figsize=(10, 8))
                        plt.ylim((0,10))
                        plt.hist(pcorr_train, bins=20, range=(0, 1), density=True)
                        plt.title("Average Pearson Correlation = {0}".format(round(avg_pcorr_train, 4)))
                        plt.show()
                        plt.savefig(folder+"pearson_correlation_train_data_hist.png")
                        
                        
                        y_val_df = pd.DataFrame(y_val)
                        y_val_hat_df = pd.DataFrame(y_val_hat)
                        spcorr_val = y_val_hat_df.corrwith(y_val_df, method='spearman', axis=1)
                        pcorr_val = y_val_hat_df.corrwith(y_val_df, method='pearson', axis=1)
                        spcorr_train.to_csv(folder+"spearman_correlation_val_data.csv")
                        pcorr_train.to_csv(folder+"pearson_correlation_val_data.csv")
                        y_val_hat_df.to_csv(folder+"y_val_predicted.csv")
                        
                        avg_pcorr_val = np.mean(pcorr_val)
                        avg_spcorr_val = np.mean(spcorr_val)
                        plt.figure(figsize=(10, 8))
                        plt.ylim((0,10))
                        plt.hist(spcorr_val, bins=20, range=(0, 1), density=True)
                        plt.title("Average Spearman Correlation = {0}".format(round(avg_spcorr_val, 4)))
                        plt.show()
                        plt.savefig(folder+"spearman_correlation_val_data_hist.png")
                        plt.figure(figsize=(10, 8))
                        plt.ylim((0,10))
                        plt.hist(pcorr_val, bins=20, range=(0, 1), density=True)
                        plt.title("Average Pearson Correlation = {0}".format(round(avg_pcorr_val, 4)))
                        plt.show()
                        plt.savefig(folder+"pearson_correlation_val_data_hist.png")
                        
                        
                        y_test_df = pd.DataFrame(y_test)
                        y_test_hat_df = pd.DataFrame(y_test_hat)
                        spcorr_test = y_test_hat_df.corrwith(y_test_df, method='spearman', axis=1)
                        pcorr_test = y_test_hat_df.corrwith(y_test_df, method='pearson', axis=1)
                        spcorr_train.to_csv(folder+"spearman_correlation_test_data.csv")
                        pcorr_train.to_csv(folder+"pearson_correlation_test_data.csv")
                        y_test_hat_df.to_csv(folder+"y_test_predicted.csv")
                        
                        avg_pcorr_test = np.mean(pcorr_test)
                        avg_spcorr_test = np.mean(spcorr_test)
                        plt.figure(figsize=(10, 8))
                        plt.ylim((0,10))
                        plt.hist(spcorr_test, bins=20, range=(0, 1), density=True)
                        plt.title("Average Spearman Correlation = {0}".format(round(avg_spcorr_test, 4)))
                        plt.show()
                        plt.savefig(folder+"spearman_correlation_test_data_hist.png")
                        plt.figure(figsize=(10, 8))
                        plt.ylim((0,10))
                        plt.hist(pcorr_test, bins=20, range=(0, 1), density=True)
                        plt.title("Average Pearson Correlation = {0}".format(round(avg_pcorr_test, 4)))
                        plt.show()
                        plt.savefig(folder+"pearson_correlation_test_data_hist.png")
                        
                        y_test = y_test_df.to_numpy()
                        y_test_hat = y_test_hat_df.to_numpy()
                        
                        y_test_classes = np.argmax(y_test, axis=1)
                        y_test_hat_classes = np.argmax(y_test_hat, axis=1)
                        
                        cf_mat = confusion_matrix(y_test_classes, y_test_hat_classes)
                        acc_score = accuracy_score(y_test_classes, y_test_hat_classes)
                        test_loss = np.average(categorical_crossentropy(y_test, y_test_hat))
                        
                        with open(folder+'accuracy.txt', 'a') as f:
                            with redirect_stdout(f):
                                print(f'accuracy score: {acc_score}')
                                f.close()
                        
                        with open(folder+'confusion_matrix.txt', 'a') as f:
                            with redirect_stdout(f):        
                                print('Confusion matrix')
                                print(cf_mat)
                                f.close()
                        
                        RESULT_LIST.append([n_top_genes, FEATURES, MODEL_TYPE, 
                                            NUM_LAYER, DROP_OUT, acc_score, test_loss])
                        
RESULT_DF = pd.DataFrame(RESULT_LIST, columns=RESULT_COLUMNS)
RESULT_DF.to_csv("Classification_Reprogramming_Two_Classes_{0}_Genes_Result.csv".format(n_top_genes))

