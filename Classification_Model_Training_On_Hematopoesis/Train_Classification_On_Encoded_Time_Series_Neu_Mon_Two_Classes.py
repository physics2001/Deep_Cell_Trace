# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 00:09:17 2022

@author: zhang
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import h5py
import tensorflow as tf
import os
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, TimeDistributed, Layer, Bidirectional, GRU, Activation
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.activations import relu, softmax
from contextlib import redirect_stdout
from keras.losses import mean_squared_error
from keras.metrics import SparseCategoricalAccuracy
import time
from keras.losses import mean_squared_error, categorical_crossentropy
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve


print(tf.config.list_physical_devices('GPU'))

n_top_genes_list = [1000] #100, 500, 5000, 2000, 
drop_out_list = [0.25, 0.5]
num_layer_list = [1, 2, 4]
model_type_list = ["LSTM", "GRU"]

RESULT_LIST = []
RESULT_COLUMNS = ['Num_Genes', 'Num_Features', 'Classification_Model_Type', 
                  'Num_Layers', 'Classification_Dropout', 'Accuracy', 'Categorical_Crossentropy']


for n_top_genes in n_top_genes_list: 

    with h5py.File('{0}_Encoded_Genes_Scaled_Time_Series_With_Monocyte_Neu-Mon_Neutrophil_11.h5'.format(n_top_genes), 'r') as f:
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
    
    def myfunc(row):
        return row[1] == 0

    bool_arr_train = np.array([myfunc(row) for row in y_train])
    bool_arr_val = np.array([myfunc(row) for row in y_val])
    bool_arr_test = np.array([myfunc(row) for row in y_test])
    
    X_train = X_train[bool_arr_train]
    X_val = X_val[bool_arr_val]
    X_test = X_test[bool_arr_test]
    y_train = y_train[bool_arr_train]
    y_val = y_val[bool_arr_val]
    y_test = y_test[bool_arr_test]
    
    print(X_train.shape)
    print(X_val.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_val.shape)
    print(y_test.shape)
    
    y_train = y_train[:, [0, 2]]
    y_val = y_val[:, [0, 2]]
    y_test = y_test[:, [0, 2]]
    print(y_train.shape)
    print(y_val.shape)
    print(y_test.shape)
    
    TIMESTEPS = X_train.shape[1]
    FEATURES = X_train.shape[2]
    TRAIN_TEST_SPLIT = 0.2
    VAL_TEST_SPLIT = 0.5
    NUM_CLASSES = y_test.shape[1]
    
    for DROP_OUT in drop_out_list: 
        for MODEL_TYPE in model_type_list: 
            for NUM_LAYER in num_layer_list: 
                folder = "Classification_On_Encoded_Data_Mon_Neu_Two_Classes_11/Features_{0}_Dropout_{1}_NumLayer_{2}_ModelType_{3}/".format(FEATURES, DROP_OUT, NUM_LAYER, MODEL_TYPE)
                filepath = "model"
                os.makedirs(folder)
                
                model = Sequential()
                for i in range(NUM_LAYER-1): 
                    if MODEL_TYPE == "LSTM": 
                        model.add(Bidirectional(LSTM(FEATURES, return_sequences=True, dropout=DROP_OUT), 
                                                input_shape=(TIMESTEPS, FEATURES), merge_mode='ave'))
                    elif MODEL_TYPE == "GRU": 
                        model.add(Bidirectional(GRU(FEATURES, return_sequences=True, dropout=DROP_OUT), 
                                                input_shape=(TIMESTEPS, FEATURES), merge_mode='ave'))
                if MODEL_TYPE == "LSTM": 
                    model.add(Bidirectional(LSTM(FEATURES, return_sequences=False, dropout=DROP_OUT), 
                                            input_shape=(TIMESTEPS, FEATURES), merge_mode='ave'))
                elif MODEL_TYPE == "GRU": 
                    model.add(Bidirectional(GRU(FEATURES, return_sequences=False, dropout=DROP_OUT), 
                                            input_shape=(TIMESTEPS, FEATURES), merge_mode='ave'))
                model.add(Dense(NUM_CLASSES, activation=softmax))
                model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'categorical_crossentropy'])
                
                with open(folder+'model_summary.txt', 'w') as f:
                    with redirect_stdout(f):
                        model.summary()
                
                model.summary()
                
                es = EarlyStopping(monitor='val_accuracy', min_delta=0.003, patience=100)
                
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
                
                y_test_df = pd.DataFrame(y_test)
                y_test_df.to_csv(folder+"y_test.csv")
                
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
                spcorr_val.to_csv(folder+"spearman_correlation_val_data.csv")
                pcorr_val.to_csv(folder+"pearson_correlation_val_data.csv")
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
                spcorr_test.to_csv(folder+"spearman_correlation_test_data.csv")
                pcorr_test.to_csv(folder+"pearson_correlation_test_data.csv")
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
                RESULT_DF.to_csv("Classification_Mon_Neu_Result_Two_Classes_11.csv")
