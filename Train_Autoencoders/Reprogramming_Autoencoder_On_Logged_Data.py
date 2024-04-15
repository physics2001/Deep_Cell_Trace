# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 21:11:06 2022

@author: zhang
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import h5py
import tensorflow as tf
import os
import sys
from keras.models import Model, load_model
from keras.layers import Dropout, Dense, Input, LeakyReLU, BatchNormalization, Lambda, concatenate as concat
from keras.callbacks import ModelCheckpoint, EarlyStopping
from contextlib import redirect_stdout
from keras import backend as K
from tensorflow import keras
from keras.losses import mean_squared_error

print(tf.config.list_physical_devices('GPU'))

RESULT_COLUMNS = ['Num_genes', 'Logged', 'Scaled', 'AE_dim1', 'AE_dim2', 
                  'AE_latent', 'Alpha', 'AE_P_Corr', 'AE_Sp_Corr', 'AE_MSE']

RESULT_LIST = []

n_top_genes=5000

LOG_DATA = [True]

SCALE_DATA = [False]

Scaled = 'Scaled'
Logged = 'Logged'

for log in LOG_DATA: 
    for scale in SCALE_DATA: 
        with h5py.File("Reprogramming_{0}_Genes_Logged_{1}_Scaled_{2}_Data.h5".format(n_top_genes, log, scale), 'r') as f:
            # Print all root level object names (aka keys) 
            # these can be group or dataset names 
            print("Keys: %s" % f.keys())
            
            X_train = list(f['X_train'])
            X_val = list(f['X_val'])
            X_test = list(f['X_test'])

        X_train = np.array(X_train)
        X_val = np.array(X_val)
        X_test = np.array(X_test)
        
        print(X_train.shape)
        print(X_val.shape)
        print(X_test.shape)
        
        NUM_EXAMPLES = X_train.shape[0]
        FEATURES = X_train.shape[1]
        
        # dim1 = 2048
        # dim2 = 512
        # dim3 = 64
        
        # function for AE, VAE and CVAE:
        def build_two_layer_network_model(X, dim1, dim2, latent_dim, alpha, drop_out, type = 'AE', n_y = FEATURES):
            ''' 
            building a two layer AE, VAE or CVAE
            input is X and output is the encoder, decoder, model and classifier
            '''
        
            n_inputs = X.shape[1]
            
            # concat label if CVAE:
            if type == 'CVAE':
                X = Input(shape=(n_inputs,))
                label = Input(shape=(n_y,))
                visible = concat([X, label])
            else:
                visible = Input(shape=(n_inputs,))
        
            # define encoder
            # encoder level 1
            e = Dropout(drop_out)(visible)
            e = Dense(dim1)(e)
            e = BatchNormalization()(e)
            e = LeakyReLU(alpha=alpha)(e)
            # encoder level 2
            e = Dense(dim2)(e)
            e = BatchNormalization()(e)
            e = LeakyReLU(alpha=alpha)(e)
            n_bottleneck = latent_dim
            bottleneck = Dense(n_bottleneck)(e)
        
            # bottleneck
            if type == 'AE':
                encoder = Model(visible, bottleneck, name="encoder")
                #encoder.summary()
                ppp=latent_dim
                visible2 = Input(shape=(ppp,))
        
            elif type == 'VAE' or type == 'CVAE':
                ''' finding mu and sigma '''
                mu      = Dense(latent_dim, name='latent_mu')(e)
                sigma   = Dense(latent_dim, name='latent_sigma')(e)
        
                if type == 'VAE':
                    def sampling(args, latent_dim= latent_dim):
                        z_mean, z_log_sigma = args
                        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=0.5)
                        return z_mean + K.exp(z_log_sigma/2) * epsilon
                    
                    z = Lambda(sampling)([mu, sigma])
                    #z = Dense(n_bottleneck)(z)
                    encoder = Model(visible, [mu, sigma, z], name='encoder')
                    # [mu, sigma, z]
                    visible2   = Input(shape=(latent_dim,))
        
        
                else:
                    def sampling(args, latent_dim= latent_dim):
                        z_mean, z_log_sigma = args
                        epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0., stddev=0.5)
                        return z_mean + K.exp(z_log_sigma/2) * epsilon
                    
                    # if CVAE 
                    z = Lambda(sampling, output_shape = (latent_dim, ))([mu, sigma])
                    #z = Dense(n_bottleneck)(z)
                    z= concat([z, label])
                    encoder = Model([X, label] ,[mu, sigma, z], name='encoder')
                    visible2 = z
                    # [mu, sigma, z]
        
            encoder.summary()
        
            # define decoder, level 1
            d = Dense(dim2)(visible2)
            d = BatchNormalization()(d)
            d = LeakyReLU(alpha=alpha)(d)
            # decoder level 2
            d = Dense(dim1)(d)
            d = BatchNormalization()(d)
            d = LeakyReLU(alpha=alpha)(d)
        
            # output layer
            output = Dense(n_inputs, activation='linear')(d)
        
            if type == 'CVAE':
                d_in = Input(shape=(latent_dim+n_y,))
                d_cvae = Dense(dim2)(d_in)
                d_cvae = BatchNormalization()(d_cvae)
                d_cvae = LeakyReLU()(d_cvae)
                # decoder level 2
                d_cvae = Dense(dim1)(d_cvae)
                d_cvae = BatchNormalization()(d_cvae)
                d_cvae = LeakyReLU()(d_cvae)
                cvae_output  = Dense(n_inputs, activation='linear')(d_cvae)
                decoder = Model(d_in, cvae_output)
        
            
            else:
                decoder = Model(visible2, output, name="decoder")
                #import pdb; pdb.set_trace()
                encoded = encoder(visible)
                # encoder.summary()
                # decoder.summary()
                decoded = decoder(encoded)
        
                
                model = Model(inputs=visible, outputs=decoded)
                
            if type == 'AE':
                model.compile(optimizer='adam', loss='mse', metrics=['mse'])
        
            elif type == 'VAE':
                model = Model(inputs =visible, name = 'vae_m', outputs = decoded)
        
                def vae_loss(x, x_decoded_mean):
                    xent_loss = keras.losses.binary_crossentropy(x, x_decoded_mean)
                    kl_loss = - 0.5 * K.sum(1 + sigma - K.square(mu) - K.exp(sigma), axis=-1)
                    return xent_loss + kl_loss
        
                model.compile(optimizer='adam', loss = vae_loss)
        
            elif type == 'CVAE':
                model = Model([X, label], output, name="decoder")
        
                def vae_loss(y_true, y_pred):
                    recon = K.sum(
                        K.binary_crossentropy(y_true, y_pred), axis=-1)
                    kl = 0.5 * K.sum(K.exp(sigma) + K.square(mu) - 1. - sigma, axis=-1)
                    return recon + kl
        
                def KL_loss(y_true, y_pred):
                    return(0.5 * K.sum(
                        K.exp(sigma) + K.square(mu) - 1. - sigma, axis=1))
        
                def recon_loss(y_true, y_pred):
                    return K.sum(
                        K.binary_crossentropy(y_true, y_pred), axis=-1)
        
                model.compile(
                    optimizer='adam', 
                    loss=vae_loss, 
                    metrics = [KL_loss, recon_loss])
            
            decoder.summary()
            with open(folder+'model_summary.txt', 'a') as f:
                with redirect_stdout(f):
                    model.summary()
                    encoder.summary()
                    decoder.summary()
                    
            
            model.summary()
            return model,encoder,decoder
        
        
        # model_hyperparameters_list = [[500, 100, 75, 0.3, 0.2]]
        # model_hyperparameters_list = [[500, 200, 50, 0.1, 0.3]]
        model_hyperparameters_list = [[4000, 2000, 1000, 0.3, 0.5], [3000, 1000, 400, 0.3, 0.2]]
        for model_hyperparameters in model_hyperparameters_list: 
            dim1 = model_hyperparameters[0]
            dim2 = model_hyperparameters[1]
            dim3 = model_hyperparameters[2]
            alpha = model_hyperparameters[3]
            dropout = model_hyperparameters[4]
            
            folder = "Dimension_Reduction/Reprogramming/{7}_Genes_Logged_{8}_Scaled_{9}_Data/E1_{0}_E2_{1}_BN_{2}_D1_{3}_D2_{4}_alpha_{5}_dropout_{6}/".format(dim1, dim2, dim3, dim2, dim1, alpha, dropout, n_top_genes, log, scale)
            
            os.makedirs(folder)
            
            autoencoder, encoder, decoder = build_two_layer_network_model(X_train, dim1, dim2, dim3, alpha, dropout)
            
            column_names = np.loadtxt("Reprogramming_{0}_Genes_Logged_{1}_Scaled_{2}_Column_Names.txt".format(n_top_genes, log, scale), dtype=str)
            
            
            # store best model in case overtrained
            cp = ModelCheckpoint(filepath=folder+"autoencoder",
                                  save_best_only=True,
                                  verbose=0,
                                  mode="auto")
            
            es = EarlyStopping(monitor='val_loss', patience=100, min_delta=5e-6)
            
            # Train the autoencoder
            history = autoencoder.fit(X_train, X_train, 
                                                epochs=10000,
                                                batch_size=1024,
                                                shuffle=True, 
                                                validation_data=(X_val, X_val), 
                                                callbacks=[cp, es])
            
            # save the encoder to file
            encoder.save(folder + 'encoder')
            decoder.save(folder + "decoder")
            
            # Draw history of loss during trainig
            plt.figure(figsize=(10, 8))
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.savefig(folder + 'history.png')
            
            # convert the history.history dict to a pandas DataFrame:     
            hist_df = pd.DataFrame(history.history) 
            
            # or save to csv: 
            hist_csv_file = 'history.csv'
            with open(folder+hist_csv_file, mode='w') as f:
                hist_df.to_csv(f)
            
            autoencoder = load_model(folder+"autoencoder")
            
            X_train_hat = autoencoder.predict(X_train)
            X_val_hat = autoencoder.predict(X_val)
            X_test_hat = autoencoder.predict(X_test)
            
            
            X_train_true_df = pd.DataFrame(X_train, columns=column_names)
            X_train_hat_df = pd.DataFrame(X_train_hat, columns=column_names)
            print(X_train_true_df.shape)
            print(X_train_hat_df.shape)
            spcorr_train = X_train_hat_df.corrwith(X_train_true_df, axis=1, method='spearman')
            pcorr_train = X_train_hat_df.corrwith(X_train_true_df, axis=1, method='pearson')
            print(spcorr_train)
            print(pcorr_train)
            avg_pcorr_train = np.mean(pcorr_train)
            avg_spcorr_train = np.mean(spcorr_train)
            spcorr_train.to_csv(folder+"spearman_correlation_train_data.csv")
            pcorr_train.to_csv(folder+"pearson_correlation_train_data.csv")
            
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
            
            X_val_df = pd.DataFrame(X_val, columns=column_names)
            X_val_hat_df = pd.DataFrame(X_val_hat, columns=column_names)
            print(X_val_df.shape)
            print(X_val_hat_df.shape)
            spcorr_val = X_val_hat_df.corrwith(X_val_df, method='spearman', axis=1)
            pcorr_val = X_val_hat_df.corrwith(X_val_df, method='pearson', axis=1)
            print(spcorr_val)
            print(pcorr_val)
            avg_pcorr_val = np.mean(pcorr_val)
            avg_spcorr_val = np.mean(spcorr_val)
            spcorr_val.to_csv(folder+"spearman_correlation_val_data.csv")
            pcorr_val.to_csv(folder+"pearson_correlation_val_data.csv")
            
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
            
            X_test_df = pd.DataFrame(X_test, columns=column_names)
            X_test_hat_df = pd.DataFrame(X_test_hat, columns=column_names)
            print(X_test_hat.shape)
            print(X_test_hat_df.shape)
            spcorr_test = X_test_hat_df.corrwith(X_test_df, method='spearman', axis=1)
            pcorr_test = X_test_hat_df.corrwith(X_test_df, method='pearson', axis=1)
            print(spcorr_test)
            print(pcorr_test)
            avg_pcorr_test = np.mean(pcorr_test)
            avg_spcorr_test = np.mean(spcorr_test)
            spcorr_test.to_csv(folder+"spearman_correlation_test_data.csv")
            pcorr_test.to_csv(folder+"pearson_correlation_test_data.csv")
            X_test_hat_df.to_csv(folder+"X_test_predicted.csv")
            
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
            
            avg_mse = np.average(mean_squared_error(X_test, X_test_hat))
            with open(folder+'mse.txt', 'w') as FOB:
                with redirect_stdout(FOB):
                    print(avg_mse)

            X_test_df = pd.DataFrame(X_test)
            X_test_hat_df = pd.DataFrame(X_test_hat)
            spcorr_test = X_test_hat_df.corrwith(X_test_df, method='spearman', axis=1)
            pcorr_test = X_test_hat_df.corrwith(X_test_df, method='pearson', axis=1)
            print(spcorr_test.shape)
            print(pcorr_test.shape)
            avg_pcorr_test = np.mean(pcorr_test)
            avg_spcorr_test = np.mean(spcorr_test)
            
            result = [n_top_genes, log, scale, dim1, dim2, dim3, 
                      alpha, avg_pcorr_test, avg_spcorr_test, avg_mse]
            
            print(result)
            
            RESULT_LIST.append(result)
            
RESULT_DF = pd.DataFrame(RESULT_LIST, columns=RESULT_COLUMNS)
# ALL_DF = pd.read_csv('Autoencoder_Result_Reprogramming_With_Different_Preprocessing.csv', index_col=0)
# ALL_DF = pd.concat([ALL_DF, RESULT_DF], ignore_index=True)
# ALL_DF.to_csv("Autoencoder_Result_With_Reprogramming_Different_Preprocessing.csv")
RESULT_DF.to_csv("Autoencoder_Result_With_Reprogramming_Different_Preprocessing_{0}_Genes.csv".format(n_top_genes))