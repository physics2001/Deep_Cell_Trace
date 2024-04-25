# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 16:01:01 2023

@author: zhang
"""

import numpy as np
import pandas as pd 

import matplotlib.pyplot as plt
import seaborn as sns

import os
import sys

# import single-cell packages
import scanpy as sc
import scanpy.external as sce
import scvelo as scv
import cellrank as cr

from typing import Optional, Sequence, Iterable
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from anndata import AnnData
import h5py

n_top_genes=1000

# set verbosity levels
sc.settings.verbosity = 2
cr.settings.verbosity = 2
scv.settings.verbosity = 3 


scv.settings.set_figure_params('scvelo', dpi_save=400, dpi=80, transparent=True, fontsize=20, color_map='viridis')
scv.settings.plot_prefix = ""


# should figures just be displayed or also saved?
save_figure = True

def predict_reprogramming_outcome(adata: AnnData, 
                                  day_key: str = 'reprogramming_day',
                                  day_selection: Optional[Sequence] = None,
                                  binary_label_key: str = 'reprogramming',
                                  predictor_key: str = 'reprogramming_probability', 
                                  random_state: int = 0, 
                                  figsize: Optional[Iterable[float]] = None, 
                                  lw: float = 2, 
                                  save: Optional[str] = None):
    """
    Predict reprogramming outcome for a subset of labelled cells given fate probabilities. 
    
    Given a subset of days from the reprogramming timecourse, treat the clonal information as ground-truth
    binary labels in a classification task in which we predict reprogramming outcome based on fate probabilities 
    using logistic regression. Data is subsampled so that sucessful/dead-end classes are balanced and split
    into train/test sets. Classification is evaluated using ROC curves. Note that cells used to define the terminal
    states are always removed from training/test sets. 
    
    Parameters
    ----
    adata
        Annotated data matrix.
    day_key
        Key from `.obs` where time-point information can be found.
    day_selection
        Lists of strings representing days that the data should be restricted to. 
    binary_label_key
        Key from `.obs` where the ground truth labels from CellTagging can be found. 
    predictor_key
        Key from `.obs` where the fate probability of sucessful reprogramming can be found.
    random_state
        Random seed used to subsample data.
    figzise
        Size of the overall figure
    save
        Path where figure is to be saved
        
    Returns
    ---
    Noting, just plots the ROC curve. 
    """
    
    # Restrict to cells from selected days that we have labels for
    # Exclude the cells that were used to define the terminal states
    if 'terminal_states' not in adata.obs.keys():
        raise ValueError('Compute terminal states first. ')
    terminal_mask = np.sum(adata.obs['terminal_states'].isna())
    label_mask = ~adata.obs[binary_label_key].isna()
    _mask = np.logical_and(terminal_mask, label_mask)
        
    if day_selection is None:
        day_selection = adata.obs[day_key].cat.categories
        
    # get the corresponding colors
    # _colors = adata.uns[f"{day_key}_colors"]
    # colors = [_colors[np.where(np.in1d(adata.obs[day_key].cat.categories, day))[0][0]] 
    #                   for day in day_selection]
    
    colors = ["cornflowerblue", "royalblue", "mediumblue"]
    
    # initialise figure and axis object
    fig = plt.figure(None, figsize)
    ax = fig.add_subplot(111)
        
    # loop over days
    for day, color in zip(day_selection, colors):
        
        # restrict to current day
        day_mask = np.in1d(adata.obs[day_key], day)
        mask = np.logical_and(_mask, day_mask)
        print(f"\nThis selects {np.sum(mask)} cells from day {day}." )

        # For these cells, get data matrices
        data = sc.get.obs_df(adata[mask], keys=[binary_label_key, predictor_key])
        X = data[predictor_key].values[:, None]
        y = np.array(data[binary_label_key].values) == 'True'
        assert(X.shape[0] == np.sum(mask)), "Shape mismatch between X and number of cells. "

        # Check for inbalanced classes
        print(f"There are {100* np.sum(y == True) / len(y):.2f}% positive examples. Correcting for class inbalance. ")

        # Correct for class inbalance
        np.random.seed(random_state)

        # split the data into positive and negative examples
        positive_mask = y
        X_positive, y_positive = X[positive_mask], y[positive_mask]
        X_negative, y_negative = X[~positive_mask], y[~positive_mask]

        # subsample the negative examples
        neg_ixs = np.random.choice(range(len(y_negative)), size=len(y_positive), replace=False)
        X_negative_sub, y_negative_sub = X_negative[neg_ixs], y_negative[neg_ixs]

        # concatenate positive and subsampled negative examples
        X_balanced = np.concatenate((X_positive, X_negative_sub))
        y_balanced = np.concatenate((y_positive, y_negative_sub))

        # assert that classes are balanced now
        assert(np.sum(y_balanced == True) / len(y_balanced) == 0.5), "Class inbalance could not be fixed. "

        # Split into training/test sets
        X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, random_state=0, test_size=0.4)
        n_data = X_train.shape[0] + X_test.shape[0]
        print(f"There are {X_train.shape[0]}/{n_data} training examples and {X_test.shape[0]}/{n_data} testing examples. ")

        # fit logistic regression classifier and show ROC on test set
        clf = LogisticRegression(random_state=random_state).fit(X_train, y_train)
        metrics.plot_roc_curve(clf, X_test, y_test, ax=ax, name=f"CellRank day {day}", lw=lw, color=color)
    
    
    folder = "Dimension_Reduction/Reprogramming/1000_Genes_Logged_True_Scaled_False_Data/E1_500_E2_100_BN_75_D1_100_D2_500_alpha_0.3_dropout_0.2/"

    with h5py.File(folder+'{0}_Genes_Data_Encoded_Time_Series_With_Class_Two_Classes.h5'.format(n_top_genes), 'r') as f:
        # Print all root level object names (aka keys) 
        # these can be group or dataset names 
        print("Keys: %s" % f.keys())
        y_test = list(f['y_test'])
    
    y_test = np.array(y_test)
    print(y_test.shape)
    y_test_0 = y_test[:, 0]
    
    
    new_colors = ["lightcoral", "indianred", "brown"]
    
    pred_folder = 'Classification_On_Encoded_Reprogramming_Data/1000_Encoded_Genes_Two_Classes_Without_15_21_28/Features_75_Dropout_0.25_NumLayer_2_ModelType_LSTM/'
    y_test_pred = pd.read_csv(pred_folder+'y_test_predicted.csv', index_col=0)
    y_test_pred_0 = y_test_pred.to_numpy(dtype=np.float32)[:, 0]
    fpr, tpr, _ = metrics.roc_curve(y_test_0, y_test_pred_0)
    auc = round(metrics.roc_auc_score(y_test_0, y_test_pred_0), 4)
    ax.plot(fpr,tpr,label="LSTM Day 12, (AUC = "+str(auc)+")", alpha=0.7, linewidth=3, color=new_colors[0])
    
    pred_folder = 'Classification_On_Encoded_Reprogramming_Data/1000_Encoded_Genes_Two_Classes_Without_21_28/Features_75_Dropout_0.25_NumLayer_2_ModelType_LSTM/'
    y_test_pred = pd.read_csv(pred_folder+'y_test_predicted.csv', index_col=0)
    y_test_pred_0 = y_test_pred.to_numpy(dtype=np.float32)[:, 0]
    fpr, tpr, _ = metrics.roc_curve(y_test_0, y_test_pred_0)
    auc = round(metrics.roc_auc_score(y_test_0, y_test_pred_0), 4)
    ax.plot(fpr,tpr,label="LSTM Day 15, (AUC = "+str(auc)+")", alpha=0.7, linewidth=3, color=new_colors[1])
    
    pred_folder = 'Classification_On_Encoded_Reprogramming_Data/1000_Encoded_Genes_Two_Classes_Without_28/Features_75_Dropout_0.25_NumLayer_2_ModelType_LSTM/'
    y_test_pred = pd.read_csv(pred_folder+'y_test_predicted.csv', index_col=0)
    y_test_pred_0 = y_test_pred.to_numpy(dtype=np.float32)[:, 0]
    fpr, tpr, _ = metrics.roc_curve(y_test_0, y_test_pred_0)
    auc = round(metrics.roc_auc_score(y_test_0, y_test_pred_0), 4)
    ax.plot(fpr,tpr,label="LSTM Day 21, (AUC = "+str(auc)+")", alpha=0.7, linewidth=3, color=new_colors[2])
    
    # pred_folder = 'Classification_On_Encoded_Reprogramming_Data/1000_Encoded_Genes_Two_Classes/Features_75_Dropout_0.25_NumLayer_2_ModelType_LSTM/'
    # y_test_pred = pd.read_csv(pred_folder+'y_test_predicted.csv', index_col=0)
    # y_test_pred_0 = y_test_pred.to_numpy(dtype=np.float32)[:, 0]
    # fpr, tpr, _ = metrics.roc_curve(y_test_0, y_test_pred_0)
    # auc = round(metrics.roc_auc_score(y_test_0, y_test_pred_0), 4)
    # ax.plot(fpr,tpr,label="LSTM Day 28, AUC="+str(auc), alpha=0.7, linewidth=3)
    
    # plt.title("ROC_of_Different_Reprogramming_Time_Series_Length")
    # plt.xlabel("False Positive Rate")
    # plt.ylabel("True Positve Rate")
    # add legend
    ax.legend(bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0, fontsize=8)
    fig.tight_layout()
    plt.savefig('Plots/ROC_of_Different_Reprogramming_Time_Series_Length_With_Cellrank.png')
    plt.savefig('Plots/pdfs/ROC_of_Different_Reprogramming_Time_Series_Length_With_Cellrank.pdf')
    plt.close()

    
    # plt.legend(bbox_to_anchor=(1.05, 1.)) 
    
    # if save is not None: 
    #     plt.savefig(save)
    # plt.show()

adata = sc.read_h5ad("adata.h5ad")

predict_reprogramming_outcome(adata, day_selection=['12', '15', '21'], lw=3, figsize=(10,6), 
                              save="Plots/roc_curves.pdf" if save_figure else None)

