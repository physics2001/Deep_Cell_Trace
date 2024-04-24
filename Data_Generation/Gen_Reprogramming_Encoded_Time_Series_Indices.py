# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 23:09:39 2022

@author: zhang
"""

import pandas as pd
import numpy as np
import h5py
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import scanpy as sc
import cospar as cs
import joblib
from collections import defaultdict

# n_top_genes = 5000
n_top_genes = 1000

adata_orig=cs.datasets.reprogramming()
    
sc.pp.filter_genes(adata_orig, min_cells=100)

sc.pp.filter_genes(adata_orig, min_counts=1)

sc.pp.filter_cells(adata_orig, min_genes=1000)

sc.pp.log1p(adata_orig)

sc.pp.highly_variable_genes(adata_orig, n_top_genes=n_top_genes)

column_names = adata_orig.var['highly_variable']

adata_orig = adata_orig[:, adata_orig.var['highly_variable']]

Total=adata_orig.obs

LOG_DATA = [True]

SCALE_DATA = [False]

model_hyperparameters_list = [[500, 100, 75, 0.3, 0.2]]
# model_hyperparameters_list = [[4000, 2000, 1000, 0.3, 0.5], [3000, 1000, 400, 0.3, 0.2]]

def neg(a): 
    count = 0
    for i in a: 
        if i < 0: 
            count += 1
    return count

def def_value():
    return [-1]

print('Here')

for model_hyperparameters in model_hyperparameters_list: 
    print('Inside first loop')
    dim1 = model_hyperparameters[0]
    dim2 = model_hyperparameters[1]
    dim3 = model_hyperparameters[2]
    alpha = model_hyperparameters[3]
    dropout = model_hyperparameters[4]
    
    for log in LOG_DATA: 
        print('Inside second loop')
        for scale in SCALE_DATA: 
            print('Inside third loop')
            folder = "Dimension_Reduction/Reprogramming/{7}_Genes_Logged_{8}_Scaled_{9}_Data/E1_{0}_E2_{1}_BN_{2}_D1_{3}_D2_{4}_alpha_{5}_dropout_{6}/".format(dim1, dim2, dim3, dim2, dim1, alpha, dropout, n_top_genes, log, scale)
            
            with h5py.File(folder+'{0}_Genes_Data_Encoded.h5'.format(n_top_genes), 'a') as q:
                print("Keys: %s" % q.keys())
                X = list(q[folder+'{0}_Genes_Data_Encoded'.format(n_top_genes)])
                q.close()
            
            X = np.array(X, dtype=np.float32)
            
            df = pd.DataFrame(X)
            
            df = pd.concat([df, Total], axis=1)
            columns = Total.columns.tolist()
            
            day0_and_day3_barcodes = df[['barcode_day0', 'barcode_day3']].value_counts(ascending=True).reset_index(name='count')
            
            total_combos = []
            for j, barcodes in day0_and_day3_barcodes.iterrows(): 
                barcode0 = barcodes['barcode_day0'] 
                barcode3 = barcodes['barcode_day3'] 
                if (barcode0 == 'NA') or (barcode3 == 'NA'): 
                    continue
            # for barcode0 in day0_barcodes: 
            #     for barcode3 in day3_barcodes: 
                clone = df.loc[(df['barcode_day0']==barcode0) & (df['barcode_day3']==barcode3)]
                d = defaultdict(def_value)
                for i, row in clone.iterrows(): 
                    day = row['time_info']
                    day = int(day[3:])
                    # if day == 15 or day == 21 or day == 28:
                    #     if df.loc[i, 'barcode_day13'] == 'NA': 
                    #         continue
                    
                    d[day].append(int(i))
                    # d[int(day[3:])].append(df.iloc[int(i), ~df.columns.isin(columns)].to_numpy())
            #     if len(d.keys()) == 6: 
            #         print(d)
            #         breakable = True
            #         break
            # if breakable: 
            #     break
                
                combos = []
                for p in d[6]: 
                    state = None
                    if len(d[6]) > 1 and p == -1: 
                        continue
                    if p != -1: 
                        state = clone.loc[str(p), 'state_info']
                    for q in d[9]: 
                        if len(d[9]) > 1 and q == -1: 
                            continue
                        if q != -1: 
                            state = clone.loc[str(q), 'state_info']
                        for r in d[12]: 
                            if len(d[12]) > 1 and r == -1: 
                                continue
                            if r != -1: 
                                state = clone.loc[str(r), 'state_info']
                            for s in d[15]: 
                                barcode13 = None
                                if len(d[15]) > 1 and s == -1: 
                                    continue
                                if s != -1: 
                                    barcode13 = clone.loc[str(s), 'barcode_day13']
                                    state = clone.loc[str(s), 'state_info']
                                    # break
                                    # if barcode13 == 'NA': 
                                    #     continue
                                for t in d[21]: 
                                    if len(d[21]) > 1 and t == -1: 
                                        continue
                                    if t != -1 and barcode13 and clone.loc[str(t), 'barcode_day13'] != barcode13: 
                                        continue
                                    if t != -1: 
                                        barcode13 = clone.loc[str(t), 'barcode_day13']
                                        state = clone.loc[str(t), 'state_info']
                                        # if barcode13 == 'NA': 
                                        #     continue
                                    for u in d[28]: 
                                        if len(d[28]) > 1 and u == -1: 
                                            continue
                                        if p + q + r + s + t + u == -6 or neg([p, q, r, s, t, u]) >= 2: #or or s + t + u == -3 or p + q + r==-3
                                            continue
                                        if u != -1 and barcode13 and clone.loc[str(u), 'barcode_day13'] != barcode13: 
                                            continue
                                        if u != -1: 
                                            barcode13 = clone.loc[str(u), 'barcode_day13']
                                            state = clone.loc[str(u), 'state_info']
                                            # if barcode13 == 'NA': 
                                            #     continue
                                        combos.append([p, q, r, s, t, u, state])
                total_combos.extend(combos)
            
            combo_df = pd.DataFrame(total_combos, columns=['Day6', 'Day9', 'Day12', 'Day15', 'Day21', 'Day28', 'state_info'])
            print(combo_df['state_info'].value_counts())
            combo_df.to_csv(folder+'Reprogramming_Encoded_Time_Series_Indices.csv')
            print(combo_df.tail(5))
            
            all_6_days = []
            others_5_days = []
            reprogrammed_5_days = []
            failed_5_days = []
            for i, row in combo_df.iterrows(): 
                if row['Day6'] == -1 or row['Day9'] == -1 or row['Day12'] == -1 or row['Day15'] == -1 or row['Day21'] == -1 or row['Day28'] == -1: 
                    if row['state_info'] == 'Others': 
                        others_5_days.append(row.tolist())
                    elif row['state_info'] == 'Reprogrammed': 
                        reprogrammed_5_days.append(row.tolist())
                    elif row['state_info'] == 'Failed': 
                        failed_5_days.append(row.tolist())
                    continue
                else: 
                    all_6_days.append(row.tolist())
            
            
            all_6_days_df = pd.DataFrame(all_6_days, columns=['Day6', 'Day9', 'Day12', 'Day15', 'Day21', 'Day28', 'state_info'])
            all_6_days_df['state_info'].value_counts()
            all_6_days_df.to_csv(folder+'Reprogramming_Encoded_Time_Series_Indices_All_6_Days.csv')
            all_6_days_df.head(5)
            
            others_5_days_df = pd.DataFrame(others_5_days, columns=['Day6', 'Day9', 'Day12', 'Day15', 'Day21', 'Day28', 'state_info'])
            others_5_days_df['state_info'].value_counts()
            others_5_days_df.to_csv(folder+'Reprogramming_Encoded_Time_Series_Indices_Others_5_Days.csv')
            others_5_days_df.head(5)
            
            reprogrammed_5_days_df = pd.DataFrame(reprogrammed_5_days, columns=['Day6', 'Day9', 'Day12', 'Day15', 'Day21', 'Day28', 'state_info'])
            reprogrammed_5_days_df['state_info'].value_counts()
            reprogrammed_5_days_df.to_csv(folder+'Reprogramming_Encoded_Time_Series_Indices_Reprogrammed_5_Days.csv')
            reprogrammed_5_days_df.head(5)
            
            failed_5_days_df = pd.DataFrame(failed_5_days, columns=['Day6', 'Day9', 'Day12', 'Day15', 'Day21', 'Day28', 'state_info'])
            failed_5_days_df['state_info'].value_counts()
            failed_5_days_df.to_csv(folder+'Reprogramming_Encoded_Time_Series_Indices_Failed_5_Days.csv')
            failed_5_days_df.head(5)
            
            print(all_6_days_df['state_info'].value_counts())
            print(others_5_days_df['state_info'].value_counts())
            print(reprogrammed_5_days_df['state_info'].value_counts())
            print(failed_5_days_df['state_info'].value_counts())
            
            others_5days_total = 30000 - all_6_days_df.shape[0]
            print(others_5days_total)
            
            rng = np.random.default_rng(seed=17179)
            
            rand_index_6_days = np.arange(all_6_days_df.shape[0])
            np.random.shuffle(rand_index_6_days)
            rand_index_others = rng.choice(others_5_days_df.shape[0], size=others_5days_total, replace=False, shuffle=True)
            rand_index_reprogrammed = rng.choice(reprogrammed_5_days_df.shape[0], size=30000, replace=False, shuffle=True)
            rand_index_failed = rng.choice(failed_5_days_df.shape[0], size=30000, replace=False, shuffle=True)
            
            print(rand_index_others)
            print(rand_index_reprogrammed)
            print(rand_index_failed)
            
            rand_index_others_list = rand_index_others.tolist()
            rand_index_reprogrammed_list = rand_index_reprogrammed.tolist()
            rand_index_failed_list = rand_index_failed.tolist()
            rand_index_6_days_list = rand_index_6_days.tolist()
            
            train_list_reprogrammed = rand_index_reprogrammed_list[0:10000]
            val_list_reprogrammed = rand_index_reprogrammed_list[10000:20000]
            test_list_reprogrammed = rand_index_reprogrammed_list[20000:]
            
            train_list_failed = rand_index_failed_list[0:10000]
            val_list_failed = rand_index_failed_list[10000:20000]
            test_list_failed = rand_index_failed_list[20000:]
            
            train_list_others = rand_index_others_list[0:9900]
            val_list_others = rand_index_others_list[9900:19800]
            test_list_others = rand_index_others_list[19800:]
            
            train_list_6_days = rand_index_6_days_list[0:100]
            val_list_6_days = rand_index_6_days_list[100:200]
            test_list_6_days = rand_index_6_days_list[200:]
            
            train_others_df = others_5_days_df.iloc[train_list_others]
            train_reprogrammed_df = reprogrammed_5_days_df.iloc[train_list_reprogrammed]
            train_failed_df = failed_5_days_df.iloc[train_list_failed]
            
            val_others_df = others_5_days_df.iloc[val_list_others]
            val_reprogrammed_df = reprogrammed_5_days_df.iloc[val_list_reprogrammed]
            val_failed_df = failed_5_days_df.iloc[val_list_failed]
            
            test_others_df = others_5_days_df.iloc[test_list_others]
            test_reprogrammed_df = reprogrammed_5_days_df.iloc[test_list_reprogrammed]
            test_failed_df = failed_5_days_df.iloc[test_list_failed]
            
            train_all_6days_df = all_6_days_df.iloc[train_list_6_days]
            val_all_6days_df = all_6_days_df.iloc[val_list_6_days]
            test_all_6days_df = all_6_days_df.iloc[test_list_6_days]
            
            train_inds_df = pd.concat([train_others_df, train_reprogrammed_df, train_failed_df, train_all_6days_df], ignore_index=True)
            
            val_inds_df = pd.concat([val_others_df, val_reprogrammed_df, val_failed_df, val_all_6days_df], ignore_index=True)
            
            test_inds_df = pd.concat([test_others_df, test_reprogrammed_df, test_failed_df, test_all_6days_df], ignore_index=True)
            
            print(train_inds_df['state_info'].value_counts())
            print(val_inds_df['state_info'].value_counts())
            print(test_inds_df['state_info'].value_counts())
            
            train_inds_df.to_csv(folder+'Reprogramming_Train_Inds.csv')
            val_inds_df.to_csv(folder+'Reprogramming_Val_Inds.csv')
            test_inds_df.to_csv(folder+'Reprogramming_Test_Inds.csv')

