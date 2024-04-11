import scanpy as sc
import cospar as cs
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict

# Load reprogramming data and gather barcode information
adata_orig = cs.datasets.reprogramming()
adata_orig.obs[adata_orig.obs["time_info"]=="Day9"]
Total=adata_orig.obs
l = pd.DataFrame(adata_orig.obs.barcode_all)
ll = pd.DataFrame(adata_orig.obs.state_info)
lll = pd.concat([l, ll], axis=1)

day0_barcodes = pd.unique(Total['barcode_day0']).to_numpy().tolist()
day3_barcodes = pd.unique(Total['barcode_day3']).to_numpy().tolist()
day13_barcodes = pd.unique(Total['barcode_day13']).to_numpy().tolist()

Total.fillna('-1', inplace=True)
df = adata_orig.to_df()
df = pd.concat([df, Total], axis=1)
columns = Total.columns.tolist()

def def_value():
    return [-1]

breakable = False
day0_and_day3_barcodes = df[['barcode_day0', 'barcode_day3']].value_counts(ascending=True).reset_index(name='count')

def neg(a): 
    count = 0
    for i in a: 
        if i < 0: 
            count += 1
    return count

# Generate time series based on barcode information
total_combos = []
for j, barcodes in day0_and_day3_barcodes.iterrows(): 
    barcode0 = barcodes['barcode_day0'] 
    barcode3 = barcodes['barcode_day3'] 
    if (barcode0 == 'NA') or (barcode3 == 'NA'): 
        continue
    clone = df.loc[(df['barcode_day0']==barcode0) & (df['barcode_day3']==barcode3)]
    d = defaultdict(def_value)
    for i, row in clone.iterrows(): 
        day = row['time_info']
        day = int(day[3:])
        d[day].append(int(i))
    
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
                    for t in d[21]: 
                        if len(d[21]) > 1 and t == -1: 
                            continue
                        if t != -1 and barcode13 and clone.loc[str(t), 'barcode_day13'] != barcode13: 
                            continue
                        if t != -1: 
                            barcode13 = clone.loc[str(t), 'barcode_day13']
                            state = clone.loc[str(t), 'state_info']
                        for u in d[28]: 
                            if len(d[28]) > 1 and u == -1: 
                                continue
                            if p + q + r + s + t + u == -6 or neg([p, q, r, s, t, u]) >= 2:
                                continue
                            if u != -1 and barcode13 and clone.loc[str(u), 'barcode_day13'] != barcode13: 
                                continue
                            if u != -1: 
                                barcode13 = clone.loc[str(u), 'barcode_day13']
                                state = clone.loc[str(u), 'state_info']
                            combos.append([p, q, r, s, t, u, state])
    total_combos.extend(combos)  

# Store all time series index
combo_df = pd.DataFrame(total_combos, columns=['Day6', 'Day9', 'Day12', 'Day15', 'Day21', 'Day28', 'state_info'])
combo_df['state_info'].value_counts()
combo_df.to_csv('Reprogramming_Data_Time_Series_Indices.csv')
combo_df.tail(5)

small_combo_df = pd.DataFrame(total_combos, columns=['Day6', 'Day9', 'Day12', 'Day15', 'Day21', 'Day28', 'state_info'])
small_combo_df['state_info'].value_counts()

combo_df = pd.read_csv('Reprogramming_Data_Time_Series_Indices.csv', index_col=0)

# Separate time series based on class and if all days had barcode information data
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

# Store the different time series types
all_6_days_df = pd.DataFrame(all_6_days, columns=['Day6', 'Day9', 'Day12', 'Day15', 'Day21', 'Day28', 'state_info'])
all_6_days_df['state_info'].value_counts()
all_6_days_df.to_csv('Reprogramming_Data_Time_Series_Indices_All_6_Days.csv')
all_6_days_df.head(5)

others_5_days_df = pd.DataFrame(others_5_days, columns=['Day6', 'Day9', 'Day12', 'Day15', 'Day21', 'Day28', 'state_info'])
others_5_days_df['state_info'].value_counts()
others_5_days_df.to_csv('Reprogramming_Data_Time_Series_Indices_Others_5_Days.csv')
others_5_days_df.head(5)

reprogrammed_5_days_df = pd.DataFrame(reprogrammed_5_days, columns=['Day6', 'Day9', 'Day12', 'Day15', 'Day21', 'Day28', 'state_info'])
reprogrammed_5_days_df['state_info'].value_counts()
reprogrammed_5_days_df.to_csv('Reprogramming_Data_Time_Series_Indices_Reprogrammed_5_Days.csv')
reprogrammed_5_days_df.head(5)

failed_5_days_df = pd.DataFrame(failed_5_days, columns=['Day6', 'Day9', 'Day12', 'Day15', 'Day21', 'Day28', 'state_info'])
failed_5_days_df['state_info'].value_counts()
failed_5_days_df.to_csv('Reprogramming_Data_Time_Series_Indices_Failed_5_Days.csv')
failed_5_days_df.head(5)

print(all_6_days_df['state_info'].value_counts())
print(others_5_days_df['state_info'].value_counts())
print(reprogrammed_5_days_df['state_info'].value_counts())
print(failed_5_days_df['state_info'].value_counts())

others_5days_total = 30000 - all_6_days_df.shape[0]
print(others_5days_total)

# Generate the training, validation and testing data
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

train_inds_df.to_csv('Reprogramming_Train_Inds.csv')
val_inds_df.to_csv('Reprogramming_Val_Inds.csv')
test_inds_df.to_csv('Reprogramming_Test_Inds.csv')


# Generate time series indices for only two classes
reprogrammed_5_days_df = pd.read_csv('Reprogramming_Encoded_Time_Series_Indices_Reprogrammed_5_Days.csv', index_col=0)
failed_5_days_df = pd.read_csv('Reprogramming_Encoded_Time_Series_Indices_Failed_5_Days.csv', index_col=0)

rng = np.random.default_rng(seed=17179)
rand_index_reprogrammed = rng.choice(reprogrammed_5_days_df.shape[0], size=45000, replace=False, shuffle=True)
rand_index_failed = rng.choice(failed_5_days_df.shape[0], size=45000, replace=False, shuffle=True)

print(rand_index_reprogrammed)
print(rand_index_failed)

rand_index_reprogrammed_list = rand_index_reprogrammed.tolist()
rand_index_failed_list = rand_index_failed.tolist()

train_list_reprogrammed = rand_index_reprogrammed_list[0:15000]
val_list_reprogrammed = rand_index_reprogrammed_list[15000:30000]
test_list_reprogrammed = rand_index_reprogrammed_list[30000:]

train_list_failed = rand_index_failed_list[0:15000]
val_list_failed = rand_index_failed_list[15000:30000]
test_list_failed = rand_index_failed_list[30000:]

train_reprogrammed_df = reprogrammed_5_days_df.iloc[train_list_reprogrammed]
train_failed_df = failed_5_days_df.iloc[train_list_failed]

val_reprogrammed_df = reprogrammed_5_days_df.iloc[val_list_reprogrammed]
val_failed_df = failed_5_days_df.iloc[val_list_failed]

test_reprogrammed_df = reprogrammed_5_days_df.iloc[test_list_reprogrammed]
test_failed_df = failed_5_days_df.iloc[test_list_failed]


train_inds_df = pd.concat([train_reprogrammed_df, train_failed_df], ignore_index=True)

val_inds_df = pd.concat([val_reprogrammed_df, val_failed_df], ignore_index=True)

test_inds_df = pd.concat([test_reprogrammed_df, test_failed_df], ignore_index=True)

print(train_inds_df['state_info'].value_counts())
print(val_inds_df['state_info'].value_counts())
print(test_inds_df['state_info'].value_counts())

train_inds_df.to_csv('Reprogramming_Train_Inds_Two_Classes.csv')
val_inds_df.to_csv('Reprogramming_Val_Inds_Two_Classes.csv')
test_inds_df.to_csv('Reprogramming_Test_Inds_Two_Classes.csv')
