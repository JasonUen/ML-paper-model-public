# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 23:47:19 2022

@author: Jason
"""
import numpy as np
import pandas as pd
#import geopandas as gpd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import matplotlib.lines as mlines
import scipy.stats as stats
import pylab as py
from sklearn.preprocessing import PowerTransformer
#from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
#from sklearn.preprocessing import RobustScaler
#from sklearn.preprocessing import QuantileTransformer
from functools import reduce
from scipy.stats import shapiro, normaltest
from pathlib import Path

# Define project root dir
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# set data path
""" 
path_base_data is the path to the base data directory, users should change this to their own data path.
Detailed paths are specified in data_configs.py.
"""
path_base_data = r"/Users/tinn-shuanuen/Library/CloudStorage/OneDrive-Personal/UIUC/Research/Data"

# create output directory for data analysis
OUTPUT_DIR = PROJECT_ROOT / "data_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %% read data
from data_configs import data_configs

def load_csv(filepath, columns=None, rename=None, drop_index=None, **kwargs):
    df = pd.read_csv(filepath, index_col=0, dtype=str, **kwargs)
    if drop_index is not None:
        df = df.drop(index=drop_index)
    if columns:
        df = df[columns].copy()
    if rename:
        df = df.rename(columns=rename)
    return df
print("Loading dataset...")
dataframes = {}
for cfg in data_configs:
    filepath = os.path.join(path_base_data, cfg["relpath"])
    dataframes[cfg["name"]] = load_csv(
        filepath,
        columns=cfg.get("columns"),
        rename=cfg.get("rename"),
        drop_index=cfg.get("drop_index"),
    )
# %%
## combine all data into one dataframe
data = None
df_all = [df for df in dataframes.values()]
data = reduce(
    lambda left, right: pd.merge(left, right, on="GEOID", how="outer"), df_all
)  # need to make sure GEOID exists in all dataframes

# %% process nan and convert to numeric
geoid = data["GEOID"]
data["Population"] = data["Population"].str.replace(",", "")
data[data.columns.drop("GEOID")] = data[data.columns.drop("GEOID")].astype(float)
trs_cols = data.iloc[:, 6:11].columns
data["TRS"] = data[trs_cols].sum(axis=1)  # combine all data related to TRS
data["EXP_FR"] = data["PC_FFRSALES"] + data["PC_FSRSALES"]
data["SWS"] = data["SNAPS"] + data["WICS"]  # SNAP and WIC authourized stores
data["REDM"] = (
    data["REDEMP_SNAPS"] + data["REDEMP_WICS"]
)  # combine all data related to REDM

# rename columns
data = data.rename(
    columns={
        "MeanEF": "FW",
        "AmlProduct": "AS",
        "Crop": "CS",
        "Population": "POP",
        "IncPerCap": "InPC",
        "TFR_Exp": "EXP_FR",
        "SNAP_WIC": "SWS",
        "REDEMP_TTL": "REDM",
        "LACCESS_POP": "LACS_POP",
        "LACCESS_LOWI": "LACS_LOWI",
        "LACCESS_HHNV": "LACS_HHNV",
        "LACCESS_SNAP": "LACS_SNAP",
    }
)  # EF = excess food

## calculate FW density
data.insert(1, "FW_density", data["FW"] / data["POP"])

## extract target columns
cols = [
    "GEOID",
    "FW", #  FW_density was used for testing
    "AS",
    "CS",
    "TRS",
    "POP",
    "InPC",
    "MEDHHINC",
    "EXP_FR",
    "SWS",
    "REDM",
    "LACS_POP",
    "LACS_LOWI",
    "LACS_HHNV",
    "LACS_SNAP",
]

keyData = data[cols].copy()
rawData = data[cols].copy()

#%% grab raw data dim
nd = rawData.shape[1]-1
data_dim = np.ones(nd) * rawData.shape[0]
nan_data = np.zeros(nd)
for i in range(nd):
    c = rawData.columns[i+1]
    nan_data[i] = rawData[c].isna().sum()
E_data_dim = data_dim - nan_data
#------------------------------------------------------------------------------------------------
#%% plot data removed
features = rawData.columns[1:]
bw=0.7
matplotlib.rcParams.update({'font.size': 16})
fig, ax = plt.subplots(figsize=(17, 9))
ax.bar(features, E_data_dim, width=bw, label='Effective data')
ax.bar(features, nan_data, bottom = E_data_dim, color='red', width=bw, label='NaN data')
# add data labels
rects = ax.patches
for rect, label in zip(rects, nan_data):
    height = np.max(E_data_dim)
    ax.text(
        rect.get_x() + rect.get_width() / 2, height + 5, int(label), ha="center", va="bottom"
    )
ax.set_xlabel("Features", fontweight='bold', fontsize=16)
ax.set_ylabel("Data count", fontweight='bold', fontsize=16)
ax.set_xticklabels(features, rotation=45, ha='right')
# make thousand comma
ax.set_yticklabels(np.arange(0,3500,500))
ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))
ax.legend(loc='center left', fontsize=16)
plt.subplots_adjust(left=0.02, right=0.98)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'data_cleaning.png')
#plt.show()
#%% remove nan rows
keyData.dropna(inplace=True)
keyData = keyData.reset_index(drop = True)

# set a string type column for geoid to keep leading zeros
keyData['GEOID'] = keyData['GEOID'].astype('string') 
rawData['GEOID'] = rawData['GEOID'].astype('string')

# Basic statistics
stat = keyData.iloc[:,1:].describe(include='all').T
stat =stat.round(2)
stat['count'] = stat['count'].astype(int)

stat.to_csv(OUTPUT_DIR / 'statistics.csv')
keyData.to_csv(OUTPUT_DIR / 'keyData.csv')
#%%
# validate data
def validate_data(df, feature_columns):
    missing_cols = [col for col in feature_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns in the dataset: {missing_cols}")

    non_numeric = [col for col in feature_columns if not pd.api.types.is_numeric_dtype(df[col])]
    if non_numeric:
        raise TypeError(f"Non-numeric columns: {non_numeric}")

    print("Data validation passed.")

# statistical tests
results = []
feature_cols = keyData.columns[1:]
validate_data(keyData, feature_cols)

for f in feature_cols:
    stat, p = shapiro(keyData[f].dropna())
    results.append({'feature': f, 'Shapiro_p': p, 'normal': True if p > 0.05 else False})
res_df = pd.DataFrame(results)

# check = []
# for f in feature_cols:
#     stat, p = shapiro(keyData[f].dropna())
#     #print(f"{f}: Shapiro-Wilk p-value = {p:.4f}")
#     if p > 0.05:
#         check.append(True)
#     else:
#         check.append(False)

print(f"Number of normal features: {res_df.normal.sum()}")
print(f"Number of non-normal features: {len(res_df) - res_df.normal.sum()}")

#%% Transformation
## Box-Cox
# check if data are all POSITIVE required for Box-Cox
Boxcox_keyData = pd.DataFrame()
if all(keyData.iloc[:, 1:].min() > 0):
    print("Data is good for BC transformation")
else:
    bad_cols = feature_cols[~(keyData.loc[:, feature_cols].min() > 0)]
    print(bad_cols)
    print("Drop rows with non-positive data...")
    keyData = keyData[(keyData[bad_cols] > 0).all(axis=1)]

keyData.reset_index(drop=True, inplace=True)
BC_cols = feature_cols # exclude geoid column
#print(keyData[BC_cols].min()) # double-check
Ld_df = pd.DataFrame(index=BC_cols, columns=["Lambda"])

# function to check visualization of normality for features
def normality_plot(data,feature):
    plt.figure(figsize=(17,7))

    plt.subplot(1,2,1)
    sns.kdeplot(data[feature])

    plt.subplot(1,2,2)
    stats.probplot(data[feature],plot=py)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f'normality_{c}.png')
    #plt.show()

for c in BC_cols:
    print(f"Applying Box-Cox transformation to {c}...")
    results = stats.boxcox(keyData[c])
    Ld_df.loc[c,'Lambda'] = results[1]
    Boxcox_keyData[c] = results[0]
    normality_plot(Boxcox_keyData,c)

print("check normality after BC transformation")
results_bc = []
for f in Boxcox_keyData.columns:
    stat, p = shapiro(Boxcox_keyData[f].dropna())
    results_bc.append({'feature': f, 'Shapiro_p': p, 'normal': True if p > 0.05 else False})
res_df_bc = pd.DataFrame(results_bc)
print(f"Number of normal features: {res_df_bc.normal.sum()}")
print(f"Number of non-normal features: {len(res_df_bc) - res_df_bc.normal.sum()}")

# export results    
Boxcox_keyData.to_csv(OUTPUT_DIR / 'Boxcox_keyData.csv')
Ld_df['Lambda'] = Ld_df['Lambda'].astype(float)
Ld_df.to_csv(OUTPUT_DIR / 'BoxCox_Lambda.csv')

#%%
## Normalization
scaler=MinMaxScaler()
scaler.fit(Boxcox_keyData)
data_m = pd.DataFrame(scaler.transform(Boxcox_keyData), columns = Boxcox_keyData.columns)

#%%
## correlation aanlysis for data BEFORE processing
labels = np.arange(0, 1.3, 0.2).tolist()
correlation = data_m.corr(method="pearson")
columns = correlation.nlargest(
    50, "FW" # FW_density was used for testing
).index  # select indices from 25 largest data
correlation_map = np.corrcoef(data_m[columns].values.T)
plt.figure(figsize=(11, 11))
sns.set(rc={"figure.figsize": (16, 12), "font.size": 14})
heatmap = sns.heatmap(
    correlation_map,
    cbar=True,
    annot=True,
    square=True,
    fmt=".2f",
    yticklabels=columns.values,
    xticklabels=columns.values,
    cmap="Blues",
    vmin=-0.2,
    vmax=1,
    cbar_kws={'shrink': 0.72}
)
heatmap.set_xticklabels(columns, fontsize=16)  # heatmap.get_xmajorticklabels(),
heatmap.set_yticklabels(columns, fontsize=16)
plt.subplots_adjust(left=0.2, top=1, right=1)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'corr_heatmap.png')
#plt.show()

#%%
## Histogram visualization
# Unit conversion
unit_dict = {
    'FW': 1e3,   'EXP_FR': 1e3, 'SWS': 1e3,  'TRS': 1e3,
    'InPC': 1e3, 'MEDHHINC': 1e3,'LACS_POP':1e3,'LACS_LOWI':1e3,
    'LACS_HHNV':1e3,'LACS_SNAP':1e3,'POP':1e6,'REDM':1e6,
    'AS':1e9, 'CS':1e9
}
unit_suffix = {
    **{i:'k' for i in unit_dict if unit_dict[i]==1e3},
    **{i:'M' for i in unit_dict if unit_dict[i]==1e6},
    **{i:'G' for i in unit_dict if unit_dict[i]==1e9},
}
def apply_units(df, UNIT_DICT):
    df = df.copy()
    for col, factor in UNIT_DICT.items():
        if col in df:
            df[col] = df[col] / factor
    return df
data_vis = apply_units(keyData, unit_dict)
cols = columns.tolist() 

plotNames = [
    'FW (kt)', 'TRS (k)', 'POP (M)',  'SWS (k)', 'LACS_POP (k)', 'LACS_HHNV (k)','LACS_LOWI (k)', 'LACS_SNAP (k)', 
    'REDM (MUSD)', 'MEDHHINC (kUSD)','InPC (kUSD)', 'CS (GUSD)', 'EXP_FR (kUSD)', 'AS (GUSD)'
    ]
#%%
## plot effective data 
props = dict(boxstyle='round', facecolor='wheat', alpha=0.75)
num_zoom_data = 1500
zoom_cols = ['FW', 'TRS', 'POP',  'SWS', 'LACS_POP',
             'LACS_HHNV','LACS_LOWI', 'LACS_SNAP', 'CS', 'AS']
matplotlib.rcParams.update({'font.size': 20})
fig, ax = plt.subplots(3,5, figsize=(16,9)) # 
fig.supylabel('Count', fontweight='bold') #fontsize=14 
#fig.supxlabel('Value', fontweight='bold')
# create legend
fig.tight_layout(rect=[0. ,0.01, 0.97, 1.0], h_pad=2.5)
ind = 0
tc = data_vis.shape[0]
for i in range(3):
    for j in range(5):
        if ind < 14:
            c = columns[ind]
            ax[i][j].hist(data_vis[c],bins = 50, edgecolor='None')
            ax[i][j].grid(True, alpha=0.5)
            ax[i][j].set_xlabel(plotNames[ind], fontsize=16) # , fontweight='bold'
            ax[i][j].get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ','))) # add thousand separator
            # set color for zoom-in data in the histogram
            if c in zoom_cols:
                v = data_vis[c].sort_values(ignore_index=True)[num_zoom_data] # value of 1500th data
                for bar in ax[i][j].containers[0]:
                    b = bar.get_x() #+ 0.5 * bar.get_width()
                    if b < v:
                        bar.set_color('orange')
        if i == 2 and j == 4:
            ax[i][j].set_xticks([])
            ax[i][j].set_yticks([])
            ax[i][j].set_facecolor("white")
            ax[i][j].text(0., 0.2, 'Total counts: %d'%tc, transform = ax[i][j].transAxes,
                          verticalalignment='top', bbox=props, fontweight='bold') #fontsize=14,
        ind+=1
        ax[i][j].tick_params(labelsize=16)  
#%%
# plot rawData
data_plot = rawData[columns]
fig, ax = plt.subplots(3,5, figsize=(16,9))
fig.supylabel('Count', fontweight='bold')
fig.tight_layout(rect=[0. ,0.01, 0.97, 1.0], h_pad=2.5)
ind = 0
for i in range(3):
    for j in range(5):
        if ind < 14:
            tc = "{:,}".format(int(E_data_dim[ind]))
            mode = data_plot.iloc[:,ind].mode().iloc[0]
            ax[i][j].hist(data_plot.iloc[:,ind],bins = 50, edgecolor='None')
            ax[i][j].grid(True, alpha=0.5)
            ax[i][j].set_xlabel(plotNames[ind],fontsize=16) # ,fontsize=15, fontweight='bold'
            ind+=1
            ax[i][j].text(0.425, 0.55, '%s'%tc, transform = ax[i][j].transAxes,verticalalignment='top', bbox=props) # plot mode line
            ax[i][j].get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ','))) # add thousand separator
        if i == 2 and j == 4:
            ax[i][j].set_xticks([])
            ax[i][j].set_yticks([])
            ax[i][j].set_facecolor("white")
            ax[i][j].text(0., 0.2, 'Total counts', transform = ax[i][j].transAxes,
                          verticalalignment='top', bbox=props, fontweight='bold') #fontsize=14,
        ax[i][j].tick_params(labelsize=12)
#%%
# plot zoom-in histograms
zoom_plot_names =  ['FW (kt)', 'TRS (k)', 'POP (M)',  'SWS (k)', 'LACS_POP (k)',
                    'LACS_HHNV (k)','LACS_LOWI (k)', 'LACS_SNAP (k)', 'CS (GUSD)', 'AS (GUSD)']
#x_max_list=[25, 2, 1.0, 0.5, 150, 3, 30, 4, 0.2, 0.5] # max of x axis to show data
data_plot = data_vis[zoom_cols]
fig, ax = plt.subplots(2,5, figsize=(16,7)) # 
fig.supylabel('Count', fontweight='bold') #fontsize=14 
#fig.supxlabel('Value', fontweight='bold')
fig.tight_layout(rect=[0. ,0.01, 0.97, 1.0], h_pad=2.5)
ind = 0
tc=data_plot.shape[0]
for i in range(2):
    for j in range(5):
        if ind < 10:
            c = zoom_cols[ind]
            median = data_plot[c].median()
            data_zoom = data_plot[c].sort_values().head(num_zoom_data)
            zoom_tc = "{:,}".format(data_zoom.shape[0])
            ax[i][j].hist(data_zoom, bins = 50, edgecolor='None', color='orange')
            ax[i][j].grid(True, alpha=0.5)
            ax[i][j].set_xlabel(zoom_plot_names[ind],fontsize=20) # , fontweight='bold'
            ind+=1
            ax[i][j].get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ','))) # add thousand separator

#%%
## correlation aanlysis for data AFTER processing
labels = np.arange(0, 1.3, 0.2).tolist()
correlation = data_m.corr(method="pearson")
columns = correlation.nlargest(
    50, "FW" # FW_density was used for testing
).index  # select indices from 25 largest data
correlation_map = np.corrcoef(data_m[columns].values.T)
plt.figure(figsize=(11, 11))
sns.set(rc={"figure.figsize": (16, 12), "font.size": 14})
heatmap = sns.heatmap(
    correlation_map,
    cbar=True,
    annot=True,
    square=True,
    fmt=".2f",
    yticklabels=columns.values,
    xticklabels=columns.values,
    cmap="Blues",
    vmin=-0.2,
    vmax=1,
    cbar_kws={'shrink': 0.72}
)
heatmap.set_xticklabels(columns, fontsize=16)  # heatmap.get_xmajorticklabels(),
heatmap.set_yticklabels(columns, fontsize=16)
plt.subplots_adjust(left=0.2, top=1, right=1)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'corr_heatmap.png')
#plt.show()