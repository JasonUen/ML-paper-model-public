#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 22:59:14 2023

@author: tuen2
"""

import numpy as np
import pandas as pd

# import geopandas as gpd
from dbfread import DBF
import os
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import matplotlib.lines as mlines
import scipy.stats as stats
import pylab as py
from sklearn.preprocessing import PowerTransformer

# from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# from sklearn.preprocessing import RobustScaler
# from sklearn.preprocessing import QuantileTransformer
from functools import reduce

# set data path
project_dir = r"/Users/tinn-shuanuen/Library/CloudStorage/OneDrive-Personal/UIUC/Research/MacPro/ML"
path_base_data = r"/Users/tinn-shuanuen/Library/CloudStorage/OneDrive-Personal/UIUC/Research/Data"  # '/path/to/your/data'
# epa_path = os.path.join(path_base_data, 'EPA/ExcessFoodPublic_USTer_2018_R9/Output_geo')
epa_path = os.path.join(path_base_data, "EPA/ExcessFoodPublic_USTer_2018_R9/Output_dig")

if "geo" in epa_path:
    outpath = os.path.join(project_dir, "ML_PreAnalysisResults/geo/")
    # r'/Users/tuen2/OneDrive - University of Illinois - Urbana/MacPro/ML/ML_PreAnalysisResults/geo/'
elif "dig" in epa_path:
    outpath = os.path.join(project_dir, "ML_PreAnalysisResults/dig/")
    # utpath = r'/Users/tuen2/OneDrive - University of Illinois - Urbana/MacPro/ML/ML_PreAnalysisResults/dig/'

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

## remove nan rows
keyData.dropna(inplace=True)
keyData = keyData.reset_index(drop=True)

# set a string type column for geoid to keep leading zeros
keyData["GEOID"] = keyData["GEOID"].astype(
    "string"
)  # Be careful about str, 'string', and 'category'
rawData["GEOID"] = rawData["GEOID"].astype(
    "string"
)  # Be careful about str, 'string', and 'category'

# %% transformation
# Box-Cox
# Boxcox_keyData = pd.DataFrame(columns = keyData.columns[1:])
# check if data are all POSITIVE required for Box-Cox
if all(keyData.iloc[:, 1:].min() > 0):
    print("Data is good for BC transformation")
else:
    bad_cols = keyData.columns[1:][~(keyData.iloc[:, 1:].min() > 0)]
    print(bad_cols)
    print("Drop rows with non-positive data...")
    keyData = keyData[(keyData[bad_cols] > 0).all(axis=1)]

keyData.reset_index(drop=True, inplace=True)
BC_cols = keyData.columns[1:]  # exclude geoid column
print(keyData[BC_cols].min())
Ld_df = pd.DataFrame(index=BC_cols, columns=["Lambda"])
for c in BC_cols:
    results = stats.boxcox(keyData[c])
    Ld_df.loc[c, "Lambda"] = results[1]
    # Boxcox_col = pd.DataFrame(results[0], columns=[c])
    keyData[c] = results[0]
    # normality(Boxcox_keyData,c)
    # plt.savefig('NormalityTest/Box_Cox/%s.png'%c)
Ld_df["Lambda"] = Ld_df["Lambda"].astype(float)

# %%
## Normalization
scaler = MinMaxScaler()
scaler.fit(keyData[BC_cols])
data_m = pd.DataFrame(scaler.transform(keyData[BC_cols]), columns=BC_cols)

#%%
## correlation aanlysis
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
plt.show()

# # export data
# os.chdir(outpath)
# keyData.to_csv('FW_density/keyData_density.csv')
# Boxcox_keyData.to_csv('TransformedData/Boxcox_keyData_density.csv')
# Ld_df.to_csv('TransformedData/BoxCox_Lambda_density.csv')
