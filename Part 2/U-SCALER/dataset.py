
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import rasterio
import sys
from tqdm import tqdm
import pandas as pd
from pathlib import Path
import tifffile as tiff

import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from util import *

import gc


class LandsatMODISDataset(Dataset):
    def __init__(self, datapath, transform=None, start_date=None, end_date=None):
        
        stats_df  = pd.read_csv(datapath+'Part 2/Input/channel_stats.csv')
        
        files_df = pd.read_csv(datapath+'Part 2/example_pairs.csv')
        files_df = files_df.reset_index(drop=True)
        
        if start_date is not None:
            start_dt = pd.to_datetime(start_date, format="%Y%m%d")
            files_df = files_df[
                pd.to_datetime(files_df["date_t2"]) >= start_dt
            ].reset_index(drop=True)
        
        if end_date is not None:
            end_dt = pd.to_datetime(end_date, format="%Y%m%d")
            files_df = files_df[
                pd.to_datetime(files_df["date_t2"]) <= end_dt
            ].reset_index(drop=True)
        
        self.files = files_df
        self.transform = transform
        self.datapath = datapath
        self.stats_df = stats_df
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        
        sub_folder = ''
        tif_path_m = self.files['tif_path'].iloc[idx]
        tif_path_l = tif_path_m.replace("GPP_M/", "GPP_L/")
        
        tif_path_m1 = self.files['tif_path_t1'].iloc[idx]
        tif_path_l1 = tif_path_m1.replace("GPP_M/", "GPP_L/")
        
        with rasterio.open(self.datapath+'Part 2/Input/'+tif_path_l1) as src_l:
            gpp_l1 = src_l.read(1).astype("float32")
            gpp_l1[gpp_l1 < 0] = 0
        
        tif_path_m3 = self.files['tif_path_t3'].iloc[idx]
        tif_path_l3 = tif_path_m3.replace("GPP_M/", "GPP_L/")
        
        with rasterio.open(self.datapath+'Part 2/Input/'+tif_path_l3) as src_l:
            gpp_l3 = src_l.read(1).astype("float32")
            gpp_l3[gpp_l3 < 0] = 0
        
        
        if Path(self.datapath+'Part 2/Input/'+tif_path_l).exists():
            with rasterio.open(self.datapath+'Part 2/Input/'+tif_path_l) as src_l:
                gpp_l = src_l.read(1).astype("float32")
                zero_mask = np.isnan(gpp_l)
                gpp_l[gpp_l < 0] = 0
        else:
            gpp_l = np.zeros((256, 256), dtype='float32')
            gpp_l[gpp_l == 0] = np.nan
            zero_mask = np.ones((1, 256, 256), dtype='float32')
        
        
        with rasterio.open(self.datapath+'Part 2/Input/'+tif_path_m) as src_m:
            gpp_m = src_m.read(1).astype("float32")
        
        with rasterio.open(self.datapath+'Part 2/Input/'+tif_path_m1) as src_m:
            gpp_m1 = src_m.read(1).astype("float32")
        
        with rasterio.open(self.datapath+'Part 2/Input/'+tif_path_m3) as src_m:
            gpp_m3 = src_m.read(1).astype("float32")
        
        
        vars_path = self.files['vars_path'].iloc[idx]
        vars_all = tiff.imread(self.datapath+'Part 2/Input/'+vars_path).astype("float32")
        
        aoi = vars_all[0, :, :]
        vars_all[2:,:,:] = vars_all[2:,:,:]/10000
        
        water_mask = vars_all[1,:,:] # from LC
        
        band_name = ["GPP_L", "GPP_M", "GPP_L_T1", "GPP_M_T1", "GPP_L_T3", "GPP_M_T3", "NDVI_P30", "NDVI_P50", "NDVI_P70"]
        stack = np.stack([gpp_l, gpp_m, gpp_l1, gpp_m1, gpp_l3, gpp_m3] + list(vars_all[2:, :, :]), axis=0)
        
        nan_ind = stack < 0
        
        for i in [0, 2, 4]:
            stack[i] = log_normalize_to_minus1_1_np(stack[i], max_value=33)
        
        for i in [1, 3, 5]:
            stack[i] = log_normalize_to_minus1_1_np(stack[i], max_value=33)
        
        for i in [6, 7, 8]:
            stack[i][stack[i] <= 0] = 0
            stack[i] = (stack[i]) / (self.stats_df[self.stats_df['channel'] == band_name[i]]['max'].values[0])
            stack[i] = (stack[i] - 0.5) * 2
        
        aoi_mask = aoi==0
        water_mask = water_mask==1
        
        gpp_l = stack[0]
        gpp_l[water_mask==1] = -1
        gpp_l[aoi_mask] = -1
        
        stack[nan_ind] = 0
        stack[:, :, :][:, water_mask] = -1
        stack[:,aoi_mask] = -1
        
        targets = stack[0:1]
        inputs = stack[1:]
        
        doys = torch.tensor([self.files['doy_t1'].iloc[idx], self.files['doy_t2'].iloc[idx], self.files['doy_t3'].iloc[idx]])
        dates = [self.files['date_t1'].iloc[idx], self.files['date_t2'].iloc[idx], self.files['date_t3'].iloc[idx]]
        
        targets = np.nan_to_num(targets)
        inputs = np.nan_to_num(inputs)
        
        if self.transform:
            targets = self.transform(targets)
            inputs = self.transform(inputs)
            zero_mask = self.transform(zero_mask)
        
        path5 = self.files['tif_path'].iloc[idx]
        save_file = path5.split("/")[-2] + "_" + path5.split("/")[-1][:-4]
        save_file_path = save_file
        
        gc.collect()
        
        return targets, inputs, doys, save_file_path, dates, zero_mask, aoi_mask, water_mask

