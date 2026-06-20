
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import rasterio
import sys
from tqdm import tqdm
import pandas as pd
from pathlib import Path
import tifffile as tiff

import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime


import random

root_dir = 'UNIQUE/'
sys.path.append(root_dir+'Part 2/U-SCALER/')

from model import CARE_Net, flow_unet
from dataset import LandsatMODISDataset
from diffusion import *
from util import *

#########################################################
# ----------------------------------------------------- #
#########################################################
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

random_seed = 2024

random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ----------------------------------------------------- #

NGPU = torch.cuda.device_count()

# GPU; CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)
print('Count of using GPUs:', torch.cuda.device_count())


#%%
# Model setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

netG = CARE_Net(dim=64, c_in=8+2, t_option=1)
flow = flow_unet(dim=64, c_in=8)

netG.to(device)
flow.to(device)


#%%
ckpt_dir = root_dir+'Part 2/pretrained/checkpoint'


netG, st_epoch = load_one(ckpt_dir=ckpt_dir, netG=netG)
flow, st_epoch = load_one(ckpt_dir=ckpt_dir+'_flow', netG=flow)

netG.to(device)
flow.to(device)

diffuzz = Diffusion(model = netG, flow=flow, device=device, training_target='v', timesteps=1000)

#%%

dataset_test = LandsatMODISDataset(root_dir, transform=lambda x: torch.tensor(x, dtype=torch.float32))
dataloader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=8)

num_data_test = len(dataset_test)
num_batch_test = int(np.ceil(num_data_test / 1))

with torch.no_grad():
    netG.eval()
    flow.eval()
    for batch, images in tqdm(enumerate(dataloader_test, 1)): #tqdm(enumerate(dataloader_test, 1)):
        gt_image = images[0].to(device).float()
        structure_image = images[1].to(device).float()
        doyts = images[2].to(device).float()
        date_info = images[4]
        filenames = images[3]
        gt_image_mask = images[5].to(device).float() #구름인 부분이 1
        aoi_mask = images[6].to(device).float().squeeze()
        water_mask = images[7].to(device).float().squeeze()
        
        if gt_image_mask.shape[0]>1:
            gt_image_mask = gt_image_mask.unsqueeze(1) 
        
        with torch.no_grad():        
            sampled_images, warped = diffuzz.ddim_sample(
                background = structure_image,
                custom_timesteps = 100,
                x_init=gt_image,
                cloud_mask=gt_image_mask,
                clip_x_start=True,
            )
            
            output = ((sampled_images)).squeeze()
            
            output[aoi_mask==1] = -1
            output[water_mask==1] = -1
            
            output = output.cpu().numpy()
            warped = warped.float().squeeze().cpu().numpy()
            
            output = log_denormalize_from_minus1_1_np(output, max_value=33)
            warped = log_denormalize_from_minus1_1_np(warped, max_value=33)
            
            if output.ndim==2:
                imgs = {
                        'output_img': output,
                        'warped': warped,
                        'date_info' : date_info[1]
                    }
                savefile = filenames[0].replace("GPP_M_","")
                np.savez(os.path.join(root_dir, 'Part 2', 'Output', savefile+".npz"), **imgs)
            else:
                for j in range(output.shape[0]):
                    imgs = {
                            'output_img': output[j],
                            'warped': warped[j],
                            'date_info' : date_info[1][j]
                        }
                    savefile = filenames[j].replace("GPP_M_","")
                    np.savez(os.path.join(root_dir, 'Part 2', 'Output', savefile+".npz"), **imgs)
