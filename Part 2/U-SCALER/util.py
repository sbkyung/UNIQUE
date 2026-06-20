import os
import torch
import matplotlib.pyplot as plt
import numpy as np


def draw_gradient_map(ax, image, a, b, bbox, vmin, vmax, title):
    try:
        from mpl_toolkits.basemap import Basemap
    except ImportError as exc:
        raise ImportError(
            "draw_gradient_map requires basemap (conda install -c conda-forge basemap)"
        ) from exc

    m = Basemap(ax=ax, llcrnrlon=bbox[0], llcrnrlat=bbox[1], urcrnrlon=bbox[2], urcrnrlat=bbox[3], area_thresh=0.001, resolution='l')
    x, y = m(a, b)
    label_font_size = 10

    parallels = np.arange(int(round(bbox[1])), int(round(bbox[3])) + 0.1, 3.)
    m.drawparallels(parallels, labels=[1, 0, 0, 0], linewidth=0.01, color='0.5', fontsize=label_font_size)
    meridians = np.arange(int(round(bbox[0])), int(round(bbox[2])) + 0.1, 3.)
    m.drawmeridians(meridians, labels=[0, 0, 0, 1], linewidth=0.01, color='0.5', fontsize=label_font_size)

    m.fillcontinents(color='grey', lake_color='grey')
    m.drawmapboundary(fill_color='white')
    m.drawcoastlines(linewidth=0.5)
    m.drawcountries(linewidth=0.25)
    m.drawlsmask(lsmask=1, resolution='h', land_color='grey')
    ss = m.pcolormesh(x, y, image, vmin=vmin, vmax=vmax)
    cbar = m.colorbar(ss, extendfrac='auto', spacing='uniform', location='bottom', pad="10%")
    cbar.ax.tick_params(labelsize=10)
    
    ax.set_title(title, fontsize=12)


def save_one(ckpt_dir, netG, epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    
    torch.save({'netG': netG.state_dict()}, 
                 "%s/model_epoch%d.pth" % (ckpt_dir, epoch))

def load_one(ckpt_dir, netG):
    if not os.path.exists(ckpt_dir):
        epoch = 0
        return netG
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ckpt_lst = os.listdir(ckpt_dir)
    ckpt_lst = [f for f in ckpt_lst if f.endswith('pth')]
    ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    dict_model = torch.load('%s/%s' % (ckpt_dir, ckpt_lst[-1]), map_location=device)

    netG.load_state_dict(dict_model['netG'])
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])
    
    return netG, epoch

               
# Directory setup
def setup_directories(base_dir, sub_dirs):
    for sub_dir in sub_dirs:
        full_path = os.path.join(base_dir, sub_dir)
        if not os.path.exists(full_path):
            os.makedirs(full_path)
            
def denorm(x, min_val, max_val):
    return ((x + 1) / 2) * (max_val - min_val) + min_val

def log_normalize_to_minus1_1_np(x, max_value=20.0):
    """
    NumPy version: apply log1p and normalize to [-1, 1]
    
    Args:
        x : np.ndarray, input precipitation (non-negative)
        max_value : float, maximum precipitation value (e.g., 20)
    
    Returns:
        normalized array in [-1, 1]
    """
    x_log = np.log1p(x)  # log(x + 1)
    x_norm = x_log / np.log1p(max_value)  # scale to [0, 1]
    x_scaled = x_norm * 2 - 1             # scale to [-1, 1]
    return x_scaled


def log_denormalize_from_minus1_1_np(x_scaled, max_value=20.0):
    """
    Reverse of log_normalize_to_minus1_1_np.
    Args:
        x_scaled : np.ndarray, values in [-1, 1]
        max_value : float, used in normalization
    Returns:
        Unnormalized values (approximate to original)
    """
    x_norm = (x_scaled + 1) / 2
    x_log = x_norm * np.log1p(max_value)
    return np.expm1(x_log)
