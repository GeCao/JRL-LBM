import sys
import os
import numpy as np
import pathlib
import torch
import torch.nn.functional as F
import math
import mcubes
from typing import List

sys.path.append("../")

from src.LBM.utils import (
    read_2d_plt_file_C_rho,
    dump_2d_plt_file_C_rho,
    create_3d_meshgrid_tensor,
    dump_smoke_pbrt,
    export_asset,
    mkdir,
)
from tqdm import tqdm


data_path = os.path.abspath(os.path.curdir)
data_path = os.path.join(data_path, "_InfinitePlane_g9/demo_data_LBM_2d_res64_Re0.0003_Pe3401")
files = os.listdir(data_path)
files = [file for file in files if ".dat" in file]
print(files)

for file in tqdm(files):
    filepath = os.path.join(data_path, file)
    
    np_C, np_density, np_u, np_v = read_2d_plt_file_C_rho(filepath)
    # flip
    # np_C = np.flip(np_C, axis=-2)
    # np_density = np.flip(np_density, axis=-2)
    # np_u = np.flip(np_u, axis=-2)
    # np_v = -np.flip(np_v, axis=-2)
    
    dumpfile = filepath
    dumppath = os.path.join(data_path, dumpfile)
    dump_2d_plt_file_C_rho(dumppath, np_C=np_C, np_density=np_density, np_u=np_u, np_v=np_v, B=0, C=0)