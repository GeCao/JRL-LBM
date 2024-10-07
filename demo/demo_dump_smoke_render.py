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
    create_3d_meshgrid_tensor,
    dump_smoke_pbrt,
    export_asset,
    mkdir,
)
from tqdm import tqdm


def main(case_name: str):
    # use cuda if exists
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    path = pathlib.Path(__file__).parent.absolute()
    dir_path = os.path.join(path, case_name)
    dat_files = os.listdir(dir_path)
    dat_files = [filename for filename in dat_files if ".dat" in filename]

    # Sort
    dat_numbers = [int(filename[:-4]) for filename in dat_files]
    dat_numbers.sort()
    dat_files = ["{:03}.dat".format(number) for number in dat_numbers]

    Rg = 3

    n = len(dat_files)
    save_dir = os.path.join(dir_path, "render")
    mkdir(save_dir)
    for i in tqdm(range(n)):
        dat_path = os.path.join(dir_path, dat_files[i])
        dump_path = os.path.join(save_dir, f"render_{i}.pbrt")
        np_C_2d, np_density_2d, np_u_2d, np_v_2d = read_2d_plt_file_C_rho(dat_path)
        # Give a flip in y-axis
        y_flip = True
        if y_flip:
            np_C_2d = np.flip(np_C_2d, axis=-2)
            np_density_2d = np.flip(np_density_2d, axis=-2)
            np_u_2d = np.flip(np_u_2d, axis=-2)
            np_v_2d = np.flip(np_v_2d, axis=-2)
        C_2d = torch.from_numpy(np_C_2d.copy()).to(device).to(dtype)  # [1, 1, H, W//2]
        phi_2d = torch.from_numpy(np_density_2d.copy()).to(device).to(dtype)  # [1, 1, H, W//2]
        C_2d[phi_2d < 0] = 1  # 0 as tip, 1 as far away. Now we set tip is far away
        C_2d = 1.0 - C_2d  # 1 as tip, 0 as far away

        dim = 3
        res = [2 * C_2d.shape[-1], C_2d.shape[-2], 2 * C_2d.shape[-1]]
        D, H, W = res
        simulation_size = (1, 1, *res)
        C = torch.zeros(simulation_size).to(dtype).to(device)
        phi = torch.zeros(simulation_size).to(dtype).to(device)

        meshgrid = create_3d_meshgrid_tensor(
            simulation_size, dtype=dtype, device=device
        )  # [1, dim, D, H, W]
        r = meshgrid - torch.Tensor([W // 2, 0, D // 2]).to(device).to(dtype).reshape(
            (1, dim, *([1] * dim))
        )
        r[:, 1:2, ...] = 0
        r = r.norm(dim=1, keepdim=True)  # [1, 1, D, H, W]
        r = (r - W / 4.0) / (W / 4.0)
        r_lt = r[0, 0, :, :, 0 : W // 2].unsqueeze(-1)  # [D, H, W//2, 1]
        r_rt = r[0, 0, :, :, W // 2 :].unsqueeze(-1)  # [D, H, W//2, 1]
        y_lt = meshgrid[0, 1, :, :, 0 : W // 2].unsqueeze(-1)  # [D, H, W//2, 1]
        y_rt = meshgrid[0, 1, :, :, W // 2 :].unsqueeze(-1)  # [D, H, W//2, 1]
        y_lt = (y_lt - (H / 2.0)) / (H / 2.0)
        y_rt = (y_rt - (H / 2.0)) / (H / 2.0)

        grid_lt = torch.cat((r_lt, y_lt), dim=-1)  # [D, H, W//2, dim]
        grid_rt = torch.cat((r_rt, y_rt), dim=-1)  # [D, H, W//2, dim]

        C_2d = C_2d.repeat(D, 1, 1, 1)
        C_lt = F.grid_sample(input=C_2d, grid=grid_lt, padding_mode="border")
        C_rt = F.grid_sample(input=C_2d, grid=grid_rt, padding_mode="border")
        C = torch.cat((C_lt, C_rt), dim=-1)  # [D, 1, H, W]
        C = C.reshape(*simulation_size)  # [1, 1, D, H, W]
        C = F.pad(C[..., 1:-1, 1:-1, 1:-1], mode="replicate", pad=(1, 1, 1, 1, 1, 1))
        
        phi_2d = phi_2d.repeat(D, 1, 1, 1)
        phi_lt = F.grid_sample(input=phi_2d, grid=grid_lt, padding_mode="border")
        phi_rt = F.grid_sample(input=phi_2d, grid=grid_rt, padding_mode="border")
        phi = torch.cat((phi_lt, phi_rt), dim=-1)  # [D, 1, H, W]
        phi = phi.reshape(*simulation_size)  # [1, 1, D, H, W]
        phi = F.pad(phi[..., 1:-1, 1:-1, 1:-1], mode="replicate", pad=(1, 1, 1, 1, 1, 1))

        dump_smoke_pbrt(dump_path, density=C)

        # Besides from smoke, dump your obstacle also
        max_scale = max(res)
        verts, faces = mcubes.marching_cubes(
            ((-phi).cpu().numpy()[0, 0, ...]), 0
        )
        verts = verts / max_scale
        faces = np.asarray(faces, dtype=np.int32)
        save_path = os.path.join(save_dir, f"geom_{i}.obj")
        verts = torch.from_numpy(verts).to(dtype)
        faces = torch.from_numpy(faces).to(torch.int)
        export_asset(save_path=save_path, vertices=verts, faces=faces)


if __name__ == "__main__":
    case_name = "_45degree_g9_water/demo_data_LBM_2d_res64_Re0.0003_Pe3401"
    main(case_name)
