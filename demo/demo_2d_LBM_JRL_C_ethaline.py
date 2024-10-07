import sys, os
import numpy as np
import h5py
import pathlib
import torch
import torch.nn.functional as F
import imageio
import argparse
import math
from typing import List

sys.path.append("../")

from src.LBM.simulation import SimulationParameters, SimulationRunner
from src.LBM.LBM_collision import LBMCollision2d
from src.LBM.utils import (
    mkdir,
    save_img,
    CellType,
    KBCType,
    ObsType,
    get_staggered,
    dump_2d_plt_file_C_rho,
    create_2d_meshgrid_tensor,
    AxiSymmetricType,
    UnionPhiObs
)
from tqdm import tqdm


def CalculateObsPhi_forC(r: torch.Tensor, box_center: torch.Tensor, box_radius: float, is_negative: bool = True):
    # phi < 0: obs, phi > 0: fluid
    res_y = r.shape[-2]
    x = r[:, 0:1, ...]
    y = r[:, 1:2, ...]
    ty = (y - box_center[1]) - box_radius
    phi_obs = ty
    
    x1 = (box_center + box_radius)[0]
    y1 = (box_center + box_radius)[1]
    A_, B_, C_ = 1.0, 1.0, -(x1 + y1)
    align_dist = (A_ * x + B_ * y + C_) / math.sqrt(2)
    phi_obs = torch.maximum(phi_obs, align_dist)
    
    if is_negative:
        ty = -(y - res_y)
        phi_obs = UnionPhiObs(phi_obs, ty, alpha=1)
    
    return phi_obs


def main(
    res: List[int] = [300, 300],
    balance_time: int = 350,
    Re: float = 0,
    Pe: float = 0,
    vel_obs_real: float = 100e-6,
    gravity_strength_real: float = -9.8,
    Rg: float = 3,
    fluid: str = "ethaline",
    refine: int =  1,
    is_negative: bool = False,
):
    dim = 2
    Q = 9

    positive_fix = "" if is_negative else "_positive"
    load_fluid = True
    saved_fluid_name=f"Re_{Re if Re > 3e-5 else 3e-5}.h5"
    
    path = pathlib.Path(__file__).parent.absolute()
    prefix = f"_{fluid}_g{int(abs(gravity_strength_real))}"
    h5py_fluid_filename = f"{path}/{prefix}/{saved_fluid_name}"
    mkdir(f"{path}/{prefix}")
    if load_fluid and not os.path.exists(h5py_fluid_filename):
        load_fluid = False
        print("Error: We cannot find this fluid field")
        exit(0)
    
    res = [refine * x for x in res]

    # === Parameters of general
    radius_obs = refine * 6 + 0.50000000000001  # 12.5+, TODO: int(12.0 * (min(res) - 12) / 100)
    radius_obs_real = 12.5e-6  # 12.5 um, the radius of obstacle [m]
    dt = 1  # [s] Warning: cannot be set as any other value!
    dx = 1  # [m] Warning: cannot be set as any other value!
    c = dx / dt
    cs2 = c * c / 3.0
    inflow_height = refine * (
        (40) + 0.50000000000001
    )  # 10.5+, TODO: int(12.0 * (min(res) - 12) / 100)
    inflow_height_int = int(inflow_height + 0.5 * dx)

    # === Parameters of convection
    radius_obs_conv = radius_obs
    tau_D = 1.0
    D = cs2 * (tau_D - 0.5)
    vel_obs_conv = Pe * D / radius_obs_conv
    if vel_obs_conv > 0.1:
        vel_obs_conv = 0.1
        D = vel_obs_conv * radius_obs_conv / Pe
        tau_D = 0.5 + D / cs2

    D_real = vel_obs_real * radius_obs_real / Pe
    dt_conv_real = (dt * vel_obs_conv / radius_obs_conv) / (
        vel_obs_real / radius_obs_real
    )

    print("=========== Parameters in real world ===========")
    print("velocity of pin (Real) = {}".format(vel_obs_real))
    print("radius of pin (Real) = {}".format(radius_obs_real))
    print("D (Real) = {}".format(D_real))
    print("time step of convection (Real) = {}".format(dt_conv_real))
    print("=========== Parameters in real world End ===========\n\n\n")

    print("=========== Parameters in Computational world ===========")
    print("velocity of pin for convection field = {}".format(vel_obs_conv))
    print("radius of pin = {}".format(radius_obs))
    print("time step = {}".format(dt))
    print("grid step = {}".format(dx))
    print("D = {}".format(D))
    print("tau_D = {}".format(tau_D))
    print("=========== Parameters in Computational world End ===========\n\n\n")

    # use cuda if exists
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    batch_size = 1
    simulation_size = (batch_size, 1, *res)

    # set up the simulation parameters
    simulationParameters = SimulationParameters(
        dim=dim,
        dtype=dtype,
        device=device,
        simulation_size=simulation_size,
        dt=dt,
        axisymmetric_type=int(AxiSymmetricType.LINE_X_EQ_0),
        contact_angle=None,
        Q=Q
    )

    # create a simulation runner
    simulationRunner = SimulationRunner(parameters=simulationParameters)

    # initialize all the required grids
    C_flags = torch.zeros((batch_size, 1, *res)).to(device).to(torch.uint8)
    phi_obs = torch.ones((batch_size, 1, *res)).to(device).to(dtype)
    C = torch.ones((batch_size, 1, *res)).to(device).to(dtype)
    C_vel = torch.zeros((batch_size, dim, *res)).to(device).to(dtype)
    h = torch.zeros((batch_size, Q, *res)).to(device).to(dtype)
    mesh_grid = (
        create_2d_meshgrid_tensor(size=simulation_size, device=device, dtype=dtype)
        + 0.5 * dx
    )

    # create external force, advection and pressure projection
    prop = simulationRunner.create_propagation()
    macro = simulationRunner.create_macro_compute()
    collision_h = simulationRunner.create_collision()
    collision_h.preset_KBC(dx=dx, dt=dt, tau=1.0, tau_D=tau_D)

    # Mark the domain:
    Rg_radius = int(Rg * radius_obs + 0.5 * dx)
    radius_obs_int = int(radius_obs + 0.5 * dx)
    C_flags[...] = int(CellType.OUTFLOW)
    C_flags[..., 1:-1, 1:-1] = int(CellType.FLUID)
    C_flags[..., 0] = int(CellType.FLUID)  # left
    C_flags[..., -1, :] = int(CellType.OBSTACLE) if is_negative else int(CellType.INFLOW_2)  # up
    C_flags[..., :-1, -1] = int(CellType.INFLOW_2) if is_negative else int(CellType.OUTFLOW)  # right
    C_flags[..., 0, Rg_radius:] = int(CellType.INFLOW_2)  # down
    C_flags[..., 0:inflow_height_int, 0:Rg_radius] = int(CellType.OBSTACLE)  # tip
    C_flags[..., inflow_height_int - 1, 0:radius_obs_int] = int(CellType.INFLOW)  # tip-head

    h5py_C_filename = f"{path}/{prefix}/C_balance{positive_fix}.h5"
    if os.path.exists(h5py_C_filename):
        load_C = True
        write_C = False
    else:
        load_C = False
        write_C = True
    dir_path = f"{path}/{prefix}/demo_data_LBM_{dim}d{positive_fix}_res{min(res)}_Re{Re}_Pe{int(Pe * 10000)}/"
    mkdir(dir_path)
    record_path = f"{path}/{prefix}/records{positive_fix}/"
    mkdir(record_path)
    fileList = []

    # A = pi * r * r
    area_array = math.pi * torch.Tensor([i * i for i in range(radius_obs_int + 1)]).to(
        device
    ).to(dtype)
    area_array = area_array[1:] - area_array[:-1]

    # Initialize your C domain
    r = mesh_grid[:, 0:1, ...]
    h = collision_h.get_feq_(dx=dx, dt=dt, rho=C, vel=C_vel, tau=tau_D)
    box_radius = Rg_radius + 100.0  # We need double variable
    
    balance_name = str(dir_path) + "/balance.txt"
    fo_balance = open(balance_name, "w")
    fo_balance.write("Step Inflow\n")
    fo_balance.close()
    
    walks = inflow_height
    walks_int = inflow_height_int
    inflow_record = [0, 0]
    current_record = [0, 0]

    box_center = (
        torch.Tensor([Rg_radius - box_radius, walks - box_radius])
        .to(device)
        .to(dtype)
    )
    # phi < 0: obs, phi > 0: fluid
    phi_obs = CalculateObsPhi_forC(r=mesh_grid, box_center=box_center, box_radius=box_radius, is_negative=is_negative)
    
    # 1. Run balance
    is_obs = phi_obs <= 0
    C_flags[is_obs] = int(CellType.OBSTACLE)
    C_flags[..., walks_int - 1, 0:radius_obs_int] = int(CellType.INFLOW)
    C[C_flags == int(CellType.OBSTACLE)] = 0.0
    if load_C:
        h5_data = h5py.File(h5py_C_filename)
        C = torch.from_numpy(np.asarray(h5_data["C"])).to(dtype).to(device)
        h = torch.from_numpy(np.asarray(h5_data["h"])).to(dtype).to(device)
        h5_data.close()
    elif write_C:
        balance_step = int(balance_time / dt_conv_real * dt)
        balance_interval = int(balance_step // 100)
        for step in tqdm(range(balance_step)):
            C_vel[...] = 0

            C, _, h = prop.propagation(
                f=h,
                flags=C_flags,
                rho=C,
                vel=C_vel,
                is_convection=True,
                inflow_density=0,
            )
            C = macro.macro_compute_C(dx=dx, dt=dt, h=h, C=C, flags=C_flags, vel=C_vel, mesh_grid=mesh_grid)
            h = collision_h.collision(
                dx=dx,
                dt=dt,
                f=h,
                rho=C,
                vel=C_vel,
                flags=C_flags,
                force=None,
                mesh_grid=mesh_grid,
                is_convection=True,
                KBC_type=int(KBCType.KBC_C),
            )
            
            if step % balance_interval == 0:
                fo_balance = open(balance_name, "a+")
                new_inflow = (
                    (C)[..., walks_int, 0:radius_obs_int].flatten() * area_array
                ).sum()
                fo_balance.write(
                    "{} {}\n".format((step + 1) * dt_conv_real, new_inflow.item())
                )
                fo_balance.close()
        h5_data = h5py.File(h5py_C_filename, "w")
        h5_data.create_dataset("C", data=C.cpu().numpy())
        h5_data.create_dataset("h", data=h.cpu().numpy())
        h5_data.close()
    filename = str(dir_path) + "/C_000.png"
    save_img((C), filename=filename)
    vel_mac = get_staggered(C_vel)
    save_path = str(dir_path) + "/{:03}.dat".format(0)
    dump_2d_plt_file_C_rho(
        filename=save_path,
        np_C=(C),
        np_density=phi_obs,
        np_u=vel_mac[0],
        np_v=vel_mac[1],
        B=0,
        C=0,
    )

    # Walk on boundary
    record_name = str(record_path) + f"/record_res{min(res)}_Re{Re}_Pe{int(Pe * 10000)}.txt"
    fo = open(record_name, "w")
    fo.write("Step Inflow Current u, v\n")
    fo.close()
    conv_timer = 0.0
    x_interval = 0.1
    h5_data = h5py.File(h5py_fluid_filename)
    vel_field = torch.from_numpy(np.asarray(h5_data["vel"])).to(dtype).to(device)
    vel_num = vel_field.shape[0]
    total_steps = int((res[0] - 1 - inflow_height_int) / vel_obs_conv)
    print("total_steps = {}".format(total_steps))
    for step in tqdm(range(total_steps)):
        # 1. Every time step, update the position of obstacle
        walks_int = int(walks + 0.5 * dx)
        box_center[1] = walks - box_radius
        phi_obs = CalculateObsPhi_forC(r=mesh_grid, box_center=box_center, box_radius=box_radius, is_negative=is_negative)
        if not is_negative:
            # positive part should give a different treatment
            C_flags[..., -1, :] = int(CellType.INFLOW_2)  # up
        C_flags[phi_obs < 0] = int(CellType.OBSTACLE)
        C_flags[..., walks_int - 1, 0:radius_obs_int] = int(CellType.INFLOW)
        C[C_flags == int(CellType.OBSTACLE)] = 0.0

        # 2. Every time step, update fluid field
        if load_fluid:
            x_real = conv_timer * vel_obs_real
            x_comp = x_real / radius_obs_real * radius_obs
            batch_idx = int(x_comp / x_interval)
            if batch_idx + 1 < vel_num:
                t = x_comp / x_interval - batch_idx
                vel = vel_field[batch_idx] * (1 - t) + vel_field[batch_idx + 1] * t
            else:
                vel = vel_field[batch_idx]
            vel = vel.unsqueeze(0)
            
            C_vel = vel / 1 * vel_obs_conv
            C_vel[:, 0, 0: walks_int, 0:Rg_radius] = 0
            C_vel[:, 1, 0: walks_int, 0:Rg_radius] = vel_obs_conv

        # 3. Decide if to update convection field in this step.
        # ur = C_vel[:, 0:1, ...]
        # r = mesh_grid[:, 0:1, ...]
        # C_vel[:, 0:1, ...] = ur - D / r
        C, _, h = prop.propagation(
            f=h,
            flags=C_flags,
            phi_obs=phi_obs,
            rho=C,
            vel=C_vel,
            is_convection=True,
            inflow_density=0,
        )

        C = macro.macro_compute_C(dx=dx, dt=dt, h=h, C=C, flags=C_flags, vel=C_vel, mesh_grid=mesh_grid)
        # C = torch.clamp(C, 0, 1)

        C_vel_hard_copy = C_vel + 0
        h = collision_h.collision(
            dx=dx,
            dt=dt,
            f=h,
            rho=C,
            vel=C_vel,
            flags=C_flags,
            force=None,
            mesh_grid=mesh_grid,
            is_convection=True,
            KBC_type=int(KBCType.KBC_C),
        )
        C_vel = C_vel_hard_copy

        conv_timer += dt_conv_real

        # 3.9 If convection field updated, record data
        # dC_z = ((C)[..., walks_int, 0:radius_obs_int] - (C)[..., walks_int - 1, 0:radius_obs_int]).flatten()
        # r = mesh_grid[..., 0:1, walks_int, 0:radius_obs_int + 1].flatten()
        # rC = r * (C)[..., walks_int, 0:radius_obs_int + 1].flatten()
        # dC_r = (rC[1:] - rC[:-1]) / r[:-1]
        # dC = torch.stack((dC_r, dC_z), dim=0)
        # vel_C_D = ((C * C_vel) / (cs2 * (tau_D - 0.5)))[..., walks_int, 0:radius_obs_int].reshape(dim, -1)
        # dC_norm = (-dC + vel_C_D).norm(dim=0)
        dC_norm = (C)[..., walks_int, 0:radius_obs_int].flatten()
        new_inflow = (dC_norm * area_array).sum()
        # new_inflow = ((C)[..., walks_int, 0:radius_obs_int].flatten() * area_array).sum()
        inflow_record[0] = (inflow_record[0] * inflow_record[1] + new_inflow) / (
            inflow_record[1] + 1
        )
        inflow_record[1] += 1

        # dC_z = ((C)[..., walks_int + 1, 0:radius_obs_int] - (C)[..., walks_int, 0:radius_obs_int]).flatten()
        # r = mesh_grid[..., 0:1, walks_int + 1, 0:radius_obs_int + 1].flatten()
        # rC = r * (C)[..., walks_int + 1, 0:radius_obs_int + 1].flatten()
        # dC_r = (rC[1:] - rC[:-1]) / r[:-1]
        # dC = torch.stack((dC_r, dC_z), dim=0)
        # vel_C_D = ((C * C_vel) / (cs2 * (tau_D - 0.5)))[..., walks_int + 1, 0:radius_obs_int].reshape(dim, -1)
        # dC_norm = (-dC + vel_C_D).norm(dim=0)
        dC_norm = (C)[..., walks_int + 1, 0:radius_obs_int].flatten()
        new_current = (dC_norm * area_array).sum()
        # new_current = ((C)[..., walks_int + 1, 0:radius_obs_int].flatten() * area_array).sum()
        current_record[0] = (
            current_record[0] * current_record[1] + new_current
        ) / (current_record[1] + 1)
        current_record[1] += 1

        simulationRunner.step()

        # 4. Simulation in this epoch finished, decide if to dump any files in this step
        if math.floor(walks) == math.floor(walks + dt * vel_obs_conv) - 1:
            inflow = inflow_record[0]
            current = current_record[0]
            inflow_record[0], inflow_record[1] = 0, 0
            current_record[0], current_record[1] = 0, 0
            record_u = (
                C_vel[..., 0, walks_int + 1, 0:radius_obs_int].flatten() * area_array
            ).sum()
            record_v = (
                C_vel[..., 1, walks_int + 1, 0:radius_obs_int].flatten() * area_array
            ).sum()

            filename = str(dir_path) + "/C_{:03}.png".format(step + 1)
            save_img((C), filename=filename)
            fileList.append(filename)

            fo = open(record_name, "a+")
            fo.write(
                "{} {} {} {} {}\n".format(
                    (step + 1) * dt_conv_real * vel_obs_real * 1e6,
                    inflow.item(),
                    current.item(),
                    record_u.item(),
                    record_v.item(),
                )
            )
            fo.close()

            vel_mac = get_staggered(C_vel)
            save_path = str(dir_path) + "/{:03}.dat".format(step + 1)
            dump_2d_plt_file_C_rho(
                filename=save_path,
                np_C=(C),
                np_density=phi_obs,
                np_u=vel_mac[0],
                np_v=vel_mac[1],
                B=0,
                C=0,
            )

        walks += vel_obs_conv * dt

    h5_data.close()

    #  VIDEO Loop


if __name__ == "__main__":
    torch.set_printoptions(precision=3, linewidth=1000, profile="full", sci_mode=False)
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, allow_abbrev=False
    )
    parser.add_argument(
        "--res",
        type=int,
        nargs="+",
        default=[128, 64],
        help="Simulation size of the current simulation currently only square",
    )

    parser.add_argument(
        "--balance_time",
        type=float,
        default=600,
        help="For how many time to run the simulation for balance",
    )

    parser.add_argument(
        "--Re",
        type=float,
        default=1e-4,
        help=("Re number"),
    )

    parser.add_argument(
        "--Pe",
        type=float,
        default=0.01,
        help=("Pe number"),
    )

    parser.add_argument(
        "--vel_obs_real",
        type=float,
        default=21e-6,
        help=("Diffuse coeff"),
    )
    
    parser.add_argument(
        "--gravity_strength_real",
        type=float,
        default=-9.8,
        help=("Gravity acceleration"),
    )

    parser.add_argument(
        "--Rg",
        type=float,
        default=4,
        help=("Rg is the ratio of the platform and the tip radius"),
    )
    
    parser.add_argument(
        "--fluid", type=str, default="ethaline", help=("The type of fluid")
    )

    parser.add_argument(
        "--refine", type=int, default=1, help="mesh refinemnet"
    )
    
    parser.add_argument('--is_negative', dest='is_negative', action='store_true')
    parser.add_argument('--no-is_negative', dest='is_negative', action='store_false')
    parser.set_defaults(is_negative=True)

    opt = vars(parser.parse_args())
    print(opt)
    main(**opt)
