import sys
import numpy as np
import pathlib
import torch
import imageio
import argparse
import math
from typing import List

sys.path.append("../")

from src.LBM.simulation import SimulationParameters, SimulationRunner
from src.LBM.utils import (
    mkdir,
    save_img,
    CellType,
    KBCType,
    create_2d_meshgrid_tensor,
    ObsType,
    dump_2d_plt_file_single,
    get_staggered,
)
from tqdm import tqdm


def CalculateObsPhi(r: torch.Tensor, cylinder_centers: torch.Tensor, cylinder_radius: float):
    # phi < 0: obs, phi > 0: fluid
    x = r[:, 0:1, ...]
    y = r[:, 1:2, ...]

    dim = r.shape[1]
    batch_size = cylinder_centers.shape[0]
    center = cylinder_centers.reshape(batch_size, dim, *([1] * dim))

    x2 = (x - center[:, 0:1, ...]) * (x - center[:, 0:1, ...])
    y2 = (y - center[:, 1:2, ...]) * (y - center[:, 1:2, ...])
    phi_obs = x2 + y2 - cylinder_radius * cylinder_radius
    phi_obs = torch.where(phi_obs > 0, torch.sqrt(phi_obs), -torch.sqrt(-phi_obs))
    
    return phi_obs


def main(
    res: List[int] = [400, 800],
    total_steps: int = 350,
    dt: float = 1.0,
    dx: float = 1.0,
):
    dim = 2
    Q = 9

    c = dx / dt
    cs2 = c * c / 3.0

    Re = 5000.0
    Vmax = 0.05
    D = 20
    Lmax = max(res) * dx
    visc = Vmax * Lmax / Re
    tau = 0.5 + visc / cs2
    print("Re = {}, tau = {}".format(Re, tau))

    density_fluid = 1.0

    # dimension of the
    batch_size = 1

    # use cuda if exists
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    # set up the size of the simulation
    simulation_size = (batch_size, 1, *res)

    # set up the simulation parameters
    simulationParameters = SimulationParameters(
        dim=dim,
        dtype=dtype,
        device=device,
        simulation_size=simulation_size,
        dt=dt,
        density_gas=0,
        density_fluid=density_fluid,
        contact_angle=torch.Tensor([0.5 * math.pi]).to(device).to(dtype),
        Q=Q,
        tau=tau,
        k=0.33,
    )

    # create a simulation runner
    simulationRunner = SimulationRunner(parameters=simulationParameters)

    # initialize all the required grids
    flags = torch.zeros((batch_size, 1, *res)).to(device).to(torch.uint8)
    phi_obs = torch.ones((batch_size, 1, *res)).to(device).to(dtype)
    phi_obs_vel = torch.zeros((batch_size, dim, *res)).to(device).to(dtype)
    vel = torch.zeros((batch_size, dim, *res)).to(device).to(dtype)
    density = torch.zeros((batch_size, 1, *res)).to(device).to(dtype)
    force = torch.zeros((batch_size, dim, *res)).to(device).to(dtype)
    f = torch.zeros((batch_size, Q, *res)).to(device).to(dtype)

    # create prop, macro and collision
    prop = simulationRunner.create_propagation()
    macro = simulationRunner.create_macro_compute()
    collision = simulationRunner.create_collision_MRT()
    collision.preset_KBC(dx=dx, dt=dt, tau=tau)
    solid_fluid_coupling = simulationRunner.create_LBM_fluid_solid_coupling()

    # initialize the domain
    # wall = ["    "]
    flags[...] = int(CellType.OBSTACLE)
    flags[..., 1:-1, 1:-1] = int(CellType.FLUID)
    flags[..., 0, 1:-1] = int(CellType.INFLOW)
    flags[..., -1, 1:-1] = int(CellType.OUTFLOW)
    inflow_vel = torch.Tensor([0.0, Vmax]).to(device).to(dtype)

    path = pathlib.Path(__file__).parent.absolute()
    mkdir(f"{path}/demo_data_LBM_cylinder_{dim}d/")
    fileList = []
    density[...] = density_fluid

    cylinder_center = torch.Tensor([[10 * D, 10 * D]]).to(device).to(dtype)
    cylinder_radius = D / 2.0
    mesh_grid = (
        create_2d_meshgrid_tensor(size=simulation_size, device=device, dtype=dtype)
        + 0.5 * dx
    )
    phi_obs = CalculateObsPhi(r=mesh_grid, cylinder_centers=cylinder_center, cylinder_radius=cylinder_radius)
    flags[phi_obs <= 0] = int(CellType.OBSTACLE)

    vel = torch.where(
        flags == int(CellType.INFLOW),
        inflow_vel.reshape(-1, dim, *([1] * dim)),
        vel,
    )
    f = collision.get_feq_(dx=dx, dt=dt, rho=density, vel=vel, force=force)

    for step in tqdm(range(total_steps)):
        density, vel, f = prop.propagation(
            f=f,
            flags=flags,
            rho=density,
            vel=vel,
            inflow_density=density_fluid,
            inflow_vel=inflow_vel,
        )
        f = solid_fluid_coupling.solve_boundary(
            dt=dt,
            dx=dx,
            rho=density,
            vel=vel,
            flags=flags,
            f=f,
            phi_obs=phi_obs,
            obs_vel=phi_obs_vel,
            tau=tau,
        )

        density, vel = macro.macro_compute(
            dx=dx, dt=dt, f=f, rho=density, vel=vel, flags=flags
        )

        f = collision.collision(
            dx=dx,
            dt=dt,
            f=f,
            rho=density,
            vel=vel,
            flags=flags,
            force=force,
            KBC_type=int(KBCType.KBC_C),
        )

        simulationRunner.step()

        if step % 1000 == 0:
            filename = str(path) + "/demo_data_LBM_cylinder_{}d/{:03}.png".format(
                dim, step + 1
            )
            vort = macro.get_vort(vel=vel, dx=dx)
            save_img(vort, filename=filename)
            fileList.append(filename)

    vel_mac = get_staggered(vel)
    save_path = str(path) + "/demo_data_LBM_cylinder_{}d/{:03}.plt".format(
        dim, step + 1
    )
    dump_2d_plt_file_single(
        filename=save_path,
        np_density=density,
        np_u=vel_mac[0],
        np_v=vel_mac[1],
        B=0,
        C=0,
    )

    #  VIDEO Loop
    writer = imageio.get_writer(f"{path}/{dim}d_LBM_cylinder.mp4", fps=25)

    for im in fileList:
        writer.append_data(imageio.imread(im))
    writer.close()


if __name__ == "__main__":
    torch.set_printoptions(precision=3, linewidth=1000, profile="full", sci_mode=False)
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter, allow_abbrev=False
    )
    parser.add_argument(
        "--res",
        type=int,
        nargs="+",
        default=[800, 400],
        help="Simulation size of the current simulation currently only square",
    )

    parser.add_argument(
        "--total_steps",
        type=int,
        default=50000,
        help="For how many step to run the simulation",
    )

    parser.add_argument(
        "--dt",
        type=float,
        default=1.0,
        help="Delta t of the simulation",
    )

    parser.add_argument(
        "--dx",
        type=float,
        default=1.0,
        help="Delta x of the simulation",
    )

    opt = vars(parser.parse_args())
    print(opt)
    main(**opt)
