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
    UnionPhiObs,
)
from tqdm import tqdm


def CalculateObsPhi(r: torch.Tensor, box_center: torch.Tensor, box_radius: float):
    # phi < 0: obs, phi > 0: fluid
    res_y = r.shape[-2]
    tx = (r[:, 0:1, ...] - box_center[0]) - box_radius
    ty = (r[:, 1:2, ...] - box_center[1]) - box_radius
    phi_obs = torch.maximum(tx, ty)
    ty = -(r[:, 1:2, ...] - res_y)
    phi_obs = UnionPhiObs(phi_obs, ty, alpha=1)
    
    return phi_obs


def main(
    res: List[int] = [300, 300],
    Re: float = 0,
    vel_obs_real: float = 100e-6,
    gravity_strength_real: float = -9.8,
    Rg: float = 3,
):
    dim = 2
    Q = 9
    path = pathlib.Path(__file__).parent.absolute()
    prefix = f"_InfinitePlane_g{int(abs(gravity_strength_real))}"
    mkdir(f"{path}/{prefix}")
    saved_fluid_name = f"Re_{Re}.h5"

    # === Parameters of fluid
    radius_obs = 6 + 0.50000000000001  # 12.5+, TODO: int(12.0 * (min(res) - 12) / 100)
    dt = 1  # [s] Warning: cannot be set as any other value!
    dx = 1  # [m] Warning: cannot be set as any other value!
    tau = 1.0
    c = dx / dt  # 1
    cs2 = c * c / 3.0
    visc = cs2 * (tau - 0.5)  # 0.1667 [m2/s]
    density_fluid = 1  # Has to be 1, if not, change solid-fluid-coupling term
    density_wall = 1
    mu = visc * density_fluid
    vel_obs = Re * visc / radius_obs  # 6.785e-6
    inflow_height = (
        (20) + 0.50000000000001
    )  # 10.5+, TODO: int(12.0 * (min(res) - 12) / 100)
    inflow_height_int = int(inflow_height + 0.5 * dx)

    density_real = 1000.0
    radius_obs_real = 1.25e-5  # 12.5 um, the radius of obstacle [m]
    dt_real = (dt * vel_obs / radius_obs) / (
        vel_obs_real / radius_obs_real
    )  # 2.458e-8 s
    dx_real = (dx / radius_obs) * radius_obs_real  # 1- um per grid step
    visc_real = (radius_obs_real * vel_obs_real) / Re  # 8.9e-6
    mu_real = visc_real * density_real  # 8.9e-3
    gravity_strength = gravity_strength_real / (vel_obs_real * vel_obs_real / radius_obs_real) * (vel_obs * vel_obs / radius_obs)

    # === End of parameters of fluid

    print("=========== Parameters in real world ===========")
    print("density (Real) = {}".format(density_real))
    print("velocity of pin (Real) = {}".format(vel_obs_real))
    print("radius of pin (Real) = {}".format(radius_obs_real))
    print("time step (Real) = {}".format(dt_real))
    print("grid step (Real) = {}".format(dx_real))
    print("viscosity (Real) = {}".format(visc_real))
    print("mu = viscosity * rho (Real) = {}".format(mu_real))
    print("g = {}".format(gravity_strength_real))
    print("=========== Parameters in real world End ===========\n\n\n")

    print("=========== Parameters in Computational world ===========")
    print("density = {}".format(density_fluid))
    print("velocity of pin = {}".format(vel_obs))
    print("radius of pin = {}".format(radius_obs))
    print("time step = {}".format(dt))
    print("grid step = {}".format(dx))
    print("viscosity = {}".format(visc))
    print("mu = viscosity * rho = {}".format(mu))
    print("g = {}".format(gravity_strength))
    print("tau = {}".format(tau))
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
        density_fluid=density_fluid,
        axisymmetric_type=int(AxiSymmetricType.LINE_X_EQ_0),
        contact_angle=None,
        Q=Q,
        tau=tau,
        gravity_strength=gravity_strength,
    )

    # create a simulation runner
    simulationRunner = SimulationRunner(parameters=simulationParameters)

    # initialize all the required grids
    flags = torch.zeros((batch_size, 1, *res)).to(device).to(torch.uint8)
    phi_obs = torch.ones((batch_size, 1, *res)).to(device).to(dtype)
    vel = torch.zeros((batch_size, dim, *res)).to(device).to(dtype)
    density = torch.zeros((batch_size, 1, *res)).to(device).to(dtype)
    f = torch.zeros((batch_size, Q, *res)).to(device).to(dtype)
    mesh_grid = (
        create_2d_meshgrid_tensor(size=simulation_size, device=device, dtype=dtype)
        + 0.5 * dx
    )

    # create external force, advection and pressure projection
    prop = simulationRunner.create_propagation()
    macro = simulationRunner.create_macro_compute()
    collision_f = simulationRunner.create_collision()
    collision_f.preset_KBC(dx=dx, dt=dt, tau=tau, tau_D=None)
    solid_fluid_coupling = simulationRunner.create_LBM_fluid_solid_coupling()

    # Mark the domain:
    Rg_radius = int(Rg * radius_obs + 0.5 * dx)
    flags[...] = int(CellType.OUTFLOW)
    flags[..., 1:-1, 1:-1] = int(CellType.FLUID)
    flags[..., 0] = int(CellType.FLUID)
    flags[..., -1, :] = int(CellType.OBSTACLE)
    flags[..., 0:inflow_height_int, 0:Rg_radius] = int(CellType.OBSTACLE)

    h5py_fluid_filename = f"{path}/{prefix}/{saved_fluid_name}"
    if os.path.exists(h5py_fluid_filename):
        print("The fluid field is already all set")
        exit(0)
    data_dir = f"{path}/{prefix}/demo_data_Re{Re}/"
    mkdir(data_dir)

    # Initialize your fluid domain
    density[flags == int(CellType.OBSTACLE)] = density_wall
    density[flags != int(CellType.OBSTACLE)] = density_fluid
    force = density * collision_f._gravity
    f = collision_f.get_feq_(dx=dx, dt=dt, rho=density, vel=vel, tau=tau, force=force)
    r = mesh_grid[:, 0:1, ...]
    box_radius = Rg_radius + 100.0  # We need double variable
    
    walks = inflow_height
    walks_int = inflow_height_int

    # 1. Run balance
    box_center = (
        torch.Tensor([Rg_radius - box_radius, walks - box_radius])
        .to(device)
        .to(dtype)
    )
    # phi < 0: obs, phi > 0: fluid
    phi_obs = CalculateObsPhi(r=mesh_grid, box_center=box_center, box_radius=box_radius)
    flags[..., 0:walks_int, 0:Rg_radius] = int(CellType.OBSTACLE)
    vel[..., 0, 0:walks_int, 0:Rg_radius] = 0
    vel[..., 1, 0:walks_int, 0:Rg_radius] = vel_obs

    # Walk on boundary
    fluid_timer = 0.0
    x_interval = 0.1
    vel_field = torch.zeros((0, dim, *res), dtype=dtype).to(device)
    total_steps = int((res[0] - 1 - inflow_height_int) / vel_obs)
    print("total_steps = {}".format(total_steps))
    for step in tqdm(range(total_steps)):
        # 1. Every time step, update the position of obstacle
        walks_int = int(walks + 0.5 * dx)
        box_center[1] = walks - box_radius
        phi_obs = CalculateObsPhi(r=mesh_grid, box_center=box_center, box_radius=box_radius)
        phi_obs_vel = torch.zeros_like(vel)
        phi_obs_vel[..., 1, 0:walks_int, 0:Rg_radius] = vel_obs

        # 2. Every time step, update fluid field
        flags[..., 0:walks_int, 0:Rg_radius] = int(CellType.OBSTACLE)
        vel[..., 0, 0:walks_int, 0:Rg_radius] = 0
        vel[..., 1, 0:walks_int, 0:Rg_radius] = vel_obs

        vel_prev = vel + 0  # A deep copy
        
        # 3. main fluid simulation part
        f = prop.propagation(f=f, flags=flags, phi_obs=phi_obs, rho=density, vel=vel)
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
        f = collision_f.collision(
            dx=dx, dt=dt, f=f, rho=density, vel=vel, flags=flags, mesh_grid=mesh_grid, force=force
        )
        
        # postprocessing: write fluid in a decent time spacing
        x_real = fluid_timer * vel_obs_real
        x_comp = x_real / radius_obs_real * radius_obs
        batch_idx = int(x_comp / x_interval)
        x_real_prev = (fluid_timer - dt_real) * vel_obs_real
        x_comp_prev = x_real_prev / radius_obs_real * radius_obs
        batch_idx_prev = int(x_comp_prev / x_interval)
        if batch_idx_prev < batch_idx:
            t1 = x_comp / x_interval - batch_idx
            t2 = batch_idx - x_comp_prev / x_interval
            h5_vel = (vel * t2 + vel_prev * t1) / (t2 + t1)
            h5_vel = h5_vel / vel_obs  # Normalize to 1
            vel_field = torch.cat((vel_field, h5_vel), dim=0)
            
            vel_mac = get_staggered(h5_vel)
            dump_2d_plt_file_C_rho(
                filename=os.path.join(data_dir, f"{step}.dat"),
                np_C=phi_obs,
                np_density=density,
                np_u=vel_mac[0],
                np_v=vel_mac[1],
                B=0,
                C=0,
            )
        
        fluid_timer += dt_real
        walks += vel_obs * dt

    h5_data = h5py.File(h5py_fluid_filename, "w")
    h5_data.create_dataset("vel", data=vel_field.cpu().numpy())
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
        "--Re",
        type=float,
        default=1e-4,
        help=("Re number"),
    )

    parser.add_argument(
        "--vel_obs_real",
        type=float,
        default=28e-6,
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
        default=200,
        help=("Rg is the ratio of the platform and the tip radius"),
    )

    opt = vars(parser.parse_args())
    print(opt)
    main(**opt)
