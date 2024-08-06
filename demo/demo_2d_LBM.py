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
from src.LBM.utils import mkdir, save_img, CellType, KBCType
from tqdm import tqdm


def main(
    res: List[int] = [130, 130],
    total_steps: int = 350,
    dt: float = 1.0,
    dx: float = 1.0,
):
    dim = 2
    Q = 9

    KBC_sigma = 0.05
    KBC_kappa = 80.0

    c = dx / dt
    cs2 = c * c / 3.0

    Re = 3000.0
    Vmax = 0.2
    Lmax = max(res) * dx
    visc = Vmax * Lmax / Re
    tau = 0.5 + visc / cs2

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
        density_gas=0.038,
        density_fluid=0.265,
        contact_angle=torch.Tensor([0.5 * math.pi]).to(device).to(dtype),
        Q=Q,
        tau=tau,
        k=0.33,
    )

    # create a simulation runner
    simulationRunner = SimulationRunner(parameters=simulationParameters)

    # initialize all the required grids
    flags = torch.zeros((batch_size, 1, *res)).to(device).to(torch.uint8)
    vel = torch.zeros((batch_size, dim, *res)).to(device).to(dtype)
    density = torch.zeros((batch_size, 1, *res)).to(device).to(dtype)
    force = torch.zeros((batch_size, dim, *res)).to(device).to(dtype)
    f = torch.zeros((batch_size, Q, *res)).to(device).to(dtype)

    # create prop, macro and collision
    prop = simulationRunner.create_propagation()
    macro = simulationRunner.create_macro_compute()
    collision = simulationRunner.create_collision()

    # initialize the domain
    # wall = ["    "]
    flags[...] = int(CellType.FLUID)

    path = pathlib.Path(__file__).parent.absolute()
    mkdir(f"{path}/demo_data_LBM_{dim}d/")
    fileList = []
    density[...] = 0.265
    for j in range(res[0]):
        for i in range(res[1]):
            vel[:, 1, j, i] = (
                KBC_sigma * Vmax * math.sin(2.0 * math.pi * (1.0 * i / res[1] + 0.25))
            )
            if j <= (res[0] / 2.0):
                vel[:, 0, j, i] = Vmax * math.tanh(
                    KBC_kappa * (1.0 * j / res[0] - 0.25)
                )
            else:
                vel[:, 0, j, i] = Vmax * math.tanh(
                    KBC_kappa * (-1.0 * j / res[0] + 0.75)
                )
    f = collision.get_feq_(dx=dx, dt=dt, rho=density, vel=vel, force=force)

    for step in tqdm(range(total_steps)):
        f = prop.propagation(f=f, flags=flags)

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
            KBC_type=None,
        )

        simulationRunner.step()

        if step % 10 == 0:
            filename = str(path) + "/demo_data_LBM_{}d/{:03}.png".format(dim, step + 1)
            vort = macro.get_vort(vel=vel, dx=dx)
            save_img(vort, filename=filename)
            fileList.append(filename)

    #  VIDEO Loop
    writer = imageio.get_writer(f"{path}/{dim}d_LBM.mp4", fps=25)

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
        default=[130, 130],
        help="Simulation size of the current simulation currently only square",
    )

    parser.add_argument(
        "--total_steps",
        type=int,
        default=1000,
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
