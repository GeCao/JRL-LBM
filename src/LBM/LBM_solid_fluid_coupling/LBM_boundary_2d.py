import sys
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load
from typing import List, Any

from src.LBM.LBM_solid_fluid_coupling import AbstractLBMBoundary
from src.LBM.utils import CellType, ObsType, create_2d_meshgrid_tensor


cuda_solid_fluid_coupling = load(
    name="solid_fluid_coupling",
    sources=[
        "../src/LBM/LBM_solid_fluid_coupling/cuda/solid_fluid_coupling.cpp",
        "../src/LBM/LBM_solid_fluid_coupling/cuda/solid_fluid_coupling.cu",
    ],
    verbose=True,
    with_cuda=True,
    extra_cuda_cflags=["--use_fast_math"],
)


class Boundary2dKernel(torch.autograd.Function):
    def __init__(
        self,
        device: torch.device = torch.device("cpu"),
        dtype=torch.float32,
        *args,
        **kwargs,
    ):
        self._device = device
        self._dtype = dtype
        super(torch.autograd.Function, self).__init__(*args, **kwargs)

    @staticmethod
    def forward(
        ctx,
        dt: float,
        dx: float,
        tau: float,
        rho: torch.Tensor,
        vel: torch.Tensor,
        flags: torch.Tensor,
        f: torch.Tensor,
        phi_obs: torch.Tensor,
        obs_vel: torch.Tensor,
        is_convection: bool,
        axisymmetric_type: int,
    ) -> Any:
        # Solve fluid-solid coupling only
        dim = 2
        B, Q, H, W = f.shape
        f_new = f + 0.0

        if obs_vel is None:
            obs_vel = torch.Tensor([]).to(phi_obs.device).to(phi_obs.dtype)

        cuda_solid_fluid_coupling.solve_obstacle_2d_forward(
            dt,
            dx,
            tau,
            rho,
            vel,
            flags,
            f,
            f_new,
            phi_obs,
            obs_vel,
            is_convection,
            axisymmetric_type,
        )

        return f_new


class LBMBoundary2d(AbstractLBMBoundary):
    rank = 2

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super(LBMBoundary2d, self).__init__(*args, **kwargs)

        Q = 9

        self._weight = (
            torch.Tensor(
                [
                    4.0 / 9.0,
                    1.0 / 9.0,
                    1.0 / 9.0,
                    1.0 / 9.0,
                    1.0 / 9.0,
                    1.0 / 36.0,
                    1.0 / 36.0,
                    1.0 / 36.0,
                    1.0 / 36.0,
                ]
            )
            .reshape(1, Q, 1, 1)
            .to(self.device)
            .to(self.dtype)
        )

        # x, y direction
        self._e = (
            torch.Tensor(
                [
                    [0, 0],
                    [1, 0],
                    [0, 1],
                    [-1, 0],
                    [0, -1],
                    [1, 1],
                    [-1, 1],
                    [-1, -1],
                    [1, -1],
                ]
            )
            .reshape(1, Q, 2, 1, 1)
            .to(self.device)
            .to(torch.int64)
        )

        # permute direction for streaming:
        self._permute_reflect = (
            torch.Tensor([0, 3, 4, 1, 2, 7, 8, 5, 6]).to(self.device).to(torch.int64)
        )

    def solve_boundary(
        self,
        dt: float,
        dx: float,
        rho: torch.Tensor,
        vel: torch.Tensor,
        flags: torch.Tensor,
        f: torch.Tensor,
        phi_obs: torch.Tensor,
        obs_vel: torch.Tensor,
        tau: float = None,
        is_convection: float = False,
    ) -> torch.Tensor:
        tau = self._tau if tau is None else tau
        return Boundary2dKernel.apply(
            dt,
            dx,
            tau,
            rho,
            vel,
            flags,
            f,
            phi_obs,
            obs_vel,
            is_convection,
            self.axisymmetric_type,
        )
