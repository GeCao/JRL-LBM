import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load
from typing import List, Any

from src.LBM.LBM_propagation import AbstractLBMPropagation
from src.LBM.LBM_collision import LBMCollision2d
from src.LBM.utils import CellType, ObsType, create_2d_meshgrid_tensor


propagation = load(
    name="propagation",
    sources=[
        "../src/LBM/LBM_propagation/cuda/propagation.cpp",
        "../src/LBM/LBM_propagation/cuda/propagation.cu",
    ],
    verbose=True,
    with_cuda=True,
    extra_cuda_cflags=["--use_fast_math"],
)


class Propagation2dKernel(torch.autograd.Function):
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
        flags: torch.Tensor,
        f: torch.Tensor,
        phi_obs: torch.Tensor = None,
        rho: torch.Tensor = None,
        vel: torch.Tensor = None,
        inflow_vel: torch.Tensor = None,
        inflow_density: float = None,
        is_convection: bool = False,
        axisymmetric_type: int = 0,
    ) -> Any:
        # Solve fluid-solid coupling only
        f_new = f + 0.0

        inflow_info = False
        if inflow_density is not None or inflow_vel is not None:
            assert rho is not None
            inflow_info = True
        else:
            inflow_info = False

        if (flags == int(CellType.INFLOW_2)).any():
            inflow_info = True

        if inflow_density is None:
            inflow_density = -1

        if inflow_vel is None:
            inflow_vel = torch.Tensor([])
            
        if phi_obs is None:
            phi_obs = torch.Tensor([])

        if rho is None:
            rho = torch.Tensor([])

        if vel is None:
            vel = torch.Tensor([])

        propagation.propagation_2d_forward(
            flags,
            f,
            phi_obs,
            f_new,
            rho,
            vel,
            inflow_vel,
            inflow_density,
            is_convection,
            axisymmetric_type,
        )

        if inflow_info:
            return (rho, vel, f_new)

        return f_new


class LBMPropagation2d(AbstractLBMPropagation):
    rank = 2

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super(LBMPropagation2d, self).__init__(*args, **kwargs)

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

    def propagation(
        self,
        f: torch.Tensor,
        flags: torch.Tensor,
        phi_obs: torch.Tensor = None,
        rho: torch.Tensor = None,
        vel: torch.Tensor = None,
        inflow_vel: torch.Tensor = None,
        inflow_density: float = None,
        is_convection: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            f (torch.Tensor): f before streaming [B, Q, res]

        Returns:
            torch.Tensor: f after streaming [B, Q, res]
        """
        return Propagation2dKernel.apply(
            flags,
            f,
            phi_obs,
            rho,
            vel,
            inflow_vel,
            inflow_density,
            is_convection,
            self.axisymmetric_type,
        )

    def rebounce_obstacle(self, f: torch.Tensor, flags: torch.Tensor) -> torch.Tensor:
        inverted_f = f[:, self._permute_reflect, ...]

        f_new = torch.where(flags == int(CellType.OBSTACLE), inverted_f, f)

        return f_new
