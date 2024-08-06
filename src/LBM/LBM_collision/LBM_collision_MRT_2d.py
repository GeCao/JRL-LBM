import math
import numpy as np
import torch
import torch.nn.functional as F
from typing import List

from src.LBM.LBM_collision import LBMCollision2d
from src.LBM.utils import CellType, KBCType, create_2d_meshgrid_tensor, AxiSymmetricType


class LBMCollisionMRT2d(LBMCollision2d):
    rank = 2

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super(LBMCollisionMRT2d, self).__init__(*args, **kwargs)

        dim = 2
        self.Minv_S_M = (
            torch.zeros((1, *([1] * dim), self._Q, self._Q))
            .to(self.device)
            .to(self.dtype)
        )
        self.Minv = torch.zeros_like(self.Minv_S_M)

        self.Minv_S_M_conv = (
            torch.zeros((1, *([1] * dim), self._Q, self._Q))
            .to(self.device)
            .to(self.dtype)
        )
        self.Minv_conv = torch.zeros_like(self.Minv_S_M_conv)

    def preset_KBC(self, dx: float, dt: float, tau: float, tau_D: float = None):
        dim = 2
        c = dx / dt
        cs2 = c * c / 3.0

        M_mat = (
            torch.Tensor(
                [
                    [1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [-4, -1, -1, -1, -1, 2, 2, 2, 2],
                    [4, -2, -2, -2, -2, 1, 1, 1, 1],
                    [0, 1, 0, -1, 0, 1, -1, -1, 1],
                    [0, -2, 0, 2, 0, 1, -1, -1, 1],
                    [0, 0, 1, 0, -1, 1, 1, -1, -1],
                    [0, 0, -2, 0, 2, 1, 1, -1, -1],
                    [0, 1, -1, 1, -1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, -1, 1, -1],
                ]
            )
            .to(self.dtype)
            .to(self.device)
        ).reshape(1, *([1] * dim), self._Q, self._Q)

        s7 = 1.0 / tau
        s4 = 8.0 * (2 - s7) / (8 - s7)
        S_mat = (
            torch.Tensor([0.0, s7, s7, 0.0, s4, 0.0, s4, s7, s7])
            .to(self.dtype)
            .to(self.device)
        ).reshape(1, *([1] * dim), self._Q, 1)

        M_inv_mat = torch.linalg.inv(M_mat)
        self.Minv_S_M = M_inv_mat @ (S_mat * M_mat)
        self.Minv = M_inv_mat

        if tau_D is None:
            self.Minv_S_M_conv = None
        else:
            s3 = 1.0 / tau_D
            S_mat = (
                torch.Tensor([1.0, 1.1, 1.1, s3, s3, s3, s3, 1.2, 1.2])
                .to(self.dtype)
                .to(self.device)
            ).reshape(1, *([1] * dim), self._Q, 1)
            self.Minv_S_M_conv = M_inv_mat @ (S_mat * M_mat)
            self.Minv_conv = M_inv_mat
            self._tau_D = tau_D

    def get_Omega(
        self, f: torch.Tensor, feq: torch.Tensor, is_convection: bool = False
    ):
        """
        Args:
            f: f before streaming [B, Q, res]
            rho: density [B, 1, res]

        Returns:
            torch.Tensor: f after streaming [B, Q, res]
        """
        assert self._Q == 9

        mat = self.Minv_S_M_conv if is_convection else self.Minv_S_M

        Omega = mat @ (feq - f).permute(0, 2, 3, 1).unsqueeze(-1)
        Omega = Omega.squeeze(-1).permute(0, 3, 1, 2)

        return Omega

    def apply_Minv(self, psi: torch.Tensor, is_convection: bool = False):
        """
        Args:
            psi: any f like tensor [B, Q, res]

        Returns:
            torch.Tensor: any f like tensor [B, Q, res]
        """
        assert self._Q == 9

        mat = self.Minv_conv if is_convection else self.Minv

        Omega = mat @ psi.permute(0, 2, 3, 1).unsqueeze(-1)
        Omega = Omega.squeeze(-1).permute(0, 3, 1, 2)

        return Omega

    def collision(
        self,
        dx: float,
        dt: float,
        f: torch.Tensor,
        rho: torch.Tensor,
        vel: torch.Tensor,
        flags: torch.Tensor,
        force: torch.Tensor,
        g: torch.Tensor = None,
        pressure: torch.Tensor = None,
        dfai: torch.Tensor = None,
        dprho: torch.Tensor = None,
        mesh_grid: torch.Tensor = None,
        is_convection: bool = False,
        KBC_type: int = None,
    ) -> List[torch.Tensor]:
        """
        Args:
            f: f before streaming [B, Q, res]
            rho: density [B, 1, res]
            vel: velocity [B, dim, res]
            flags: flags [B, 1, res]
            force: force [B, dim, res]
            KBC_type: int = [None, 'A', 'B', 'C', 'D'], where None is LBGK case, 'A/B/C/D' is different KBC cases

        Returns:
            torch.Tensor: f after streaming [B, Q, res]
        """
        c = dx / dt
        cs2 = c * c / 3.0
        tau = self._tau_D if is_convection else self._tau
        eps = 1e-10
        
        assert not is_convection

        feq = self.get_feq_(
            dx=dx,
            dt=dt,
            rho=rho,
            vel=vel,
            force=force,
            tau=tau,
            is_convection=is_convection,
        )

        Omega = self.get_Omega(f=f, feq=feq, is_convection=is_convection)
        Gi = self.get_G_source_from_axisymmetric(
            dx=dx,
            dt=dt,
            rho=rho,
            vel=vel,
            flags=flags,
            f=f,
            feq=feq,
            is_convection=is_convection,
            mesh_grid=mesh_grid,
        )

        collision_f = f + Omega + dt * Gi

        f_new = torch.where(flags == int(CellType.OBSTACLE), f, collision_f)

        return f_new
