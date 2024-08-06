import math
import numpy as np
import torch
import torch.nn.functional as F
from typing import List

from src.LBM.LBM_collision import LBMCollision2d
from src.LBM.utils import CellType, KBCType, AxiSymmetricType


class LBMCollisionKBC2d(LBMCollision2d):
    rank = 2

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super(LBMCollisionKBC2d, self).__init__(*args, **kwargs)

        dim = 2
        self.Minv_S_M = (
            torch.zeros((1, *([1] * dim), self._Q, self._Q))
            .to(self.device)
            .to(self.dtype)
        )

    def preset_KBC(self, dx: float, dt: float, tau: float, tau_D: float = None):
        dim = 2
        c = dx / dt
        cs2 = c * c / 3.0

        self._tau = tau
        if tau_D is not None:
            self._tau_D = tau_D

        assert self._Q == 9
        self.C_mat = (
            torch.Tensor(
                [
                    [1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 1, 0, -1, 0, 1, -1, -1, 1],
                    [0, 0, 1, 0, -1, 1, 1, -1, -1],
                    [0, 1, 0, 1, 0, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 1, -1, 1, -1],
                    [0, 0, 1, 0, 1, 1, 1, 1, 1],
                    [0, 0, 0, 0, 0, 1, 1, -1, -1],
                    [0, 0, 0, 0, 0, 1, -1, -1, 1],
                    [0, 0, 0, 0, 0, 1, 1, 1, 1],
                ]
            )
            .reshape(1, *([1] * dim), self._Q, self._Q)
            .to(self.dtype)
            .to(self.device)
        )

        self._p = (
            torch.Tensor([0, 1, 0, 2, 1, 0, 2, 1, 2])
            .reshape(1, self._Q, 1, *([1] * dim))
            .to(torch.int32)
            .to(self.device)
        )
        self._q = (
            torch.Tensor([0, 0, 1, 0, 1, 2, 1, 2, 2])
            .reshape(1, self._Q, 1, *([1] * dim))
            .to(torch.int32)
            .to(self.device)
        )

    def get_Omega(
        self,
        rho: torch.Tensor,
        vel: torch.Tensor,
        f: torch.Tensor,
        feq: torch.Tensor,
        KBC_type: int = None,
        is_convection: bool = False,
    ):
        """
        Args:
            f: f before streaming [B, Q, res]
            rho: density [B, 1, res]

        Returns:
            torch.Tensor: f after streaming [B, Q, res]

        Note: self.C_mat: [1, *([1] * dim), Q, Q]
        """
        assert self._Q == 9
        tau = self._tau_D if is_convection else self._tau

        # rho * M_{pq} = C @ f
        # order: [00] [10][01][20][11] [02][21][12][22]
        if KBC_type == int(KBCType.KBC_A) or KBC_type == int(KBCType.KBC_B):
            dim = 2
            dx = 1.0
            dt = 1.0
            c = dx / dt

            e = self._e  # [B, Q, dim, *res]
            euev = torch.pow(
                (c * e[:, :, 0, ...] - vel[:, 0:1, ...]).unsqueeze(1), self._p
            ) * torch.pow(
                (c * e[:, :, 1, ...] - vel[:, 1:2, ...]).unsqueeze(1), self._q
            )  # [B, Q, Q, *res]
            euev = euev.permute(0, 3, 4, 1, 2)

            rhoM = (
                (euev @ f.permute(0, 2, 3, 1).unsqueeze(-1))
                .squeeze(-1)
                .permute(0, 3, 1, 2)
            )  # [B, Q, *res]

            rhoMeq = (
                (euev @ feq.permute(0, 2, 3, 1).unsqueeze(-1))
                .squeeze(-1)
                .permute(0, 3, 1, 2)
            )  # [B, Q, *res]
        else:
            rhoM = (
                (self.C_mat @ f.permute(0, 2, 3, 1).unsqueeze(-1))
                .squeeze(-1)
                .permute(0, 3, 1, 2)
            )  # [B, Q, *res]

            rhoMeq = (
                (self.C_mat @ feq.permute(0, 2, 3, 1).unsqueeze(-1))
                .squeeze(-1)
                .permute(0, 3, 1, 2)
            )  # [B, Q, *res]

        rhoT = rhoM[:, 3:4, ...] + rhoM[:, 5:6, ...]  # T = M20 + M02
        rhoN = rhoM[:, 3:4, ...] - rhoM[:, 5:6, ...]  # N = M20 - M02
        rhoPIxy = rhoM[:, 4:5, ...]  # PI_xy = M11
        # rhoQxxy = rhoM[:, 6:7, ...]  # Qxxy = M21
        # rhoQxyy = rhoM[:, 7:8, ...]  # Qxyy = M12
        # rhoA = rhoM[:, 8:9, ...]  # A = M22

        rhoTeq = rhoMeq[:, 3:4, ...] + rhoMeq[:, 5:6, ...]  # T = M20 + M02
        rhoNeq = rhoMeq[:, 3:4, ...] - rhoMeq[:, 5:6, ...]  # N = M20 - M02
        rhoPIxyeq = rhoMeq[:, 4:5, ...]  # PI_xy = M11
        # rhoQxxyeq = rhoMeq[:, 6:7, ...]  # Qxxy = M21
        # rhoQxyyeq = rhoMeq[:, 7:8, ...]  # Qxyy = M12
        # rhoAeq = rhoMeq[:, 8:9, ...]  # A = M22

        df = f - feq
        ds = torch.zeros_like(f)
        dh = torch.zeros_like(f)
        if KBC_type == int(KBCType.LBGK):
            # s: All;
            # h: None;
            ds = df
            # ds[:, 0:1, ...] = rho - rhoT + rhoA
            # ds[:, 1:2, ...] = 0.5 * (
            #     0.5 * (rhoT + rhoN) + rho * vel[:, 0:1, ...] - rhoQxyy - rhoA
            # )
            # ds[:, 3:4, ...] = 0.5 * (
            #     0.5 * (rhoT + rhoN) - rho * vel[:, 0:1, ...] + rhoQxyy - rhoA
            # )
            # ds[:, 2:3, ...] = 0.5 * (
            #     0.5 * (rhoT - rhoN) + rho * vel[:, 1:2, ...] - rhoQxxy - rhoA
            # )
            # ds[:, 4:5, ...] = 0.5 * (
            #     0.5 * (rhoT - rhoN) - rho * vel[:, 1:2, ...] + rhoQxxy - rhoA
            # )
            # ds[:, 5:6, ...] = 0.25 * (rhoA + rhoPIxy + rhoQxyy + rhoQxxy)
            # ds[:, 6:7, ...] = 0.25 * (rhoA - rhoPIxy - rhoQxyy + rhoQxxy)
            # ds[:, 7:8, ...] = 0.25 * (rhoA + rhoPIxy - rhoQxyy - rhoQxxy)
            # ds[:, 8:9, ...] = 0.25 * (rhoA - rhoPIxy + rhoQxyy - rhoQxxy)

            # ds[:, 0:1, ...] -= rho - rhoTeq + rhoAeq
            # ds[:, 1:2, ...] -= 0.5 * (
            #     0.5 * (rhoTeq + rhoNeq) + rho * vel[:, 0:1, ...] - rhoQxyyeq - rhoAeq
            # )
            # ds[:, 3:4, ...] -= 0.5 * (
            #     0.5 * (rhoTeq + rhoNeq) - rho * vel[:, 0:1, ...] + rhoQxyyeq - rhoAeq
            # )
            # ds[:, 2:3, ...] -= 0.5 * (
            #     0.5 * (rhoTeq - rhoNeq) + rho * vel[:, 1:2, ...] - rhoQxxyeq - rhoAeq
            # )
            # ds[:, 4:5, ...] -= 0.5 * (
            #     0.5 * (rhoTeq - rhoNeq) - rho * vel[:, 1:2, ...] + rhoQxxyeq - rhoAeq
            # )
            # ds[:, 5:6, ...] -= 0.25 * (rhoAeq + rhoPIxyeq + rhoQxyyeq + rhoQxxyeq)
            # ds[:, 6:7, ...] -= 0.25 * (rhoAeq - rhoPIxyeq - rhoQxyyeq + rhoQxxyeq)
            # ds[:, 7:8, ...] -= 0.25 * (rhoAeq + rhoPIxyeq - rhoQxyyeq - rhoQxxyeq)
            # ds[:, 8:9, ...] -= 0.25 * (rhoAeq - rhoPIxyeq + rhoQxyyeq - rhoQxxyeq)
        elif KBC_type == int(KBCType.KBC_A) or KBC_type == int(KBCType.KBC_C):
            # s: PIxy, N, T;
            # h: Qxxy, Qxyy, A;
            ds[:, 0:1, ...] = rho - rhoT
            ds[:, 1:2, ...] = 0.5 * (0.5 * (rhoT + rhoN) + rho * vel[:, 0:1, ...])
            ds[:, 3:4, ...] = 0.5 * (0.5 * (rhoT + rhoN) - rho * vel[:, 0:1, ...])
            ds[:, 2:3, ...] = 0.5 * (0.5 * (rhoT - rhoN) + rho * vel[:, 1:2, ...])
            ds[:, 4:5, ...] = 0.5 * (0.5 * (rhoT - rhoN) - rho * vel[:, 1:2, ...])
            ds[:, 5:6, ...] = 0.25 * (rhoPIxy)
            ds[:, 6:7, ...] = 0.25 * (-rhoPIxy)
            ds[:, 7:8, ...] = 0.25 * (rhoPIxy)
            ds[:, 8:9, ...] = 0.25 * (-rhoPIxy)

            ds[:, 0:1, ...] -= rho - rhoTeq
            ds[:, 1:2, ...] -= 0.5 * (0.5 * (rhoTeq + rhoNeq) + rho * vel[:, 0:1, ...])
            ds[:, 3:4, ...] -= 0.5 * (0.5 * (rhoTeq + rhoNeq) - rho * vel[:, 0:1, ...])
            ds[:, 2:3, ...] -= 0.5 * (0.5 * (rhoTeq - rhoNeq) + rho * vel[:, 1:2, ...])
            ds[:, 4:5, ...] -= 0.5 * (0.5 * (rhoTeq - rhoNeq) - rho * vel[:, 1:2, ...])
            ds[:, 5:6, ...] -= 0.25 * (rhoPIxyeq)
            ds[:, 6:7, ...] -= 0.25 * (-rhoPIxyeq)
            ds[:, 7:8, ...] -= 0.25 * (rhoPIxyeq)
            ds[:, 8:9, ...] -= 0.25 * (-rhoPIxyeq)

            dh = df - ds
        elif KBC_type == int(KBCType.KBC_B) or KBC_type == int(KBCType.KBC_D):
            # s: PIxy, N;
            # h: T, Qxxy, Qxyy, A;
            ds[:, 0:1, ...] = rho
            ds[:, 1:2, ...] = 0.5 * (0.5 * (rhoN) + rho * vel[:, 0:1, ...])
            ds[:, 3:4, ...] = 0.5 * (0.5 * (rhoN) - rho * vel[:, 0:1, ...])
            ds[:, 2:3, ...] = 0.5 * (0.5 * (-rhoN) + rho * vel[:, 1:2, ...])
            ds[:, 4:5, ...] = 0.5 * (0.5 * (-rhoN) - rho * vel[:, 1:2, ...])
            ds[:, 5:6, ...] = 0.25 * (rhoPIxy)
            ds[:, 6:7, ...] = 0.25 * (-rhoPIxy)
            ds[:, 7:8, ...] = 0.25 * (rhoPIxy)
            ds[:, 8:9, ...] = 0.25 * (-rhoPIxy)

            ds[:, 0:1, ...] -= rho
            ds[:, 1:2, ...] -= 0.5 * (0.5 * (rhoNeq) + rho * vel[:, 0:1, ...])
            ds[:, 3:4, ...] -= 0.5 * (0.5 * (rhoNeq) - rho * vel[:, 0:1, ...])
            ds[:, 2:3, ...] -= 0.5 * (0.5 * (-rhoNeq) + rho * vel[:, 1:2, ...])
            ds[:, 4:5, ...] -= 0.5 * (0.5 * (-rhoNeq) - rho * vel[:, 1:2, ...])
            ds[:, 5:6, ...] -= 0.25 * (rhoPIxyeq)
            ds[:, 6:7, ...] -= 0.25 * (-rhoPIxyeq)
            ds[:, 7:8, ...] -= 0.25 * (rhoPIxyeq)
            ds[:, 8:9, ...] -= 0.25 * (-rhoPIxyeq)

            dh = df - ds
        else:
            raise RuntimeError("Please specific a right KBC Type value")

        beta = 1.0 / (2.0 * tau)

        gamma = 2.0
        eps = 1e-10
        if KBC_type != int(KBCType.LBGK):
            dsdh = (ds * dh / (feq + eps)).sum(dim=1, keepdim=True)
            dhdh = (dh * dh / (feq + eps)).sum(dim=1, keepdim=True)
            gamma = 1.0 / beta - (2.0 - 1.0 / beta) * (dsdh / (dhdh + eps))

        Omega = -beta * (2.0 * ds + gamma * dh)
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

        # if force is not None:
        #     force = self._gravity * 
        
        # if is_convection:
        #     D = cs2 * (self._tau_D - 0.5)
        #     if self.axisymmetric_type == int(AxiSymmetricType.LINE_X_EQ_0):
        #         ur = vel[:, 0:1, ...]
        #         r = mesh_grid[:, 0:1, ...]
        #         vel[:, 0:1, ...] = ur + D / r
        #     elif self.axisymmetric_type == int(AxiSymmetricType.LINE_Y_EQ_0):
        #         ur = vel[:, 1:2, ...]
        #         r = mesh_grid[:, 1:2, ...]
        #         vel[:, 1:2, ...] = ur + D / r

        feq = self.get_feq_(dx=dx, dt=dt, rho=rho, vel=vel, force=force, tau=tau)
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

        Omega = self.get_Omega(rho=rho, vel=vel, f=f, feq=feq, KBC_type=KBC_type, is_convection=is_convection)

        collision_f = (
            f
            + Omega
            + dt * Gi
        )

        f_new = torch.where(flags == int(CellType.OBSTACLE), f, collision_f)

        return f_new
