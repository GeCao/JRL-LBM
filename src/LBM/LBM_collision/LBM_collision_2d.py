import math
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load
from typing import Any

from src.LBM.LBM_collision import AbstractLBMCollision
from src.LBM.utils import CellType, KBCType, create_2d_meshgrid_tensor, AxiSymmetricType


collision_2d = load(
    name="collision_2d",
    sources=[
        "../src/LBM/LBM_collision/cuda/collision_2d.cpp",
        "../src/LBM/LBM_collision/cuda/collision_2d.cu",
    ],
    verbose=True,
    with_cuda=True,
    extra_cuda_cflags=["--use_fast_math"],
)

class GetGrad2dKernel(torch.autograd.Function):
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
        dx: float,
        rho: torch.Tensor,
        flags: torch.Tensor,
        axisymmetric_type: int = 0
    ) -> Any:
        # Solve fluid-solid coupling only
        dim = 2
        B, _, H, W = rho.shape
        grad_rho = torch.zeros((B, dim, H, W), dtype=rho.dtype).to(rho.device)

        collision_2d.get_grad_2d_forward(
            rho, flags, grad_rho, dx, axisymmetric_type
        )

        return grad_rho
    
    

class GetDiv2dKernel(torch.autograd.Function):
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
        dx: float,
        vel: torch.Tensor,
        flags: torch.Tensor,
        axisymmetric_type: int = 0
    ) -> Any:
        # Solve fluid-solid coupling only
        dim = 2
        B, _, H, W = vel.shape
        div_vel = torch.zeros((B, 1, H, W), dtype=vel.dtype).to(vel.device)

        collision_2d.get_div_2d_forward(
            vel, flags, div_vel, dx, axisymmetric_type
        )

        return div_vel
    
    
class Collision2dKernel(torch.autograd.Function):
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
        dx: float,
        dt: float,
        tau: float,
        f: torch.Tensor,
        rho: torch.Tensor,
        vel: torch.Tensor,
        flags: torch.Tensor,
        force: torch.Tensor = None,
        mesh_grid: torch.Tensor = None,
        is_convection: bool = False,
        KBC_type: int = None,
        axisymmetric_type: int = 0,
    ) -> Any:
        # Solve fluid-solid coupling only
        f_new = f + 0.0

        collision_2d.collision_2d_forward(
            dx, dt, tau, f, rho, vel, flags, force, mesh_grid, f_new, is_convection, KBC_type, axisymmetric_type
        )

        return f_new


class LBMCollision2d(AbstractLBMCollision):
    rank = 2

    def __init__(
        self,
        Q: int = 9,
        tau: float = 1.0,
        density_liquid: float = 0.265,
        density_gas: float = 0.038,
        rho_liquid: float = 0.265,
        rho_gas: float = 0.038,
        gravity_strength: float = 0.0,
        kappa: float = 0.08,
        tau_f: float = 0.7,
        tau_g: float = 0.7,
        axisymmetric_type: int = 0,
        contact_angle: float = math.pi / 2.0,
        device: torch.device = torch.device("cpu"),
        dtype=torch.float32,
        *args,
        **kwargs,
    ):
        super(LBMCollision2d, self).__init__(*args, **kwargs)
        self._Q = Q
        self._tau = tau

        # parameters for multiphase case
        self._density_liquid = density_liquid
        self._density_gas = density_gas
        self._rho_liquid = rho_liquid
        self._rho_gas = rho_gas
        self._kappa = kappa
        self._tau_f = tau_f
        self._tau_g = tau_g
        self.axisymmetric_type = axisymmetric_type
        self._contact_angle = contact_angle

        self.device = device
        self.dtype = dtype

        dim = 2
        self._gravity = (
            torch.Tensor([0.0, -gravity_strength])
            .reshape(1, dim, *([1] * dim))
            .to(self.device)
            .to(self.dtype)
        )

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

    def preset_KBC(self, dx: float, dt: float, tau: float, tau_D: float = None):
        self._tau = tau
        if tau_D is not None:
            self._tau_D = tau_D

    def equation_of_states(self, dx: float, dt: float, rho: torch.Tensor):
        c = dx / dt
        cs2 = c * c / 3.0
        RT = cs2
        a = 12.0 * RT
        b = 4.0

        temp_rho = b * rho / 4.0
        pressure = (
            rho
            * RT
            * (4.0 * temp_rho - 2.0 * temp_rho * temp_rho)
            / torch.pow(1.0 - temp_rho, 3)
            + rho * RT
            - a * rho * rho
        )

        return pressure

    @staticmethod
    def get_feq_static_(
        dx: float,
        dt: float,
        rho: torch.Tensor,
        vel: torch.Tensor,
        e: torch.Tensor,
        weight: torch.Tensor,
        tau: float,
        force: torch.Tensor = None,
        is_convection: bool = False,
    ) -> torch.Tensor:
        if force is not None:
            eps = 1e-10
            # We assume m=1/2, see this from the Phd thesis of Prof. Huang, eq.3.33
            vel = vel + torch.where(
                rho <= eps, torch.zeros_like(vel), 0.5 * force / rho
            )

        c = dx / dt
        cs2 = c * c / 3.0
        eu = (vel.unsqueeze(1) * e * c).sum(dim=2)  # [B, Q, res]
        feq = rho * weight * (1.0 + eu / cs2)
        # if not is_convection:
        uv = (vel * vel).sum(dim=1, keepdim=True)  # [B, 1, res]
        feq += rho * weight * (0.5 * eu * eu / cs2 / cs2 - 0.5 * uv / cs2)

        return feq

    def get_feq_(
        self,
        dx: float,
        dt: float,
        rho: torch.Tensor,
        vel: torch.Tensor,
        tau: float = None,
        force: torch.Tensor = None,
        is_convection: bool = False,
    ) -> torch.Tensor:
        tau = self._tau if tau is None else tau
        return LBMCollision2d.get_feq_static_(
            dx=dx,
            dt=dt,
            rho=rho,
            vel=vel,
            e=self._e,
            weight=self._weight,
            tau=tau,
            force=force,
            is_convection=is_convection,
        )

    def get_geq_(
        self,
        dx: float,
        dt: float,
        rho: torch.Tensor,
        density: torch.Tensor,
        vel: torch.Tensor,
        pressure: torch.Tensor,
        force: torch.Tensor,
        feq: torch.Tensor = None,
    ) -> torch.Tensor:
        c = dx / dt
        cs2 = c * c / 3.0
        if feq is None:
            feq = self.get_feq_(dx=dx, dt=dt, rho=rho, vel=vel, force=force)

        geq = self._weight * (pressure - cs2 * density) + cs2 * density / rho * feq

        return geq
    
    def get_grad(self, rho: torch.Tensor, dx: float, flags: torch.Tensor) -> torch.Tensor:
        return GetGrad2dKernel.apply(dx, rho, flags, self.axisymmetric_type)
    
    def get_div(self, vel: torch.Tensor, dx: float, flags: torch.Tensor) -> torch.Tensor:
        return GetDiv2dKernel.apply(dx, vel, flags, self.axisymmetric_type)

    def get_laplacian(
        self, input_: torch.Tensor, dx: float, flags: torch.Tensor
    ) -> torch.Tensor:
        output_ = F.pad(
            (
                4.0
                * (
                    input_[..., 1:-1, 2:]
                    + input_[..., 1:-1, :-2]
                    + input_[..., 2:, 1:-1]
                    + input_[..., :-2, 1:-1]
                )
                + (
                    input_[..., 2:, 2:]
                    + input_[..., 2:, :-2]
                    + input_[..., :-2, 2:]
                    + input_[..., :-2, :-2]
                )
                - (20 * input_[..., 1:-1, 1:-1])
            )
            / 5.0
            / (dx * dx),
            pad=(1, 1, 1, 1),
            mode="constant",
            value=0,
        )

        return output_

    def get_G_source_from_axisymmetric(
        self,
        dx: float,
        dt: float,
        rho: torch.Tensor,
        vel: torch.Tensor,
        flags: torch.Tensor,
        f: torch.Tensor,
        feq: torch.Tensor,
        is_convection: bool = False,
        mesh_grid: torch.Tensor = None,
    ):
        # if is_convection:
        #     return 0
        
        c = dx / dt
        cs2 = c * c / 3.0
        tau = self._tau_D if is_convection else self._tau
        if mesh_grid is None:
            mesh_grid = (
                create_2d_meshgrid_tensor(
                    size=[*(rho.shape)], device=rho.device, dtype=rho.dtype
                )
                + 0.5 * dx
            )
        # If axisymmetric applied, further source item needed.
        if self.axisymmetric_type == int(AxiSymmetricType.LINE_X_EQ_0):
            # [y, x] stands as [z, r]
            r = mesh_grid[:, 0:1, ...]  # Boundary not count
            er = self._e[:, :, 0, ...]
            ur = vel[:, 0:1, ...]
            A1 = -rho * ur / r
            if is_convection:
                D = cs2 * (tau - 0.5)
                # grad_C = self.get_grad(rho, dx=dx, flags=flags)
                # G = A1 + D * grad_C[:, 0:1, ...] / r
                
                # G = A1 + D * rho / r / r - D * self._e[:, :, 0, ...] * rho / (dt * cs2 * tau * r)
                
                # axisymmetric reference: A lattice Boltzmann method for axisymmetric thermocapillary flows, Liu et al, 2017
                s = (1 - 0.5 / tau) * er / r
                return -ur / r * feq * (1 - 0.5 / tau - 0.5 * s)
            else:
                G = A1
                
                visc = cs2 * (tau - 0.5)
                mu = visc * rho
                
                pressure = rho * cs2
                grad_u = self.get_grad(vel[:, 0:1, ...], dx=dx, flags=flags)
                grad_v = self.get_grad(vel[:, 1:2, ...], dx=dx, flags=flags)
                
                # grad_u[:, 0, :, 0] = 0
                # grad_v[:, 0, :, 0] = 0

                A2 = self.get_grad(pressure, dx=dx, flags=flags)[:, 0:1, ...]
                # A2[..., 0] = 0
                A2 = A2 + self.get_div(rho * ur * vel, dx=dx, flags=flags)
                A2 = A2 * (dt / (2 * r))

                F2 = (mu / r) * torch.cat(
                    (grad_u[:, 0:1, ...] - ur / r, grad_v[:, 0:1, ...]), dim=1
                )
                F2 = F2 + vel * A1
                F2 = F2 - dt * (tau - 1) * cs2 * self.get_grad(
                    A1, dx=dx, flags=flags
                )
                G = (
                    A1
                    + A2
                    + (F2.unsqueeze(1) * self._e).sum(dim=2) / cs2  # * ((1 - 0.5 / tau) / cs2)
                )
                
                # H1 = (
                #     mu * (grad_u[:, 0:1, ...] * 2) / r
                #     - 2 * mu * vel[:, 0:1, ...] / (r * r)
                #     - rho * vel[:, 0:1, ...] * vel[:, 0:1, ...] / r
                # )
                # H2 = (
                #     mu * (grad_v[:, 0:1, ...] + grad_u[:, 1:2, ...]) / r
                #     - rho * vel[:, 1:2, ...] * vel[:, 0:1, ...] / r
                # )
                # G = A1 + (self._e[:, :, 0, ...] * H1 + self._e[:, :, 1, ...] * H2) / cs2
            G = G * self._weight
            return G
        elif self.axisymmetric_type == int(AxiSymmetricType.LINE_Y_EQ_0):
            # [y, x] stands as [r, z]
            r = mesh_grid[:, 1:2, ...] - 0.5 * dx  # Boundary not count
            r[..., 0, :] = 1
            ur = vel[:, 1:2, ...]
            A1 = -rho * ur / r
            if is_convection:
                G = A1
            else:
                visc = cs2 * (tau - 0.5)
                mu = visc * rho
                pressure = rho * cs2
                A2 = self.get_grad(pressure, dx=dx, flags=flags)[:, 1:2, ...]
                A2 = A2 + self.get_div(rho * ur * vel, dx=dx, flags=flags)
                A2 = A2 * (dt / (2 * r))

                grad_u = self.get_grad(vel[:, 0:1, ...], dx=dx, flags=flags)
                grad_v = self.get_grad(vel[:, 1:2, ...], dx=dx, flags=flags)
                # F2 = (mu / r) * torch.cat(
                #     (grad_u[:, 1:2, ...], grad_v[:, 1:2, ...] - ur / r), dim=1
                # )
                # F2 = F2 - rho * ur * vel / r
                # F2 = F2 - dt * (tau - 1) * cs2 * self.get_grad(
                #     input_=A1, dx=dx, flags=flags
                # )
                # G = (
                #     A1
                #     + A2
                #     + (F2.unsqueeze(1) * self._e).sum(dim=2) * ((1 - 0.5 / tau) / cs2)
                # )
                H1 = (
                    mu * (grad_u[:, 1:2, ...] + grad_v[:, 0:1, ...]) / r
                    - rho * vel[:, 0:1, ...] * vel[:, 1:2, ...] / r
                )
                H2 = (
                    mu * (grad_v[:, 1:2, ...] * 2) / r
                    - 2 * mu * vel[:, 1:2, ...] / (r * r)
                    - rho * vel[:, 1:2, ...] * vel[:, 1:2, ...] / r
                )
                G = A1 + (self._e[:, :, 0, ...] * H1 + self._e[:, :, 1, ...] * H2) / cs2
            G = G * self._weight
            return G
        else:
            return 0.0

    def collision(
        self,
        dx: float,
        dt: float,
        f: torch.Tensor,
        rho: torch.Tensor,
        vel: torch.Tensor,
        flags: torch.Tensor,
        force: torch.Tensor = None,
        mesh_grid: torch.Tensor = None,
        is_convection: bool = False,
        KBC_type: int = None,
    ) -> torch.Tensor:
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

        feq = self.get_feq_(
            dx=dx,
            dt=dt,
            rho=rho,
            vel=vel,
            force=force,
            tau=tau,
            is_convection=is_convection,
        )
        
        if is_convection:
            s = 0.0
            Sa = 0.0
            if self.axisymmetric_type == int(AxiSymmetricType.LINE_X_EQ_0):
                er = self._e[:, :, 0, ...]
                ur = vel[:, 0:1, ...]
                r = mesh_grid[:, 0:1, ...]
                s = dt * (1 - 0.5 / tau) * er / r
                Sa = -ur / r * feq
            elif self.axisymmetric_type == int(AxiSymmetricType.LINE_X_EQ_0):
                er = self._e[:, :, 1, ...]
                ur = vel[:, 1:2, ...]
                r = mesh_grid[:, 1:2, ...]
                s = dt * (1 - 0.5 / tau) * er / r
                Sa = -ur / r * feq
            
            w = 1.0 / tau + s
        else:
            w = 1.0 / tau
        
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
        
        collision_f = f + w * (feq - f) + dt * Gi
        
        # if is_convection:
        #     f = (f + 0.5 * dt * Sa + 0.5 * feq / (tau - 0.5)) / (1 + 0.5 / (tau - 0.5))
        
        # if force is not None:
        #     collision_f = collision_f + dt / cs2 * self._weight * (self._e * force.unsqueeze(1)).sum(dim=2)

        f_new = torch.where(flags == int(CellType.OBSTACLE), f, collision_f)

        return f_new
