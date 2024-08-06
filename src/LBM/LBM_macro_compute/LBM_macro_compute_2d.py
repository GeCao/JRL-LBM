import math
import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load
from typing import List, Any

from src.LBM.LBM_macro_compute import AbstractLBMMacroCompute
from src.LBM.LBM_collision import LBMCollision2d
from src.LBM.utils import CellType, create_2d_meshgrid_tensor, AxiSymmetricType


macro_compute = load(
    name="macro_compute",
    sources=[
        "../src/LBM/LBM_macro_compute/cuda/macro_compute.cpp",
        "../src/LBM/LBM_macro_compute/cuda/macro_compute.cu",
    ],
    verbose=True,
    with_cuda=True,
    extra_cuda_cflags=["--use_fast_math"],
)


class CMacroCompute2dKernel(torch.autograd.Function):
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
        h: torch.Tensor,
        C: torch.Tensor,
        flags: torch.Tensor,
        vel: torch.Tensor,
        mesh_grid: torch.Tensor,
        axisymmetric_type: int = 0,
    ) -> Any:
        C_new = C + 0.0
        macro_compute.macro_compute_C_2d_forward(dx, dt, h, C, flags, vel, mesh_grid, C_new, int(axisymmetric_type))

        return C_new
    
    
class FluidMacroCompute2dKernel(torch.autograd.Function):
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
        f: torch.Tensor,
        rho: torch.Tensor,
        vel: torch.Tensor,
        flags: torch.Tensor,
        g: torch.Tensor = None,
        pressure: torch.Tensor = None,
        density: torch.Tensor = None,
        axisymmetric_type: int = 0,
    ) -> Any:
        rho_new = rho + 0.0
        vel_new = vel + 0.0
        if g is None:
            g = torch.Tensor([]).to(f.device).to(f.dtype)
        
        if pressure is None:
            pressure = torch.Tensor([]).to(f.device).to(f.dtype)
        
        if density is None:
            density = torch.Tensor([]).to(f.device).to(f.dtype)
            density_new = torch.Tensor([]).to(f.device).to(f.dtype)
        else:
            density_new = density + 0.0
        
        macro_compute.macro_compute_2d_forward(
            dx, dt, f, rho, vel, flags, rho_new, vel_new, g, pressure, density, density_new, int(axisymmetric_type)
        )

        results = [rho_new, vel_new]
        if density_new.numel() > 0:
            results.append(density_new)
        
        return results


class LBMMacroCompute2d(AbstractLBMMacroCompute):
    rank = 2

    def __init__(
        self,
        Q: int = 9,
        tau: float = 1.0,
        density_liquid: float = 0.265,
        density_gas: float = 0.038,
        rho_liquid: float = 0.265,
        rho_gas: float = 0.038,
        axisymmetric_type: int = 0,
        contact_angle: float = 0.5 * math.pi,
        device: torch.device = torch.device("cpu"),
        dtype=torch.float32,
        *args,
        **kwargs,
    ):
        super(LBMMacroCompute2d, self).__init__(*args, **kwargs)
        self._Q = Q
        self._tau = tau

        # parameters for multiphase case
        self._density_liquid = density_liquid
        self._density_gas = density_gas
        self._rho_liquid = rho_liquid
        self._rho_gas = rho_gas
        self.axisymmetric_type = axisymmetric_type
        self._contact_angle = contact_angle

        self.device = device
        self.dtype = dtype

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

    def get_pressure(self, dx: float, dt: float, density: torch.Tensor) -> torch.Tensor:
        c = dx / dt
        cs2 = c * c / 3.0
        RT = cs2
        a = 12.0 * RT
        b = 4.0

        temp_density = b * density / 4.0
        pressure = (
            density
            * RT
            * temp_density
            * (4.0 - 2.0 * temp_density)
            / torch.pow((1 - temp_density), 3)
            - a * density * density
            + density * RT
        )

        return pressure

    @staticmethod
    def _average_with_kernel(
        base_image: torch.Tensor,
        flags: torch.Tensor,
        value: torch.Tensor,
        kernel: torch.Tensor,
    ) -> torch.Tensor:
        """Static method to average occurrence of a value with a given kernel

        Args:
            flags (torch.Tensor): flags (Flag2dGrid):  Flags defining the type of each cell
            value (torch.Tensor): For which type of cell should the count be done
            kernel (torch.Tensor): kernel describing the neighbourhood

        Returns:
            torch.Tensor: average for each cell.
        """
        matching_flags = (flags == value).to(base_image.dtype)
        pad = (1, 1, 1, 1)

        # first pad the tensor with zeros
        matching_flags_pad = F.pad(
            input=torch.ones_like(base_image) * matching_flags,
            pad=pad,
            mode="constant",
            value=0,
        )
        count = F.conv2d(matching_flags_pad, kernel)

        base_image_pad = F.pad(
            input=base_image * matching_flags, pad=pad, mode="constant", value=0
        )
        sum = F.conv2d(base_image_pad, kernel)

        eps = 1e-10
        result = torch.where(count.abs() <= eps, sum, sum / count)

        # reset the count on all other spots beside the ones where it was matched
        return result

    def contact_angle_correctness(self, rho: torch.Tensor, flags: torch.Tensor):
        # Providing we are simulating multiphase case
        rho_obs = torch.zeros_like(rho)

        # ===========================
        #      Contact Angle
        # ===========================
        # 1. neg x
        hlp_CA = torch.abs(rho[..., 2:, 1] - rho[..., :-2, 1])
        rho_obs[..., 1:-1, 0] = (
            rho[..., 1:-1, 2] + torch.tan(math.pi / 2.0 - self._contact_angle) * hlp_CA
        )
        # 2. pos x
        hlp_CA = torch.abs(rho[..., 2:, -2] - rho[..., :-2, -2])
        rho_obs[..., 1:-1, -1] = (
            rho[..., 1:-1, -3] + torch.tan(math.pi / 2.0 - self._contact_angle) * hlp_CA
        )
        # 3. neg y
        hlp_CA = torch.abs(rho[..., 1, 2:] - rho[..., 1, :-2])
        rho_obs[..., 0, 1:-1] = (
            rho[..., 2, 1:-1] + torch.tan(math.pi / 2.0 - self._contact_angle) * hlp_CA
        )
        # 4.pos y
        hlp_CA = torch.abs(rho[..., -2, 2:] - rho[..., -2, :-2])
        rho_obs[..., -1, 1:-1] = (
            rho[..., -3, 1:-1] + torch.tan(math.pi / 2.0 - self._contact_angle) * hlp_CA
        )

        # 5. edge points
        rho_obs[..., 0, 0] = 0.5 * (rho_obs[..., 1, 0] + rho_obs[..., 0, 1])
        rho_obs[..., -1, 0] = 0.5 * (rho_obs[..., -2, 0] + rho_obs[..., -1, 1])
        rho_obs[..., 0, -1] = 0.5 * (rho_obs[..., 0, -2] + rho_obs[..., 1, -1])
        rho_obs[..., -1, -1] = 0.5 * (rho_obs[..., -2, -2] + rho_obs[..., -2, -2])

        return torch.where(flags == int(CellType.OBSTACLE), rho_obs, rho)

    def macro_compute_C(
        self,
        dx: float,
        dt: float,
        h: torch.Tensor,
        C: torch.Tensor,
        flags: torch.Tensor,
        vel: torch.Tensor,
        mesh_grid: torch.Tensor,
    ) -> torch.Tensor:
        macro_C = h.sum(dim=1, keepdim=True)  # [B, 1, res]
        # if self.axisymmetric_type == int(AxiSymmetricType.LINE_X_EQ_0):
        #     ur = vel[:, 0:1, ...]
        #     r = mesh_grid[:, 0:1, ...]
        #     macro_C = macro_C / (1.0 + 0.5 * dt * ur / r)
        # elif self.axisymmetric_type == int(AxiSymmetricType.LINE_Y_EQ_0):
        #     ur = vel[:, 1:2, ...]
        #     r = mesh_grid[:, 1:2, ...]
        #     macro_C = macro_C / (1.0 + 0.5 * dt * ur / r)

        C_new = torch.where(flags == int(CellType.OBSTACLE), C, macro_C)

        return C_new

    def ApplyCompute_C(
        self, dx: float, dt: float, h: torch.Tensor, C: torch.Tensor, flags: torch.Tensor, vel: torch.Tensor, mesh_grid: torch.Tensor
    ):
        return CMacroCompute2dKernel.apply(dx, dt, h, C, flags, vel, mesh_grid, self.axisymmetric_type)
    
    def ApplyCompute_fluid(
        self,
        dx: float,
        dt: float,
        f: torch.Tensor,
        rho: torch.Tensor,
        vel: torch.Tensor,
        flags: torch.Tensor,
        g: torch.Tensor = None,
        pressure: torch.Tensor = None,
        density: torch.Tensor = None,
    ):
        return FluidMacroCompute2dKernel.apply(dx, dt, f, rho, vel, flags, g, pressure, density, self.axisymmetric_type)

    def macro_compute(
        self,
        dx: float,
        dt: float,
        f: torch.Tensor,
        rho: torch.Tensor,
        vel: torch.Tensor,
        flags: torch.Tensor,
        g: torch.Tensor = None,
        pressure: torch.Tensor = None,
        density: torch.Tensor = None,
    ) -> List[torch.Tensor]:
        c = dx / dt

        macro_rho = f.sum(dim=1).unsqueeze(1)  # [B, 1, res]
        rho_new = torch.where(flags == int(CellType.OBSTACLE), rho, macro_rho)
        if self._contact_angle is not None:
            rho_new = self.contact_angle_correctness(rho=rho_new, flags=flags)

        vel_new = (f.unsqueeze(2) * self._e).sum(dim=1) * (c / rho_new)
        vel_new = torch.where(flags == int(CellType.OBSTACLE), vel, vel_new)

        if density is not None:
            density_liquid = self._density_liquid
            density_gas = self._density_gas
            rho_liquid = self._rho_liquid
            rho_gas = self._rho_gas
            density = density_gas + (density_liquid - density_gas) * (
                (rho_new - rho_gas) / (rho_liquid - rho_gas)
            )
            if pressure is not None:
                pressure = self.get_pressure(dx=dx, dt=dt, density=density)

            return [rho_new, vel_new, density]

        return [rho_new, vel_new]

    def get_vort(self, vel: torch.Tensor, dx: float) -> torch.Tensor:
        vort = (
            (vel[..., 0:1, 2:, 1:-1] - vel[..., 0:1, :-2, 1:-1])
            - (vel[..., 1:2, 1:-1, 2:] - vel[..., 1:2, 1:-1, :-2])
        ) / (2.0 * dx)

        vort_pad = F.pad(vort, pad=(1, 1, 1, 1), mode="replicate")

        return vort_pad
