import torch
import math
from typing import List


class SimulationParameters(object):
    def __init__(
        self,
        dtype=torch.float32,
        dim: int = 2,
        dt: float = 1.0,
        device: torch.device = torch.device("cpu"),
        simulation_size: List[int] = [1, 1, 256, 256],
        density_gas: float = 0.0,
        density_fluid: float = 1.0,
        gravity_strength: float = 0.0,
        axisymmetric_type: int = 0,
        contact_angle: float = 0.75 * math.pi,
        Q: float = 9,
        tau: float = 1.0,
        rho_gas: float = 0.038,
        rho_fluid: float = 0.265,
        kappa: float = 0.08,
        tau_g: float = 0.7,
        tau_f: float = 0.7,
        k: float = 1.0,
    ):
        self.dim = dim
        self.dtype = dtype
        self.dt = dt
        self.device = device

        self.frame = 0
        self.time_per_frame = 0
        self.frame_length = 1.0
        self.time_total = 0

        if dim == 2 and len(simulation_size) != 4:
            raise ValueError(
                "For 2d simulation simulation size should have 4 parameters B x C x H x W"
            )

        self.simulation_size = simulation_size

        self.density_gas = density_gas
        self.density_fluid = density_fluid
        self.gravity_strength = gravity_strength
        self.axisymmetric_type = axisymmetric_type
        self.contact_angle = contact_angle

        self.Q = Q
        self.tau = tau
        self.rho_gas = rho_gas
        self.rho_fluid = rho_fluid
        self.kappa = kappa
        self.tau_g = tau_g
        self.tau_f = tau_f

        self.k = k  # susceptibilty

    def step(self):
        """Advances the simulation one time step"""
        self.time_per_frame += self.dt
        self.time_total += self.dt

        if self.time_per_frame >= self.frame_length:
            self.frame += 1

            # re-calculate total time to prevent drift
            self.time_total = self.frame * self.frame_length
            self.time_per_frame = 0

    def get_dx(self):
        return 1.0 / max(self.simulation_size)

    def is_2d(self):
        return self.dim == 2

    def is_3d(self):
        return self.dim == 3

    def set_device(self, device: str = "cuda"):
        if device not in ["cuda", "cpu"]:
            raise ValueError(
                "Set_device: device {} must be either cuda or cpu.".format(device)
            )
        self.device = torch.device(device)
