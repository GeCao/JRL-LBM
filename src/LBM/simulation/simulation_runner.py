from src.LBM.LBM_propagation import LBMPropagation2d

from src.LBM.LBM_macro_compute import LBMMacroCompute2d

from src.LBM.LBM_collision import (
    LBMCollision2d,
    LBMCollisionKBC2d,
    LBMCollisionMRT2d,
)

from src.LBM.LBM_solid_fluid_coupling import LBMBoundary2d

from src.LBM.simulation import SimulationParameters


class SimulationRunner(object):
    def __init__(
        self,
        parameters: SimulationParameters,
    ):
        self.parameters = parameters

    def create_propagation(self):
        if self.parameters.is_2d():
            return LBMPropagation2d(
                tau=self.parameters.tau,
                axisymmetric_type=self.parameters.axisymmetric_type,
                dtype=self.parameters.dtype,
                device=self.parameters.device,
            )
        else:
            raise RuntimeError("Not Implemented for 3D propagation")

    def create_macro_compute(self):
        if self.parameters.is_2d():
            return LBMMacroCompute2d(
                Q=self.parameters.Q,
                tau=self.parameters.tau,
                density_liquid=self.parameters.density_fluid,
                density_gas=self.parameters.density_gas,
                rho_liquid=self.parameters.rho_fluid,
                rho_gas=self.parameters.rho_gas,
                axisymmetric_type=self.parameters.axisymmetric_type,
                contact_angle=self.parameters.contact_angle,
                dtype=self.parameters.dtype,
                device=self.parameters.device,
            )
        else:
            raise RuntimeError("Not Implemented for 3D macro computation")

    def create_collision(self):
        if self.parameters.is_2d():
            return LBMCollision2d(
                Q=self.parameters.Q,
                tau=self.parameters.tau,
                density_liquid=self.parameters.density_fluid,
                density_gas=self.parameters.density_gas,
                rho_liquid=self.parameters.rho_fluid,
                rho_gas=self.parameters.rho_gas,
                gravity_strength=self.parameters.gravity_strength,
                kappa=self.parameters.kappa,
                tau_f=self.parameters.tau_f,
                tau_g=self.parameters.tau_g,
                axisymmetric_type=self.parameters.axisymmetric_type,
                contact_angle=self.parameters.contact_angle,
                dtype=self.parameters.dtype,
                device=self.parameters.device,
            )
        else:
            raise RuntimeError("Not Implemented for 3D collision")

    def create_collision_KBC(self):
        if self.parameters.is_2d():
            return LBMCollisionKBC2d(
                Q=self.parameters.Q,
                tau=self.parameters.tau,
                density_liquid=self.parameters.density_fluid,
                density_gas=self.parameters.density_gas,
                rho_liquid=self.parameters.rho_fluid,
                rho_gas=self.parameters.rho_gas,
                gravity_strength=self.parameters.gravity_strength,
                kappa=self.parameters.kappa,
                tau_f=self.parameters.tau_f,
                tau_g=self.parameters.tau_g,
                axisymmetric_type=self.parameters.axisymmetric_type,
                contact_angle=self.parameters.contact_angle,
                dtype=self.parameters.dtype,
                device=self.parameters.device,
            )
        else:
            raise RuntimeError("Not Implemented for 3D KBC collision")

    def create_collision_MRT(self):
        if self.parameters.is_2d():
            return LBMCollisionMRT2d(
                Q=self.parameters.Q,
                tau=self.parameters.tau,
                density_liquid=self.parameters.density_fluid,
                density_gas=self.parameters.density_gas,
                rho_liquid=self.parameters.rho_fluid,
                rho_gas=self.parameters.rho_gas,
                gravity_strength=self.parameters.gravity_strength,
                kappa=self.parameters.kappa,
                tau_f=self.parameters.tau_f,
                tau_g=self.parameters.tau_g,
                axisymmetric_type=self.parameters.axisymmetric_type,
                contact_angle=self.parameters.contact_angle,
                dtype=self.parameters.dtype,
                device=self.parameters.device,
            )
        else:
            raise RuntimeError("Not Implemented for 3D MRT collision")

    def create_LBM_fluid_solid_coupling(self):
        if self.parameters.is_2d():
            return LBMBoundary2d(
                tau=self.parameters.tau,
                axisymmetric_type=self.parameters.axisymmetric_type,
                dtype=self.parameters.dtype,
                device=self.parameters.device,
            )
        else:
            raise NotImplementedError("3D Immersed boundary not implemented")

    def step(self):
        self.parameters.step()
