from abc import ABC, abstractmethod


class AbstractLBMMacroCompute(ABC):
    type_flags = ["x", "X", "y", "Y", "z", "Z"]

    @property
    @abstractmethod
    def rank(self) -> int:
        ...

    @abstractmethod
    def macro_compute(self, f, rho, vel, flags):
        ...
