import torch

from abc import ABC, abstractmethod


class AbstractLBMBoundary(ABC):
    type_flags = ["x", "X", "y", "Y", "z", "Z"]

    def __init__(
        self,
        tau: float,
        axisymmetric_type: int = 0,
        device: torch.device = torch.device("cpu"),
        dtype=torch.float32,
    ):
        self._tau = tau
        self.axisymmetric_type = axisymmetric_type
        self.device = device
        self.dtype = dtype

    @property
    @abstractmethod
    def rank(self) -> int:
        ...
