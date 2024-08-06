from abc import ABC, abstractmethod


class AbstractLBMCollision(ABC):
    type_flags = ["x", "X", "y", "Y", "z", "Z"]

    @property
    @abstractmethod
    def rank(self) -> int:
        ...

    @abstractmethod
    def collision(self, f, rho, vel, flags, force):
        ...
