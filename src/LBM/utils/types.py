from enum import Enum
import numpy as np


class CellType(Enum):
    NOTHING = 0
    FLUID = 1
    OBSTACLE = 2
    EMPTY = 4
    INFLOW = 8
    OUTFLOW = 16
    INFLOW_2 = 32

    def __int__(self):
        return self.value


class KBCType(Enum):
    LBGK = 0  # 0b00000000
    KBC_A = 0b10000101
    KBC_B = 0b10000110
    KBC_C = 0b10001001
    KBC_D = 0b10001010

    def __int__(self):
        return self.value

    @staticmethod
    def is_KBC(input: int) -> bool:
        if input is None:
            return False

        return (input & 0b10000000) > 0

    @staticmethod
    def is_KBC_AC(input: int) -> bool:
        if input is None:
            return False

        return (input & 0b10000001) > 0

    @staticmethod
    def is_KBC_BD(input: int) -> bool:
        if input is None:
            return False

        return (input & 0b10000010) > 0

    @staticmethod
    def is_KBC_AB(input: int) -> bool:
        if input is None:
            return False

        return (input & 0b10000100) > 0

    @staticmethod
    def is_KBC_CD(input: int) -> bool:
        if input is None:
            return False

        return (input & 0b10001000) > 0


class ObsType(Enum):
    BOX = 0
    SPHERE = 1

    def __int__(self):
        return self.value


class AxiSymmetricType(Enum):
    NOT = 0
    LINE_X_EQ_0 = 1
    LINE_Y_EQ_0 = 2
    LINE_Z_EQ_0 = 3

    def __int__(self):
        return self.value
