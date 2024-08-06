import numpy as np
from typing import List
import torch
import torch.nn.functional as F


def get_staggered_x(input: torch.Tensor, mode: str = "replicate") -> torch.Tensor:
    if len(input.shape) == 4:
        grid_staggered_x = (input[..., 1:] + input[..., :-1]) * 0.5
        grid_staggered_x_pad = F.pad(
            grid_staggered_x, pad=(1, 1, 0, 0), mode=mode, value=0
        )
        return grid_staggered_x_pad
    elif len(input.shape) == 5:
        grid_staggered_x = (input[..., 1:] + input[..., :-1]) * 0.5
        grid_staggered_x_pad = F.pad(
            grid_staggered_x, pad=(1, 1, 0, 0, 0, 0), mode=mode, value=0
        )
        return grid_staggered_x_pad
    else:
        raise RuntimeError("A grid has to be 2D(3D) [B, C, (D), H, W] to be staggered")


def get_staggered_y(input: torch.Tensor, mode: str = "replicate") -> torch.Tensor:
    if len(input.shape) == 4:
        grid_staggered_y = (input[..., 1:, :] + input[..., :-1, :]) * 0.5
        grid_staggered_y_pad = F.pad(
            grid_staggered_y, pad=(0, 0, 1, 1), mode=mode, value=0
        )
        return grid_staggered_y_pad
    elif len(input.shape) == 5:
        grid_staggered_y = (input[..., 1:, :] + input[..., :-1, :]) * 0.5
        grid_staggered_y_pad = F.pad(
            grid_staggered_y, pad=(0, 0, 1, 1, 0, 0), mode=mode, value=0
        )
        return grid_staggered_y_pad
    else:
        raise RuntimeError("A grid has to be 2D(3D) [B, C, (D), H, W] to be staggered")


def get_staggered_z(input: torch.Tensor, mode: str = "replicate") -> torch.Tensor:
    if len(input.shape) == 5:
        grid_staggered_z = (input[..., 1:, :, :] + input[..., :-1, :, :]) * 0.5
        grid_staggered_z_pad = F.pad(
            grid_staggered_z, pad=(0, 0, 0, 0, 1, 1), mode=mode, value=0
        )
        return grid_staggered_z_pad
    else:
        raise RuntimeError("A grid has to be 3D [B, C, D, H, W] to be staggered")


def get_staggered(input: torch.Tensor, mode: str = "replicate") -> List[torch.Tensor]:
    dim = input.shape[1]
    if dim < 2 or dim > 3:
        raise RuntimeError("Only 2D or 3D scene supported")

    output = [
        get_staggered_x(input=input[:, 0:1, ...], mode=mode),
        get_staggered_y(input=input[:, 1:2, ...], mode=mode),
    ]
    if dim == 3:
        output.append(get_staggered_z(input=input[:, 2:3, ...], mode=mode))

    return output


def create_2d_meshgrid_tensor(
    size: List[int],
    device: torch.device = torch.device("cpu"),
    dtype=torch.float32,
) -> torch.Tensor:
    [batch, _, height, width] = size
    y_pos, x_pos = torch.meshgrid(
        [
            torch.arange(0, height, device=device, dtype=dtype),
            torch.arange(0, width, device=device, dtype=dtype),
        ]
    )
    mgrid = torch.stack([x_pos, y_pos], dim=0)  # [C, H, W]
    mgrid = mgrid.unsqueeze(0)  # [B, C, H, W]
    mgrid = mgrid.repeat(batch, 1, 1, 1)
    return mgrid


def create_droplet_2d(
    droplet_center: torch.Tensor,
    droplet_radius: float,
    rho_liquid: float,
    rho: torch.Tensor,
) -> torch.Tensor:
    dim = 2
    device = rho.device
    dtype = rho.dtype
    simulation_size = rho.shape

    dist = (
        create_2d_meshgrid_tensor(simulation_size, device=device, dtype=dtype)
        - droplet_center.reshape(1, dim, *([1] * dim)).to(device).to(dtype)
    ).norm(dim=1, keepdim=True)
    is_droplet = dist < droplet_radius
    rho[is_droplet] = rho_liquid

    return rho


def dot(
    dim: int, x: torch.Tensor, y: torch.Tensor, keep_dim: bool = False
) -> torch.Tensor:
    x_shape = x.shape
    y_shape = y.shape

    len_x_shape = len(x_shape)
    len_y_shape = len(y_shape)

    assert -1 <= (len_x_shape - len_y_shape) <= 1
    assert 2 <= dim <= 3

    if len_x_shape == len_y_shape:
        # We assume tensor x and y share same shape
        if dim == 2:
            assert x_shape[-3] == dim
            assert y_shape[-3] == dim

            result = (
                x[..., 0, :, :] * y[..., 0, :, :] + x[..., 1, :, :] * y[..., 1, :, :]
            )
            if keep_dim:
                result = result.unsqueeze(-3)

            return result
        elif dim == 3:
            assert x_shape[-4] == dim
            assert y_shape[-4] == dim

            result = x[..., 0, :, :, :] * y[..., 0, :, :, :]
            result += x[..., 1, :, :, :] * y[..., 1, :, :, :]
            result += x[..., 2, :, :, :] * y[..., 2, :, :, :]
            if keep_dim:
                result = result.unsqueeze(-4)

            return result
    elif len_x_shape == len_y_shape + 1:
        # We assume x is e and y is Q/dim defined vector
        Q = x_shape[1]
        if dim == 2:
            assert len_x_shape == 5  # [B, Q, dim, *res]
            assert len_y_shape == 4  # [B, Q/dim, *res]
            assert x_shape[-3] == dim

            if y_shape[-3] == dim:
                result = (
                    x[..., 0, :, :] * y[..., 0:1, :, :]
                    + x[..., 1, :, :] * y[..., 1:2, :, :]
                )
                if not keep_dim:
                    result = result.squeeze(-3)

                return result
            elif y_shape[-3] == Q:
                result = x[:, 0, :, :, :] * y[..., 0:1, :, :]
                result += x[:, 1, :, :, :] * y[..., 1:2, :, :]
                result += x[:, 2, :, :, :] * y[..., 2:3, :, :]
                result += x[:, 3, :, :, :] * y[..., 3:4, :, :]
                result += x[:, 4, :, :, :] * y[..., 4:5, :, :]
                result += x[:, 5, :, :, :] * y[..., 5:6, :, :]
                result += x[:, 6, :, :, :] * y[..., 6:7, :, :]
                result += x[:, 7, :, :, :] * y[..., 7:8, :, :]
                result += x[:, 8, :, :, :] * y[..., 8:9, :, :]
                if not keep_dim:
                    result = result.squeeze(-4)

                return result
        elif dim == 3:
            assert len_x_shape == 6  # [B, Q, dim, *res]
            assert len_y_shape == 5  # [B, Q/dim, *res]
            assert x_shape[-4] == dim

            if y_shape[-4] == dim:
                result = x[..., 0, :, :, :] * y[..., 0:1, :, :, :]
                result += x[..., 1, :, :, :] * y[..., 1:2, :, :, :]
                result += x[..., 2, :, :, :] * y[..., 2:3, :, :, :]
                if not keep_dim:
                    result = result.squeeze(-4)

                return result
            elif y_shape[-4] == Q:
                result = x[:, 0, :, :, :, :] * y[..., 0:1, :, :, :]
                result += x[:, 1, :, :, :, :] * y[..., 1:2, :, :, :]
                result += x[:, 2, :, :, :, :] * y[..., 2:3, :, :, :]
                result += x[:, 3, :, :, :, :] * y[..., 3:4, :, :, :]
                result += x[:, 4, :, :, :, :] * y[..., 4:5, :, :, :]
                result += x[:, 5, :, :, :, :] * y[..., 5:6, :, :, :]
                result += x[:, 6, :, :, :, :] * y[..., 6:7, :, :, :]
                result += x[:, 7, :, :, :, :] * y[..., 7:8, :, :, :]
                result += x[:, 8, :, :, :, :] * y[..., 8:9, :, :, :]
                result += x[:, 9, :, :, :, :] * y[..., 9:10, :, :, :]
                result += x[:, 10, :, :, :, :] * y[..., 10:11, :, :, :]
                result += x[:, 11, :, :, :, :] * y[..., 11:12, :, :, :]
                result += x[:, 12, :, :, :, :] * y[..., 12:13, :, :, :]
                result += x[:, 13, :, :, :, :] * y[..., 13:14, :, :, :]
                result += x[:, 14, :, :, :, :] * y[..., 14:15, :, :, :]
                result += x[:, 15, :, :, :, :] * y[..., 15:16, :, :, :]
                result += x[:, 16, :, :, :, :] * y[..., 16:17, :, :, :]
                result += x[:, 17, :, :, :, :] * y[..., 17:18, :, :, :]
                result += x[:, 18, :, :, :, :] * y[..., 18:19, :, :, :]
                if not keep_dim:
                    result = result.squeeze(-4)

                return result
    elif len_x_shape + 1 == len_y_shape:
        return dot(dim=dim, x=y, y=x, keep_dim=keep_dim)


def dot_e_vel(e: torch.Tensor, vel: torch.Tensor) -> torch.Tensor:
    dim = vel.shape[1]
    if dim == 2:
        result = e[:, :, 0, ...] * vel[:, 0:1, ...] + e[:, :, 1, ...] * vel[:, 1:2, ...]
        return result
    elif dim == 3:
        result = (
            e[:, :, 0, ...] * vel[:, 0:1, ...]
            + e[:, :, 1, ...] * vel[:, 1:2, ...]
            + e[:, :, 2, ...] * vel[:, 2:3, ...]
        )
        return result


def dot_vel_vel(
    vel1: torch.Tensor, vel2: torch.Tensor, keep_dim: bool = False
) -> torch.Tensor:
    dim = vel1.shape[1]
    assert vel1.shape[1] == vel2.shape[1]

    if dim == 2:
        result = vel1[:, 0, ...] * vel2[:, 0, ...] + vel1[:, 1, ...] * vel2[:, 1, ...]
        if keep_dim:
            result = result.unsqueeze(1)
        return result
    elif dim == 3:
        result = (
            vel1[:, 0, ...] * vel2[:, 0, ...]
            + vel1[:, :, 1, ...] * vel2[:, 1, ...]
            + vel1[:, :, 2, ...] * vel2[:, 2, ...]
        )
        if keep_dim:
            result = result.unsqueeze(1)
        return result


def union(self, grid1: torch.Tensor, grid2: torch.Tensor, alpha: float = 0.0):
    """Performs union operator: self {cup} lvl2dGrid,

    Args:
        lvl2dGrid (Level2dGrid): The grid to perform the union with
        alpha (float): parameter to control blending,
                   should be in the range [0,1], alpha=1.0 is equivalent
                   to boolean blending, i.e. torch.min(f, g)
    """
    if alpha < 0.0 or alpha > 1.0:
        raise ValueError("alpha should be in the range [0,1].")
    if alpha == 1.0:
        result = torch.min(grid1, grid2)
    else:
        result = (
            1.0
            / (1.0 + alpha)
            * (
                grid1
                + grid2
                - torch.sqrt(grid1 * grid1 + grid2 * grid2 - 2 * alpha * grid1 * grid2)
            )
        )
    return result
