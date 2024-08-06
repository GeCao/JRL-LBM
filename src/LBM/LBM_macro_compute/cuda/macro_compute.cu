#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>

#define half_sqrt_2 0.7071067811865476

#define CELLTYPE_NOTHING 0
#define CELLTYPE_FLUID 1
#define CELLTYPE_OBSTACLE 2
#define CELLTYPE_EMPTY 4
#define CELLTYPE_INFLOW 8
#define CELLTYPE_OUTFLOW 16
#define CELLTYPE_INFLOW_2 32

#define AXISYMMETRIC_NOT 0
#define AXISYMMETRIC_LINE_X_EQ_0 1
#define AXISYMMETRIC_LINE_Y_EQ_0 2
#define AXISYMMETRIC_LINE_Z_EQ_0 3

template <typename scalar_t>
__global__ void kernel_macompute_C_2d_forward(
    double dx,
    double dt,
    const scalar_t *h,
    const scalar_t *C,
    const uint8_t *flags,
    const scalar_t *vel,
    const scalar_t *mesh_grid,
    scalar_t *C_new,
    int axisymmetric_type,
    int batch_size,
    int res_y,
    int res_x)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    const int Q = 9;
    const int dim = 2;

    if (idx >= batch_size * res_y * res_x)
    {
        // This thread out of the grid boundary
        return;
    }

    const int res_yx = res_y * res_x;
    const int batch_idx = idx / res_yx;
    const int y_idx = idx % res_yx / res_x;
    const int x_idx = idx % res_yx % res_x;

    const int outcome_idx[1] = {y_idx * res_x + x_idx};
    const int vel_offset = batch_idx * (dim * res_yx);
    const int h_offset = batch_idx * (Q * res_yx);

    // Navier-Stokes Equation
    // (Or, ) Surface tracking Equation, of course.
    if (flags[idx] == CELLTYPE_OBSTACLE)
    {
        C_new[idx] = C[idx];
    }
    else
    {
        C_new[idx] = 0.0;
        for (unsigned int Q_idx = 0; Q_idx < Q; ++Q_idx)
        {
            C_new[idx] += h[h_offset + Q_idx * res_yx + outcome_idx[0]];
        }
        // According to Li et al, 2009.
        if (axisymmetric_type == AXISYMMETRIC_LINE_X_EQ_0)
        {
            scalar_t ur = vel[vel_offset + 0 * res_yx + outcome_idx[0]];
            scalar_t r = mesh_grid[vel_offset + 0 * res_yx + outcome_idx[0]];
            C_new[idx] = C_new[idx] / (1.0 + 0.5 * dt * ur / r);
        }
    }
}

template <typename scalar_t>
__global__ void kernel_macompute_2d_forward(
    double dx,
    double dt,
    const scalar_t *f,
    const scalar_t *rho,
    const scalar_t *vel,
    const uint8_t *flags,
    scalar_t *rho_new,
    scalar_t *vel_new,
    const scalar_t *g,
    const scalar_t *pressure,
    const scalar_t *density,
    scalar_t *density_new,
    int axisymmetric_type,
    int batch_size,
    int res_y,
    int res_x)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    const int Q = 9;
    const int dim = 2;
    const scalar_t c = dx / dt;

    if (idx >= batch_size * res_y * res_x)
    {
        // This thread out of the grid boundary
        return;
    }

    const int e[9][2] = {{0, 0}, {1, 0}, {0, 1}, {-1, 0}, {0, -1}, {1, 1}, {-1, 1}, {-1, -1}, {1, -1}};

    const int res_yx = res_y * res_x;
    const int batch_idx = idx / res_yx;
    const int y_idx = idx % res_yx / res_x;
    const int x_idx = idx % res_yx % res_x;

    const int outcome_idx[1] = {y_idx * res_x + x_idx};
    const int ux_idx = batch_idx * (dim * res_yx) + 0 * (res_yx) + outcome_idx[0];
    const int uy_idx = batch_idx * (dim * res_yx) + 1 * (res_yx) + outcome_idx[0];
    const int f_offset = batch_idx * (Q * res_yx);

    // Navier-Stokes Equation
    // (Or, ) Surface tracking Equation, of course.
    if (flags[idx] == CELLTYPE_OBSTACLE)
    {
        rho_new[idx] = rho[idx];
        vel_new[ux_idx] = vel[ux_idx];
        vel_new[uy_idx] = vel[uy_idx];
    }
    else
    {
        rho_new[idx] = 0.0;
        vel_new[ux_idx] = 0.0;
        vel_new[uy_idx] = 0.0;
        for (unsigned int Q_idx = 0; Q_idx < Q; ++Q_idx)
        {
            rho_new[idx] += f[f_offset + Q_idx * res_yx + outcome_idx[0]];
        }
        for (unsigned int Q_idx = 0; Q_idx < Q; ++Q_idx)
        {
            vel_new[ux_idx] += f[f_offset + Q_idx * res_yx + outcome_idx[0]] * e[Q_idx][0] * c;
            vel_new[uy_idx] += f[f_offset + Q_idx * res_yx + outcome_idx[0]] * e[Q_idx][1] * c;
        }
        vel_new[ux_idx] /= rho_new[idx];
        vel_new[uy_idx] /= rho_new[idx];
    }

    if (density_new != nullptr)
    {
        // rho is certainly a surface tracking equation!
        // TODO: DO it better!
        density_new[idx] = rho_new[idx];
        // const scalar_t RT = c * c / 3.0;
        // pressure[idx] = density[idx] * RT * density[idx] * (4.0 - 2.0 * density[idx]) / pow((1 - density[idx]), 3) - 4.0 * density[idx] * density[idx] + density[idx] * RT;
    }
}

void macompute_C_2d_forward(
    double dx,
    double dt,
    const torch::Tensor &h,
    const torch::Tensor &C,
    const torch::Tensor &flags,
    const torch::Tensor &vel,
    const torch::Tensor &mesh_grid,
    torch::Tensor &C_new,
    int64_t axisymmetric_type)
{
    int batch_size = h.size(0);
    int res_y = h.size(2);
    int res_x = h.size(3);

    const int threads = 512;
    const dim3 blocks((batch_size * res_y * res_x - 1) / threads + 1);

    AT_DISPATCH_FLOATING_TYPES(h.type(), "macompute_C_2d_forward", ([&]
                                                                    { kernel_macompute_C_2d_forward<scalar_t><<<blocks, threads>>>(
                                                                          dx, dt,
                                                                          (const scalar_t *)h.data_ptr(),
                                                                          (const scalar_t *)C.data_ptr(),
                                                                          (const uint8_t *)flags.data_ptr(),
                                                                          (const scalar_t *)vel.data_ptr(),
                                                                          (const scalar_t *)mesh_grid.data_ptr(),
                                                                          (scalar_t *)C_new.data_ptr(),
                                                                          axisymmetric_type,
                                                                          batch_size,
                                                                          res_y,
                                                                          res_x); }));
}

void macompute_2d_forward(
    double dx,
    double dt,
    const torch::Tensor &f,
    const torch::Tensor &rho,
    const torch::Tensor &vel,
    const torch::Tensor &flags,
    torch::Tensor &rho_new,
    torch::Tensor &vel_new,
    const torch::Tensor &g,
    const torch::Tensor &pressure,
    const torch::Tensor &density,
    torch::Tensor &density_new,
    int64_t axisymmetric_type)
{
    int batch_size = f.size(0);
    int res_y = f.size(2);
    int res_x = f.size(3);

    const int threads = 512;
    const dim3 blocks((batch_size * res_y * res_x - 1) / threads + 1);

    AT_DISPATCH_FLOATING_TYPES(f.type(), "macompute_2d_forward", ([&]
                                                                  { kernel_macompute_2d_forward<scalar_t><<<blocks, threads>>>(
                                                                        dx, dt,
                                                                        (const scalar_t *)f.data_ptr(),
                                                                        (const scalar_t *)rho.data_ptr(),
                                                                        (const scalar_t *)vel.data_ptr(),
                                                                        (const uint8_t *)flags.data_ptr(),
                                                                        (scalar_t *)rho_new.data_ptr(),
                                                                        (scalar_t *)vel_new.data_ptr(),
                                                                        g.numel() == 0 ? nullptr : (const scalar_t *)g.data_ptr(),
                                                                        pressure.numel() == 0 ? nullptr : (const scalar_t *)pressure.data_ptr(),
                                                                        density.numel() == 0 ? nullptr : (const scalar_t *)density.data_ptr(),
                                                                        density_new.numel() == 0 ? nullptr : (scalar_t *)density_new.data_ptr(),
                                                                        axisymmetric_type,
                                                                        batch_size,
                                                                        res_y,
                                                                        res_x); }));
}