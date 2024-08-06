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
__global__ void kernel_get_grad_2d_forward(
    const scalar_t *rho,
    const uint8_t *flags,
    scalar_t *grad_rho,
    double dx,
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
    const int rho_offset = batch_idx * res_yx;
    const int vel_offset = batch_idx * (dim * res_yx);

    grad_rho[vel_offset + 0 * res_yx + y_idx * res_x + x_idx] = 0;
    grad_rho[vel_offset + 1 * res_yx + y_idx * res_x + x_idx] = 0;
    if (flags[idx] == CELLTYPE_OBSTACLE)
    {
        // We don't care about obstacles
        return;
    }

    bool on_x_axis = false;
    bool on_y_axis = false;
    if (x_idx == 0 && axisymmetric_type == AXISYMMETRIC_LINE_X_EQ_0)
    {
        on_x_axis = true;
    }
    else if (y_idx == 0 && axisymmetric_type == AXISYMMETRIC_LINE_Y_EQ_0)
    {
        on_y_axis = true;
    }
    const int neg_x = on_x_axis ? 0 : (flags[idx] == CELLTYPE_FLUID ? (x_idx - 1 + res_x) % res_x : max(0, x_idx - 1));
    const int pos_x = flags[idx] == CELLTYPE_FLUID ? (x_idx + 1 + res_x) % res_x : min(res_x - 1, x_idx + 1);
    const int neg_y = on_y_axis ? 0 : (flags[idx] == CELLTYPE_FLUID ? (y_idx - 1 + res_y) % res_y : max(0, y_idx - 1));
    const int pos_y = flags[idx] == CELLTYPE_FLUID ? (y_idx + 1 + res_y) % res_y : min(res_y - 1, y_idx + 1);
    const int outcome_idx[9] = {
        y_idx * res_x + x_idx,
        y_idx * res_x + pos_x,
        pos_y * res_x + x_idx,
        y_idx * res_x + neg_x,
        neg_y * res_x + x_idx,
        pos_y * res_x + pos_x,
        pos_y * res_x + neg_x,
        neg_y * res_x + neg_x,
        neg_y * res_x + pos_x};

    bool detected_obs = false;
    if (flags[rho_offset + outcome_idx[1]] != CELLTYPE_OBSTACLE && flags[rho_offset + outcome_idx[3]] != CELLTYPE_OBSTACLE)
    {
        grad_rho[vel_offset + 0 * res_yx + outcome_idx[0]] += 4 * (rho[rho_offset + outcome_idx[1]] - rho[rho_offset + outcome_idx[3]]);
    } // else: We assume no Neumann flux from obstacles
    else
    {
        detected_obs = true;
    }
    if (flags[rho_offset + outcome_idx[5]] != CELLTYPE_OBSTACLE && flags[rho_offset + outcome_idx[6]] != CELLTYPE_OBSTACLE)
    {
        grad_rho[vel_offset + 0 * res_yx + outcome_idx[0]] += rho[rho_offset + outcome_idx[5]] - rho[rho_offset + outcome_idx[6]];
    }
    else
    {
        detected_obs = true;
    }
    if (flags[rho_offset + outcome_idx[7]] != CELLTYPE_OBSTACLE && flags[rho_offset + outcome_idx[8]] != CELLTYPE_OBSTACLE)
    {
        grad_rho[vel_offset + 0 * res_yx + outcome_idx[0]] += rho[rho_offset + outcome_idx[8]] - rho[rho_offset + outcome_idx[7]];
    }
    else
    {
        detected_obs = true;
    }

    if (detected_obs)
    {
        grad_rho[vel_offset + 0 * res_yx + outcome_idx[0]] = 0;
    }

    detected_obs = false;
    if (flags[rho_offset + outcome_idx[2]] != CELLTYPE_OBSTACLE && flags[rho_offset + outcome_idx[4]] != CELLTYPE_OBSTACLE)
    {
        grad_rho[vel_offset + 1 * res_yx + outcome_idx[0]] += 4 * (rho[rho_offset + outcome_idx[2]] - rho[rho_offset + outcome_idx[4]]);
    } // else: We assume no Neumann flux from obstacles
    else
    {
        detected_obs = true;
    }
    if (flags[rho_offset + outcome_idx[5]] != CELLTYPE_OBSTACLE && flags[rho_offset + outcome_idx[8]] != CELLTYPE_OBSTACLE)
    {
        grad_rho[vel_offset + 1 * res_yx + outcome_idx[0]] += rho[rho_offset + outcome_idx[5]] - rho[rho_offset + outcome_idx[8]];
    }
    else
    {
        detected_obs = true;
    }
    if (flags[rho_offset + outcome_idx[6]] != CELLTYPE_OBSTACLE && flags[rho_offset + outcome_idx[7]] != CELLTYPE_OBSTACLE)
    {
        grad_rho[vel_offset + 1 * res_yx + outcome_idx[0]] += rho[rho_offset + outcome_idx[6]] - rho[rho_offset + outcome_idx[7]];
    }
    else
    {
        detected_obs = true;
    }

    if (detected_obs)
    {
        grad_rho[vel_offset + 1 * res_yx + outcome_idx[0]] = 0;
    }

    grad_rho[vel_offset + 0 * res_yx + outcome_idx[0]] /= 12 * dx;
    grad_rho[vel_offset + 1 * res_yx + outcome_idx[0]] /= 12 * dx;
}

template <typename scalar_t>
__global__ void kernel_get_div_2d_forward(
    const scalar_t *vel,
    const uint8_t *flags,
    scalar_t *div_vel,
    double dx,
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
    const int rho_offset = batch_idx * res_yx;
    const int vel_offset = batch_idx * (dim * res_yx);

    div_vel[idx] = 0;
    if (flags[idx] == CELLTYPE_OBSTACLE)
    {
        // We don't care about obstacles
        return;
    }

    bool on_x_axis = false;
    bool on_y_axis = false;
    if (x_idx == 0 && axisymmetric_type == AXISYMMETRIC_LINE_X_EQ_0)
    {
        on_x_axis = true;
    }
    else if (y_idx == 0 && axisymmetric_type == AXISYMMETRIC_LINE_Y_EQ_0)
    {
        on_y_axis = true;
    }
    const int neg_x = on_x_axis ? 0 : (flags[idx] == CELLTYPE_FLUID ? (x_idx - 1 + res_x) % res_x : max(0, x_idx - 1));
    const int pos_x = flags[idx] == CELLTYPE_FLUID ? (x_idx + 1 + res_x) % res_x : min(res_x - 1, x_idx + 1);
    const int neg_y = on_y_axis ? 0 : (flags[idx] == CELLTYPE_FLUID ? (y_idx - 1 + res_y) % res_y : max(0, y_idx - 1));
    const int pos_y = flags[idx] == CELLTYPE_FLUID ? (y_idx + 1 + res_y) % res_y : min(res_y - 1, y_idx + 1);
    const int outcome_idx[9] = {
        y_idx * res_x + x_idx,
        y_idx * res_x + pos_x,
        pos_y * res_x + x_idx,
        y_idx * res_x + neg_x,
        neg_y * res_x + x_idx,
        pos_y * res_x + pos_x,
        pos_y * res_x + neg_x,
        neg_y * res_x + neg_x,
        neg_y * res_x + pos_x};

    div_vel[idx] += 4 * (vel[vel_offset + outcome_idx[1]] - vel[vel_offset + outcome_idx[3]]);
    div_vel[idx] += vel[vel_offset + outcome_idx[5]] - vel[vel_offset + outcome_idx[6]];
    div_vel[idx] += vel[vel_offset + outcome_idx[8]] - vel[vel_offset + outcome_idx[7]];

    div_vel[idx] += 4 * (vel[vel_offset + res_yx + outcome_idx[2]] - vel[vel_offset + res_yx + outcome_idx[4]]);
    div_vel[idx] += vel[vel_offset + res_yx + outcome_idx[5]] - vel[vel_offset + res_yx + outcome_idx[8]];
    div_vel[idx] += vel[vel_offset + res_yx + outcome_idx[6]] - vel[vel_offset + res_yx + outcome_idx[7]];

    div_vel[idx] /= 12 * dx;
}

template <typename scalar_t>
__global__ void kernel_collision_2d_forward(
    double dx,
    double dt,
    double tau,
    const scalar_t *f,
    const scalar_t *rho,
    const scalar_t *vel,
    const uint8_t *flags,
    const scalar_t *force,
    const scalar_t *mesh_grid,
    scalar_t *new_f,
    bool is_convection,
    int KBC_type,
    int axisymmetric_type,
    int batch_size,
    int res_y,
    int res_x)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    const int Q = 9;
    const int dim = 2;
    const scalar_t c = dx / dt;
    const scalar_t cs2 = c * c / 3.0;

    if (idx >= batch_size * res_y * res_x)
    {
        // This thread out of the grid boundary
        return;
    }

    const int e[9][2] = {{0, 0}, {1, 0}, {0, 1}, {-1, 0}, {0, -1}, {1, 1}, {-1, 1}, {-1, -1}, {1, -1}};
    const scalar_t weight[9] = {
        4.0 / 9.0,
        1.0 / 9.0,
        1.0 / 9.0,
        1.0 / 9.0,
        1.0 / 9.0,
        1.0 / 36.0,
        1.0 / 36.0,
        1.0 / 36.0,
        1.0 / 36.0,
    };

    const int res_yx = res_y * res_x;
    const int batch_idx = idx / res_yx;
    const int y_idx = idx % res_yx / res_x;
    const int x_idx = idx % res_yx % res_x;

    const int rho_offset = batch_idx * res_yx;
    const int vel_offset = batch_idx * (dim * res_yx);
    const int f_offset = batch_idx * (Q * res_yx);

    if (flags[idx] == CELLTYPE_OBSTACLE)
    {
        // We don't care about obstacles
        return;
    }
    const int outcome_idx[1] = {y_idx * res_x + x_idx};
    scalar_t ux = vel[vel_offset + 0 * res_yx + outcome_idx[0]];
    scalar_t uy = vel[vel_offset + 1 * res_yx + outcome_idx[0]];

    // 1. Calculate feq
    scalar_t eps = 1e-10;
    if (force != nullptr && rho[idx] > eps)
    {
        ux = ux + 0.5 * force[vel_offset + 0 * res_yx + outcome_idx[0]] / rho[idx];
        uy = uy + 0.5 * force[vel_offset + 1 * res_yx + outcome_idx[0]] / rho[idx];
    }
    scalar_t uv = ux * ux + uy * uy;
    scalar_t feq[9] = {0.0};
    for (unsigned int Q_id = 0; Q_id < Q; ++Q_id)
    {
        scalar_t eu = ux * e[Q_id][0] + uy * e[Q_id][1];
        feq[Q_id] = rho[idx] * weight[Q_id] * (1.0 + eu / cs2);
        // if (!is_convection)
        // {
        feq[Q_id] += rho[idx] * weight[Q_id] * (0.5 * eu * eu / cs2 / cs2 - 0.5 * uv / cs2);
        // }
    }

    // 2. Compute Gi if in axisymmetric
    scalar_t Gi[9] = {0.0};
    if (axisymmetric_type != AXISYMMETRIC_NOT)
    {
        scalar_t r = 0.0;
        scalar_t ur = 0.0;
        int r_axis = -1;
        if (axisymmetric_type == AXISYMMETRIC_LINE_X_EQ_0)
        {
            r = mesh_grid[vel_offset + 0 * res_yx + outcome_idx[0]];
            ur = ux;
            r_axis = 0;
        }
        else if (axisymmetric_type == AXISYMMETRIC_LINE_Y_EQ_0)
        {
            r = mesh_grid[vel_offset + 1 * res_yx + outcome_idx[0]];
            ur = uy;
            r_axis = 1;
        }

        if (is_convection)
        {
            // convection
            for (unsigned int Q_id = 0; Q_id < Q; ++Q_id)
            {
                scalar_t s = (1 - 0.5 / tau) * e[Q_id][r_axis] / r;
                Gi[Q_id] = -ur / r * feq[Q_id] * (1 - 0.5 / tau - 0.5 * s) + s * (feq[Q_id] - f[f_offset + Q_id * res_yx + outcome_idx[0]]);
            }
        }
        else
        {
            // TODO: fluid
            for (unsigned int Q_id = 0; Q_id < Q; ++Q_id)
            {
                Gi[Q_id] = 0.0;
            }
        }
    }
}

void cu_get_grad_2d_forward(
    const torch::Tensor &rho,
    const torch::Tensor &flags,
    torch::Tensor &grad_rho,
    double dx,
    int64_t axisymmetric_type)
{
    int batch_size = flags.size(0);
    int res_y = flags.size(2);
    int res_x = flags.size(3);

    const int threads = 512;
    const dim3 blocks((batch_size * res_y * res_x - 1) / threads + 1);

    AT_DISPATCH_FLOATING_TYPES(rho.type(), "cu_get_grad_2d_forward", ([&]
                                                                      { kernel_get_grad_2d_forward<scalar_t><<<blocks, threads>>>(
                                                                            (const scalar_t *)rho.data_ptr(),
                                                                            (const uint8_t *)flags.data_ptr(),
                                                                            (scalar_t *)grad_rho.data_ptr(),
                                                                            dx,
                                                                            axisymmetric_type,
                                                                            batch_size,
                                                                            res_y,
                                                                            res_x); }));
}

void cu_get_div_2d_forward(
    const torch::Tensor &vel,
    const torch::Tensor &flags,
    torch::Tensor &div_vel,
    double dx,
    int64_t axisymmetric_type)
{
    int batch_size = flags.size(0);
    int res_y = flags.size(2);
    int res_x = flags.size(3);

    const int threads = 512;
    const dim3 blocks((batch_size * res_y * res_x - 1) / threads + 1);

    AT_DISPATCH_FLOATING_TYPES(vel.type(), "cu_get_div_2d_forward", ([&]
                                                                     { kernel_get_div_2d_forward<scalar_t><<<blocks, threads>>>(
                                                                           (const scalar_t *)vel.data_ptr(),
                                                                           (const uint8_t *)flags.data_ptr(),
                                                                           (scalar_t *)div_vel.data_ptr(),
                                                                           dx,
                                                                           axisymmetric_type,
                                                                           batch_size,
                                                                           res_y,
                                                                           res_x); }));
}

void cu_collision_2d_forward(
    double dx,
    double dt,
    double tau,
    const torch::Tensor &f,
    const torch::Tensor &rho,
    const torch::Tensor &vel,
    const torch::Tensor &flags,
    const torch::Tensor &force,
    const torch::Tensor &mesh_grid,
    torch::Tensor &new_f,
    bool is_convection,
    int64_t KBC_type,
    int64_t axisymmetric_type)
{
    int batch_size = flags.size(0);
    int res_y = flags.size(2);
    int res_x = flags.size(3);

    const int threads = 512;
    const dim3 blocks((batch_size * res_y * res_x - 1) / threads + 1);

    AT_DISPATCH_FLOATING_TYPES(f.type(), "cu_collision_2d_forward", ([&]
                                                                     { kernel_collision_2d_forward<scalar_t><<<blocks, threads>>>(
                                                                           dx, dt, tau,
                                                                           (const scalar_t *)f.data_ptr(),
                                                                           (const scalar_t *)rho.data_ptr(),
                                                                           (const scalar_t *)vel.data_ptr(),
                                                                           (const uint8_t *)flags.data_ptr(),
                                                                           (const scalar_t *)force.data_ptr(),
                                                                           (const scalar_t *)mesh_grid.data_ptr(),
                                                                           (scalar_t *)new_f.data_ptr(),
                                                                           is_convection,
                                                                           KBC_type,
                                                                           axisymmetric_type,
                                                                           batch_size,
                                                                           res_y,
                                                                           res_x); }));
}