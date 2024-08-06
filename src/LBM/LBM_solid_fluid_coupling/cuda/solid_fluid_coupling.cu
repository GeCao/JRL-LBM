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

#define OBS_BOX 0
#define OBS_SPHERE 1

#define AXISYMMETRIC_NOT 0
#define AXISYMMETRIC_LINE_X_EQ_0 1
#define AXISYMMETRIC_LINE_Y_EQ_0 2
#define AXISYMMETRIC_LINE_Z_EQ_0 3

template <typename scalar_t>
__global__ void kernel_2d_forward(
    double dt,
    double dx,
    double tau,
    const scalar_t *rho,
    const scalar_t *vel,
    const uint8_t *flags,
    const scalar_t *f,
    scalar_t *f_new,
    const scalar_t *phi_obs,
    const scalar_t *obs_vel,
    bool is_convection,
    int axisymmetric_type,
    int batch_size,
    int res_y,
    int res_x)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    const int Q = 9;
    const int dim = 2;

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
    const scalar_t e_normalized[9][2] = {{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}, {-1.0, 0.0}, {0.0, -1.0}, {half_sqrt_2, half_sqrt_2}, {-half_sqrt_2, half_sqrt_2}, {-half_sqrt_2, -half_sqrt_2}, {half_sqrt_2, -half_sqrt_2}};
    const int e_idx_reflect[9] = {0, 3, 4, 1, 2, 7, 8, 5, 6};
    const int e_idx_xaxis_symmetric[9] = {0, 3, 2, 1, 4, 6, 5, 8, 7};
    const int e_idx_yaxis_symmetric[9] = {0, 1, 4, 3, 2, 8, 7, 6, 5};

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
    const int f_offset = batch_idx * (Q * res_yx);

    for (unsigned int Q_idx = 0; Q_idx < Q; ++Q_idx)
    {
        f_new[f_offset + Q_idx * res_yx + y_idx * (res_x) + x_idx] =
            f[f_offset + Q_idx * res_yx + y_idx * (res_x) + x_idx];
    }

    if (flags[idx] == CELLTYPE_OBSTACLE)
    {
        // We do not care about obstacle node
        return;
    }

    if (x_idx == res_x - 1 || y_idx == res_y - 1)
    {
        // Boundary points will be handled specifically,
        // Besides, their neighbor is ambiguous
        return;
    }

    bool on_x_axis = false;
    bool on_y_axis = false;
    if (x_idx == 0 && axisymmetric_type == AXISYMMETRIC_LINE_X_EQ_0)
    {
        on_x_axis = true;
    }
    if (y_idx == 0 && axisymmetric_type == AXISYMMETRIC_LINE_Y_EQ_0)
    {
        on_y_axis = true;
    }

    const int neg_x = on_x_axis ? 0 : (flags[idx] == CELLTYPE_FLUID ? (x_idx - 1 + res_x) % res_x : max(0, x_idx - 1));
    const int pos_x = flags[idx] == CELLTYPE_FLUID ? (x_idx + 1) % res_x : min(res_x - 1, x_idx + 1);
    const int neg_y = on_y_axis ? 0 : (flags[idx] == CELLTYPE_FLUID ? (y_idx - 1 + res_y) % res_y : max(0, y_idx - 1));
    const int pos_y = flags[idx] == CELLTYPE_FLUID ? (y_idx + 1) % res_y : min(res_y - 1, y_idx + 1);
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
    const int income_idx[9] = {
        y_idx * res_x + x_idx,
        y_idx * res_x + neg_x,
        neg_y * res_x + x_idx,
        y_idx * res_x + pos_x,
        pos_y * res_x + x_idx,
        neg_y * res_x + neg_x,
        neg_y * res_x + pos_x,
        pos_y * res_x + pos_x,
        pos_y * res_x + neg_x};

    const scalar_t c = dx / dt;
    const scalar_t cs2 = c * c / 3.0;
    scalar_t vel_tgt[2] = {0.0, 0.0};
    scalar_t rho_tgt = 0.0;
    bool is_obs[9] = {false};
    const scalar_t eps = 1e-10;
    unsigned int count_of_obs_neighbor = 0;
    for (unsigned int i = 0; i < Q; ++i)
    {
        int Q_idx = i;
        int Q_idx_inv = e_idx_reflect[Q_idx];
        const int idx_neighbor = rho_offset + outcome_idx[Q_idx];
        if (flags[idx_neighbor] == CELLTYPE_OBSTACLE)
        {
            // Found a neighbor of obstacle
            // 1. Try to get t
            const scalar_t pos_i[2] = {x_idx + 0.5 * dx, y_idx + 0.5 * dx}; // Always keep your position centered
            const scalar_t t = phi_obs[idx];
            if (t > -eps && t <= 1 + eps)
            {
                // This is a correct obstacle!
                // 2. Get your velocity_target.
                is_obs[i] = true;
                count_of_obs_neighbor += 1;
                scalar_t ux = vel[vel_offset + 0 * res_yx + income_idx[Q_idx]];
                scalar_t uy = vel[vel_offset + 1 * res_yx + income_idx[Q_idx]];
                if (on_x_axis && (i == 1 || i == 5 || i == 8))
                {
                    ux = -ux;
                }
                if (on_y_axis && (i == 2 || i == 5 || i == 6))
                {
                    uy = -uy;
                }
                if (obs_vel != nullptr)
                {
                    const scalar_t obs_ux = obs_vel[vel_offset + 0 * res_yx + outcome_idx[Q_idx]];
                    const scalar_t obs_uy = obs_vel[vel_offset + 1 * res_yx + outcome_idx[Q_idx]];
                    vel_tgt[0] += (t * ux + obs_ux) / (t + 1.0);
                    vel_tgt[1] += (t * uy + obs_uy) / (t + 1.0);
                    // Only when obstacle is moving
                    // const int r = (x_idx == 0 || !is_convection) ? 1 : x_idx;
                    rho_tgt += 6.0 * weight[Q_idx] * (e[Q_idx_inv][0] * obs_ux + e[Q_idx_inv][1] * obs_uy);
                }
                else
                {
                    vel_tgt[0] += t * ux / (t + 1.0);
                    vel_tgt[1] += t * uy / (t + 1.0);
                }
            }
        }

        // 3. Get your rho_target.
        // You might did not find an obstacle you want,
        // However, its still an obstacle and requries a reflection.
        // IMPORTANT: Please make sure all of your obstacle boundaries have been reflected.
        //
        // |--------|--------|
        // |  OBS   |  FLUID |
        // |        |        |
        // |--------|--------|
        //  Q_idx = 3 (-x axix direction)
        // It controls the e_idx_reflect[Q_idx] = 1 part,
        // Which will be rebounced by obstacle area with 1-direction also.
        rho_tgt += f[f_offset + Q_idx_inv * res_yx + outcome_idx[0]];
    }
    if (count_of_obs_neighbor == 0)
    {
        // No obstacle neighbor node found.
        return;
    }
    vel_tgt[0] /= count_of_obs_neighbor;
    vel_tgt[1] /= count_of_obs_neighbor;

    // 4. Get your pressure tensor target
    scalar_t grad_u[2][2] = {{0.0, 0.0}, {0.0, 0.0}};
    grad_u[0][0] = (4.0 * vel[vel_offset + 0 * res_yx + outcome_idx[1]] -
                    4.0 * vel[vel_offset + 0 * res_yx + outcome_idx[3]] +
                    vel[vel_offset + 0 * res_yx + outcome_idx[5]] -
                    vel[vel_offset + 0 * res_yx + outcome_idx[6]] +
                    vel[vel_offset + 0 * res_yx + outcome_idx[8]] -
                    vel[vel_offset + 0 * res_yx + outcome_idx[7]]) /
                   12.0;
    grad_u[0][1] = (4.0 * vel[vel_offset + 0 * res_yx + outcome_idx[2]] -
                    4.0 * vel[vel_offset + 0 * res_yx + outcome_idx[4]] +
                    vel[vel_offset + 0 * res_yx + outcome_idx[5]] -
                    vel[vel_offset + 0 * res_yx + outcome_idx[8]] +
                    vel[vel_offset + 0 * res_yx + outcome_idx[6]] -
                    vel[vel_offset + 0 * res_yx + outcome_idx[7]]) /
                   12.0;
    grad_u[1][0] = (4.0 * vel[vel_offset + 1 * res_yx + outcome_idx[1]] -
                    4.0 * vel[vel_offset + 1 * res_yx + outcome_idx[3]] +
                    vel[vel_offset + 1 * res_yx + outcome_idx[5]] -
                    vel[vel_offset + 1 * res_yx + outcome_idx[6]] +
                    vel[vel_offset + 1 * res_yx + outcome_idx[8]] -
                    vel[vel_offset + 1 * res_yx + outcome_idx[7]]) /
                   12.0;
    grad_u[1][1] = (4.0 * vel[vel_offset + 1 * res_yx + outcome_idx[2]] -
                    4.0 * vel[vel_offset + 1 * res_yx + outcome_idx[4]] +
                    vel[vel_offset + 1 * res_yx + outcome_idx[5]] -
                    vel[vel_offset + 1 * res_yx + outcome_idx[8]] +
                    vel[vel_offset + 1 * res_yx + outcome_idx[6]] -
                    vel[vel_offset + 1 * res_yx + outcome_idx[7]]) /
                   12.0;
    for (unsigned int Q_idx = 0; Q_idx < Q; ++Q_idx)
    {
        scalar_t Pab_part = 0.0;
        int Q_idx_inv = e_idx_reflect[Q_idx];
        if (is_obs[Q_idx])
        {
            Pab_part += (rho_tgt * vel_tgt[0] * vel_tgt[0] - rho_tgt * cs2 * tau * (grad_u[0][0] + grad_u[0][0])) * (e[Q_idx_inv][0] * e[Q_idx_inv][0] - cs2);
            Pab_part += (rho_tgt * vel_tgt[1] * vel_tgt[0] - rho_tgt * cs2 * tau * (grad_u[1][0] + grad_u[0][1])) * (e[Q_idx_inv][1] * e[Q_idx_inv][0] - 0);
            Pab_part += (rho_tgt * vel_tgt[0] * vel_tgt[1] - rho_tgt * cs2 * tau * (grad_u[0][1] + grad_u[1][0])) * (e[Q_idx_inv][0] * e[Q_idx_inv][1] - 0);
            Pab_part += (rho_tgt * vel_tgt[1] * vel_tgt[1] - rho_tgt * cs2 * tau * (grad_u[1][1] + grad_u[1][1])) * (e[Q_idx_inv][1] * e[Q_idx_inv][1] - cs2);
            f_new[f_offset + Q_idx_inv * res_yx + outcome_idx[0]] =
                weight[Q_idx_inv] * (rho_tgt + rho_tgt * c / cs2 * (vel_tgt[0] * e[Q_idx_inv][0] + vel_tgt[1] * e[Q_idx_inv][1]) + 0.5 * Pab_part / cs2 / cs2);
        }
    }
}

void obstacle_2d_forward(
    double dt,
    double dx,
    double tau,
    const torch::Tensor &rho,
    const torch::Tensor &vel,
    const torch::Tensor &flags,
    const torch::Tensor &f,
    torch::Tensor &f_new,
    const torch::Tensor &phi_obs,
    const torch::Tensor &obs_vel,
    bool is_convection,
    int axisymmetric_type)
{
    int batch_size = f.size(0);
    int res_y = f.size(2);
    int res_x = f.size(3);

    const int threads = 512;
    const dim3 blocks((batch_size * res_y * res_x - 1) / threads + 1);

    AT_DISPATCH_FLOATING_TYPES(f.type(), "obstacle_2d_forward", ([&]
                                                                 { kernel_2d_forward<scalar_t><<<blocks, threads>>>(
                                                                       dt, dx, tau,
                                                                       (scalar_t *)rho.data_ptr(),
                                                                       (scalar_t *)vel.data_ptr(),
                                                                       (const uint8_t *)flags.data_ptr(),
                                                                       (const scalar_t *)f.data_ptr(),
                                                                       (scalar_t *)f_new.data_ptr(),
                                                                       (const scalar_t *)phi_obs.data_ptr(),
                                                                       obs_vel.numel() == 0 ? nullptr : (const scalar_t *)obs_vel.data_ptr(),
                                                                       is_convection,
                                                                       axisymmetric_type,
                                                                       batch_size,
                                                                       res_y,
                                                                       res_x); }));
}