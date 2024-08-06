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
__global__ void kernel_prop_2d_forward(
    const uint8_t *flags,
    const scalar_t *f,
    const scalar_t *phi_obs,
    scalar_t *f_new,
    scalar_t *rho,
    scalar_t *vel,
    const scalar_t *inflow_vel,
    double inflow_density,
    bool is_convection,
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
    const int e_idx_x_axisymmetric[9] = {0, 3, 2, 1, 4, 6, 5, 8, 7};
    const int e_idx_y_axisymmetric[9] = {0, 1, 4, 3, 2, 8, 7, 6, 5};

    const int res_yx = res_y * res_x;
    const int batch_idx = idx / res_yx;
    const int y_idx = idx % res_yx / res_x;
    const int x_idx = idx % res_yx % res_x;
    const int rho_offset = batch_idx * res_yx;
    const int vel_offset = batch_idx * (dim * res_yx);
    const int f_offset = batch_idx * (Q * res_yx);

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
    const int x_neg = on_x_axis ? 0 : (flags[idx] == CELLTYPE_FLUID ? (x_idx - 1 + res_x) % res_x : (x_idx - 1 < 0 ? 1 : x_idx - 1));
    const int x_pos = flags[idx] == CELLTYPE_FLUID ? (x_idx + 1) % res_x : (x_idx + 1 >= res_x ? res_x - 2 : x_idx + 1);
    const int y_neg = on_y_axis ? 0 : (flags[idx] == CELLTYPE_FLUID ? (y_idx - 1 + res_y) % res_y : (y_idx - 1 < 0 ? 1 : y_idx - 1));
    const int y_pos = flags[idx] == CELLTYPE_FLUID ? (y_idx + 1) % res_y : (y_idx + 1 >= res_y ? res_y - 2 : y_idx + 1);
    const int income_idx[9] = {
        y_idx * res_x + x_idx,
        y_idx * res_x + x_neg,
        y_neg * res_x + x_idx,
        y_idx * res_x + x_pos,
        y_pos * res_x + x_idx,
        y_neg * res_x + x_neg,
        y_neg * res_x + x_pos,
        y_pos * res_x + x_pos,
        y_pos * res_x + x_neg};

    const int outcome_idx[9] = {
        y_idx * res_x + x_idx,
        y_idx * res_x + x_pos,
        y_pos * res_x + x_idx,
        y_idx * res_x + x_neg,
        y_neg * res_x + x_idx,
        y_pos * res_x + x_pos,
        y_pos * res_x + x_neg,
        y_neg * res_x + x_neg,
        y_neg * res_x + x_pos};

    f_new[f_offset + 0 * res_yx + income_idx[0]] = f[f_offset + 0 * res_yx + income_idx[0]];
    if (flags[idx] == CELLTYPE_OBSTACLE)
    {
        // Bounce-back boundary condition (Do not delete it as it contributes to Solid-Fluid Coupling)
        for (unsigned int Q_idx = 1; Q_idx < Q; Q_idx++)
        {
            unsigned int income_Q_idx = Q_idx;
            if (on_x_axis && (Q_idx == 5 || Q_idx == 8 || Q_idx == 1))
            {
                income_Q_idx = e_idx_x_axisymmetric[Q_idx];
            }
            else if (on_y_axis && (Q_idx == 5 || Q_idx == 6 || Q_idx == 2))
            {
                income_Q_idx = e_idx_y_axisymmetric[Q_idx];
            }

            f_new[f_offset + e_idx_reflect[Q_idx] * res_yx + income_idx[0]] =
                f[f_offset + income_Q_idx * res_yx + income_idx[Q_idx]];
        }
        // Do not return this easily, you should check your axisymmetric setting subsequently.
    }
    else
    {
        for (unsigned int Q_idx = 1; Q_idx < Q; Q_idx++)
        {
            unsigned int income_Q_idx = Q_idx;
            if (on_x_axis && (Q_idx == 5 || Q_idx == 8 || Q_idx == 1))
            {
                income_Q_idx = e_idx_x_axisymmetric[Q_idx];
            }
            else if (on_y_axis && (Q_idx == 5 || Q_idx == 6 || Q_idx == 2))
            {
                income_Q_idx = e_idx_y_axisymmetric[Q_idx];
            }

            if (flags[rho_offset + income_idx[Q_idx]] != CELLTYPE_OBSTACLE)
            {
                // come from fluid
                f_new[f_offset + Q_idx * res_yx + income_idx[0]] =
                    f[f_offset + income_Q_idx * res_yx + income_idx[Q_idx]];
            }
            else
            {
                // come from obstacle
                const scalar_t f_this_step = f[f_offset + e_idx_reflect[Q_idx] * res_yx + income_idx[0]];
                if (phi_obs == nullptr)
                {
                    // default as 0.5-dx obstacle
                    f_new[f_offset + Q_idx * res_yx + income_idx[0]] = f_this_step;
                }
                else
                {
                    const scalar_t t = phi_obs[idx];
                    if (t >= 0.5 && t <= 1)
                    {
                        // Interpolation between this step and previous step.
                        const scalar_t t1 = 2 * t - 1;
                        const scalar_t f_prev_step = f[f_offset + income_Q_idx * res_yx + income_idx[Q_idx]];
                        f_new[f_offset + Q_idx * res_yx + income_idx[0]] = t1 * f_prev_step + (1 - t1) * f_this_step;
                    }
                    else if (t <= 0.5 && t >= 0)
                    {
                        unsigned int outcome_Q_idx = Q_idx;
                        if (on_x_axis && (Q_idx == 6 || Q_idx == 7 || Q_idx == 3))
                        {
                            outcome_Q_idx = e_idx_x_axisymmetric[Q_idx];
                        }
                        else if (on_y_axis && (Q_idx == 7 || Q_idx == 8 || Q_idx == 4))
                        {
                            outcome_Q_idx = e_idx_y_axisymmetric[Q_idx];
                        }
                        const scalar_t t2 = 2 * t;
                        const scalar_t f_next_step = f[f_offset + e_idx_reflect[outcome_Q_idx] * res_yx + outcome_idx[Q_idx]];
                        f_new[f_offset + Q_idx * res_yx + income_idx[0]] = t2 * f_this_step + (1 - t2) * f_next_step;
                    }
                    else
                    {
                        // default as 0.5-dx obstacle
                        f_new[f_offset + Q_idx * res_yx + income_idx[0]] = f_this_step;
                    }
                }
            }
        }
    }

    if (flags[idx] == CELLTYPE_OUTFLOW)
    {
        // OutFlow boundary condition
        int outflow_from = income_idx[0];
        int outflow_from_from = income_idx[0];
        if (x_idx == 0)
        {
            outflow_from = outflow_from + 1;
            outflow_from_from = outflow_from_from + 2;
        }
        else if (x_idx == res_x - 1)
        {
            outflow_from = outflow_from - 1;
            outflow_from_from = outflow_from_from - 2;
        }

        if (y_idx == 0)
        {
            outflow_from = outflow_from + res_x;
            outflow_from_from = outflow_from_from + 2 * res_x;
        }
        else if (y_idx == res_y - 1)
        {
            outflow_from = outflow_from - res_x;
            outflow_from_from = outflow_from_from - 2 * res_x;
        }

        // scalar_t new_rho = rho[rho_offset + outflow_from];            // * 2 - rho[rho_offset + outflow_from_from];
        // scalar_t new_u = vel[vel_offset + outflow_from];              // * 2 - vel[vel_offset + outflow_from_from];
        // scalar_t new_v = vel[vel_offset + 1 * res_yx + outflow_from]; // * 2 - vel[vel_offset + 1 * res_yx + outflow_from_from];

        // if (rho != nullptr)
        // {
        //     rho[idx] = new_rho;
        // }

        // if (vel != nullptr)
        // {
        //     vel[vel_offset + income_idx[0]] = new_u;
        //     vel[vel_offset + 1 * res_yx + income_idx[0]] = new_v;
        // }

        // scalar_t cs2 = 1.0 / 3.0;
        // scalar_t uv = new_u * new_u + new_v * new_v;
        // for (unsigned int Q_id = 0; Q_id < Q; ++Q_id)
        // {
        //     scalar_t eu = new_u * e[Q_id][0] + new_v * e[Q_id][1];
        //     f_new[f_offset + Q_id * res_yx + income_idx[0]] = new_rho * weight[Q_id] * (1.0 + eu / cs2);
        //     if (!is_convection)
        //     {
        //         f_new[f_offset + Q_id * res_yx + income_idx[0]] +=
        //             new_rho * weight[Q_id] * (0.5 * eu * eu / cs2 / cs2 - 0.5 * uv / cs2);
        //     }
        // }

        for (unsigned int Q_idx = 0; Q_idx < Q; ++Q_idx)
        {
            f_new[f_offset + Q_idx * res_yx + income_idx[0]] =
                f_new[f_offset + Q_idx * res_yx + outflow_from];
        }
    }

    // Axisymmetric boundary condition
    // if (x_idx == 0 && axisymmetric_type == AXISYMMETRIC_LINE_X_EQ_0)
    // {
    //     // 1, 5, 8 missing
    //     int miss_idx1 = 1;
    //     int miss_idx2 = 5;
    //     int miss_idx3 = 8;
    //     // 3, 6, 7 fill
    //     int fill_idx1 = 3;
    //     int fill_idx2 = 6;
    //     int fill_idx3 = 7;

    //     f_new[f_offset + miss_idx1 * res_yx + y_idx * (res_x) + x_idx] =
    //         f_new[f_offset + fill_idx1 * res_yx + y_idx * (res_x) + x_idx];
    //     f_new[f_offset + miss_idx2 * res_yx + y_idx * (res_x) + x_idx] =
    //         f_new[f_offset + fill_idx2 * res_yx + y_idx * (res_x) + x_idx];
    //     f_new[f_offset + miss_idx3 * res_yx + y_idx * (res_x) + x_idx] =
    //         f_new[f_offset + fill_idx3 * res_yx + y_idx * (res_x) + x_idx];
    // }
    // else if (y_idx == 0 && axisymmetric_type == AXISYMMETRIC_LINE_Y_EQ_0)
    // {
    //     // 2, 5, 6 missing
    //     int miss_idx1 = 2;
    //     int miss_idx2 = 5;
    //     int miss_idx3 = 6;
    //     // 4, 8, 7 fill
    //     int fill_idx1 = 4;
    //     int fill_idx2 = 8;
    //     int fill_idx3 = 7;

    //     f_new[f_offset + miss_idx1 * res_yx + income_idx[0]] =
    //         f_new[f_offset + fill_idx1 * res_yx + income_idx[0]];
    //     f_new[f_offset + miss_idx2 * res_yx + income_idx[0]] =
    //         f_new[f_offset + fill_idx2 * res_yx + income_idx[0]];
    //     f_new[f_offset + miss_idx3 * res_yx + income_idx[0]] =
    //         f_new[f_offset + fill_idx3 * res_yx + income_idx[0]];
    // }

    if (flags[idx] == CELLTYPE_INFLOW)
    {
        scalar_t rho_temp = 0.0;
        if (rho != nullptr && inflow_density >= 0)
        {
            rho[idx] = inflow_density;
            rho_temp = inflow_density;
        }
        else if (rho != nullptr)
        {
            rho_temp = rho[idx];
        }
        else if (inflow_density >= 0)
        {
            rho_temp = inflow_density;
        }
        else
        {
            printf("Warn: For inflow cell, you have to indicate density either from rho grid or a 1-dim vector.");
        }

        scalar_t ux = 0.0;
        scalar_t uy = 0.0;
        if (vel != nullptr && inflow_vel != nullptr)
        {
            vel[vel_offset + 0 * res_yx + income_idx[0]] = inflow_vel[0];
            vel[vel_offset + 1 * res_yx + income_idx[0]] = inflow_vel[1];
            ux = inflow_vel[0];
            uy = inflow_vel[1];
        }
        else if (vel != nullptr)
        {
            ux = vel[vel_offset + 0 * res_yx + income_idx[0]];
            uy = vel[vel_offset + 1 * res_yx + income_idx[0]];
        }
        else if (inflow_vel != nullptr)
        {
            ux = inflow_vel[0];
            uy = inflow_vel[1];
        }
        // else
        // {
        //     printf("Warn: For inflow cell, you have to indicate velocity either from vel grid or a 2-dim vector.");
        // }

        scalar_t cs2 = 1.0 / 3.0;
        scalar_t uv = ux * ux + uy * uy;
        for (unsigned int Q_id = 0; Q_id < Q; ++Q_id)
        {
            scalar_t eu = ux * e[Q_id][0] + uy * e[Q_id][1];
            f_new[f_offset + Q_id * res_yx + y_idx * (res_x) + x_idx] = rho_temp * weight[Q_id] * (1.0 + eu / cs2);
            // if (!is_convection)
            // {
            f_new[f_offset + Q_id * res_yx + y_idx * (res_x) + x_idx] +=
                rho_temp * weight[Q_id] * (0.5 * eu * eu / cs2 / cs2 - 0.5 * uv / cs2);
            // }
        }
    }

    if (flags[idx] == CELLTYPE_INFLOW_2)
    {
        scalar_t ux = vel == nullptr ? 0 : vel[vel_offset + 0 * res_yx + income_idx[0]];
        scalar_t uy = vel == nullptr ? 0 : vel[vel_offset + 1 * res_yx + income_idx[0]];
        scalar_t rho_temp = 1; //(x_idx + 0.5) / (res_x - 0.5); // TODO: rC

        if (rho != nullptr)
        {
            rho[idx] = rho_temp;
        }

        scalar_t cs2 = 1.0 / 3.0;
        scalar_t uv = ux * ux + uy * uy;
        for (unsigned int Q_id = 0; Q_id < Q; ++Q_id)
        {
            scalar_t eu = ux * e[Q_id][0] + uy * e[Q_id][1];
            f_new[f_offset + Q_id * res_yx + income_idx[0]] = rho_temp * weight[Q_id] * (1.0 + eu / cs2);
            // if (!is_convection)
            // {
            f_new[f_offset + Q_id * res_yx + income_idx[0]] +=
                rho_temp * weight[Q_id] * (0.5 * eu * eu / cs2 / cs2 - 0.5 * uv / cs2);
            // }
        }
    }

    // // Axisymmetric boundary condition
    // if (x_idx == 0 && axisymmetric_type == AXISYMMETRIC_LINE_X_EQ_0)
    // {
    //     // 1, 5, 8 missing
    //     int miss_idx1 = 1;
    //     int miss_idx2 = 5;
    //     int miss_idx3 = 8;
    //     // 3, 6, 7 fill
    //     int fill_idx1 = 3;
    //     int fill_idx2 = 6;
    //     int fill_idx3 = 7;

    //     uint8_t neg_y_flag = flags[rho_offset + y_neg * (res_x) + x_idx];
    //     uint8_t pos_y_flag = flags[rho_offset + y_pos * (res_x) + x_idx];

    //     if (flags[idx] == CELLTYPE_FLUID)
    //     {
    //         // Yourself will never be obstacle
    //         f_new[f_offset + miss_idx1 * res_yx + income_idx[0]] =
    //             f[f_offset + fill_idx1 * res_yx + income_idx[0]];

    //         if (neg_y_flag != CELLTYPE_OBSTACLE)
    //         {
    //             f_new[f_offset + miss_idx2 * res_yx + income_idx[0]] =
    //                 f[f_offset + fill_idx2 * res_yx + y_neg * (res_x) + x_idx];
    //         }

    //         if (pos_y_flag != CELLTYPE_OBSTACLE)
    //         {
    //             f_new[f_offset + miss_idx3 * res_yx + income_idx[0]] =
    //                 f[f_offset + fill_idx3 * res_yx + y_pos * (res_x) + x_idx];
    //         }
    //     }
    //     else if (flags[idx] == CELLTYPE_OBSTACLE)
    //     {
    //         // Yourself is obstacle.
    //         // If your neg_y || pos_y is not obstacle, you are in trouble
    //         if (neg_y_flag != CELLTYPE_OBSTACLE)
    //         {
    //             // You have to handle your missing 7, which is reflected by 5
    //             // 5 is passed by your (neg_x, neg_y) cell, where neg_x ~ x_idx
    //             // In other word, from your (x_idx, neg_y) cell, in 6
    //             f_new[f_offset + 7 * res_yx + income_idx[0]] =
    //                 f[f_offset + 6 * res_yx + y_neg * (res_x) + x_idx];
    //         }
    //         if (pos_y_flag != CELLTYPE_OBSTACLE)
    //         {
    //             // You have to handle your missing 6, which is reflected by 8
    //             // 8 is passed by your (neg_x, pos_y) cell, where neg_x ~ x_idx
    //             // In other word, from your (x_idx, pos_y) cell, in 7
    //             f_new[f_offset + 6 * res_yx + income_idx[0]] =
    //                 f[f_offset + 7 * res_yx + y_pos * (res_x) + x_idx];
    //         }
    //     } // else: Other nodes in axis are specifically handlled, do not take them
    // }
    // else if (y_idx == 0 && axisymmetric_type == AXISYMMETRIC_LINE_Y_EQ_0)
    // {
    //     // 2, 5, 6 missing
    //     int miss_idx1 = 2;
    //     int miss_idx2 = 5;
    //     int miss_idx3 = 6;
    //     // 4, 8, 7 fill
    //     int fill_idx1 = 4;
    //     int fill_idx2 = 8;
    //     int fill_idx3 = 7;

    //     uint8_t neg_x_flag = flags[rho_offset + y_idx * (res_x) + x_neg];
    //     uint8_t pos_x_flag = flags[rho_offset + y_idx * (res_x) + x_pos];

    //     if (flags[idx] == CELLTYPE_FLUID)
    //     {
    //         // Yourself will never be obstacle
    //         f_new[f_offset + miss_idx1 * res_yx + income_idx[0]] =
    //             f[f_offset + fill_idx1 * res_yx + income_idx[0]];

    //         if (neg_x_flag != CELLTYPE_OBSTACLE)
    //         {
    //             f_new[f_offset + miss_idx2 * res_yx + income_idx[0]] =
    //                 f[f_offset + fill_idx2 * res_yx + y_idx * (res_x) + x_neg];
    //         }

    //         if (pos_x_flag != CELLTYPE_OBSTACLE)
    //         {
    //             f_new[f_offset + miss_idx3 * res_yx + income_idx[0]] =
    //                 f[f_offset + fill_idx3 * res_yx + y_idx * (res_x) + x_pos];
    //         }
    //     }
    //     else if (flags[idx] == CELLTYPE_OBSTACLE)
    //     {
    //         // Yourself is obstacle.
    //         // If your neg_x || pos_x is not obstacle, you are in trouble
    //         if (neg_x_flag != CELLTYPE_OBSTACLE)
    //         {
    //             // You have to handle your missing 7, which is reflected by 5
    //             // 5 is passed by your (neg_x, neg_y) cell, where neg_y ~ y_idx
    //             // In other word, from your (neg_x, y_idx) cell, in 8
    //             f_new[f_offset + 7 * res_yx + income_idx[0]] =
    //                 f[f_offset + 8 * res_yx + y_idx * (res_x) + x_neg];
    //         }
    //         if (pos_x_flag != CELLTYPE_OBSTACLE)
    //         {
    //             // You have to handle your missing 8, which is reflected by 6
    //             // 6 is passed by your (pos_x, neg_y) cell, where neg_y ~ y_idx
    //             // In other word, from your (pos_x, y_idx) cell, in 7
    //             f_new[f_offset + 8 * res_yx + income_idx[0]] =
    //                 f[f_offset + 7 * res_yx + y_idx * (res_x) + x_pos];
    //         }
    //     } // else: Other nodes in axis are specifically handlled, do not take them
    // }
}

void prop_2d_forward(
    const torch::Tensor &flags,
    const torch::Tensor &f,
    const torch::Tensor &phi_obs,
    torch::Tensor &f_new,
    torch::Tensor &rho,
    torch::Tensor &vel,
    const torch::Tensor &inflow_vel,
    double inflow_density,
    bool is_convection,
    int axisymmetric_type)
{
    int batch_size = f.size(0);
    int res_y = f.size(2);
    int res_x = f.size(3);

    const int threads = 512;
    const dim3 blocks((batch_size * res_y * res_x - 1) / threads + 1);

    AT_DISPATCH_FLOATING_TYPES(f.type(), "prop_2d_forward", ([&]
                                                             { kernel_prop_2d_forward<scalar_t><<<blocks, threads>>>(
                                                                   (const uint8_t *)flags.data_ptr(),
                                                                   (const scalar_t *)f.data_ptr(),
                                                                   phi_obs.numel() == 0 ? nullptr : (const scalar_t *)phi_obs.data_ptr(),
                                                                   (scalar_t *)f_new.data_ptr(),
                                                                   rho.numel() == 0 ? nullptr : (scalar_t *)rho.data_ptr(),
                                                                   vel.numel() == 0 ? nullptr : (scalar_t *)vel.data_ptr(),
                                                                   inflow_vel.numel() == 0 ? nullptr : (const scalar_t *)inflow_vel.data_ptr(),
                                                                   inflow_density,
                                                                   is_convection,
                                                                   axisymmetric_type,
                                                                   batch_size,
                                                                   res_y,
                                                                   res_x); }));
}
