#include <torch/extension.h>

void macompute_C_2d_forward(
    double dx,
    double dt,
    const torch::Tensor &h,
    const torch::Tensor &C,
    const torch::Tensor &flags,
    const torch::Tensor &vel,
    const torch::Tensor &mesh_grid,
    torch::Tensor &C_new,
    int64_t axisymmetric_type);

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
    int64_t axisymmetric_type);

void macro_compute_C_2d_forward(
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
    macompute_C_2d_forward(dx, dt, h, C, flags, vel, mesh_grid, C_new, axisymmetric_type);
}

void macro_compute_2d_forward(
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
    macompute_2d_forward(
        dx, dt, f, rho, vel, flags, rho_new, vel_new, g, pressure, density, density_new, axisymmetric_type);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("macro_compute_2d_forward", &macro_compute_2d_forward, "Forward: macro compute fluid in 2D");
    m.def("macro_compute_C_2d_forward", &macro_compute_C_2d_forward, "Forward: macro compute C in 2D");
}

TORCH_LIBRARY(macro_compute, m)
{
    m.def("macro_compute_2d_forward", macro_compute_2d_forward);
    m.def("macro_compute_C_2d_forward", macro_compute_C_2d_forward);
}