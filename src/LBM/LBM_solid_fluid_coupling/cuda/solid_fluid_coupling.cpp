#include <torch/extension.h>

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
    int axisymmetric_type);

void solve_obstacle_2d_forward(
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
    int64_t axisymmetric_type)
{
    obstacle_2d_forward(
        dt, dx, tau, rho, vel, flags, f, f_new, phi_obs, obs_vel, is_convection, axisymmetric_type);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("solve_obstacle_2d_forward", &solve_obstacle_2d_forward, "Forward: solve solid-fluid coupling");
}

TORCH_LIBRARY(solid_fluid_coupling, m)
{
    m.def("solve_obstacle_2d_forward", solve_obstacle_2d_forward);
}