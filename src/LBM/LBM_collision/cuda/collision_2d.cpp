#include <torch/extension.h>

void cu_get_grad_2d_forward(
    const torch::Tensor &rho,
    const torch::Tensor &flags,
    torch::Tensor &grad_rho,
    double dx,
    int64_t axisymmetric_type);

void cu_get_div_2d_forward(
    const torch::Tensor &vel,
    const torch::Tensor &flags,
    torch::Tensor &div_vel,
    double dx,
    int64_t axisymmetric_type);

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
    int64_t axisymmetric_type);

void get_grad_2d_forward(
    const torch::Tensor &rho,
    const torch::Tensor &flags,
    torch::Tensor &grad_rho,
    double dx,
    int64_t axisymmetric_type)
{
    cu_get_grad_2d_forward(rho, flags, grad_rho, dx, axisymmetric_type);
}

void get_div_2d_forward(
    const torch::Tensor &vel,
    const torch::Tensor &flags,
    torch::Tensor &div_vel,
    double dx,
    int64_t axisymmetric_type)
{
    cu_get_div_2d_forward(vel, flags, div_vel, dx, axisymmetric_type);
}

void collision_2d_forward(
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
    cu_collision_2d_forward(
        dx, dt, tau, f, rho, vel, flags, force, mesh_grid, new_f, is_convection, KBC_type, axisymmetric_type);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("get_grad_2d_forward", &get_grad_2d_forward, "Forward: vector operator, get grad in 2D.");
    m.def("get_div_2d_forward", &get_div_2d_forward, "Forward: vector operator, get div in 2D.");
    m.def("collision_2d_forward", &get_div_2d_forward, "Forward: collision step in 2D.");
}

TORCH_LIBRARY(collision_2d, m)
{
    m.def("get_grad_2d_forward", get_grad_2d_forward);
    m.def("get_div_2d_forward", get_div_2d_forward);
    m.def("collision_2d_forward", collision_2d_forward);
}