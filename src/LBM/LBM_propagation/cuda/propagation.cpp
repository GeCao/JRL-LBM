#include <torch/extension.h>

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
    int axisymmetric_type);

void propagation_2d_forward(
    const torch::Tensor &flags,
    const torch::Tensor &f,
    const torch::Tensor &phi_obs,
    torch::Tensor &f_new,
    torch::Tensor &rho,
    torch::Tensor &vel,
    const torch::Tensor &inflow_vel,
    double inflow_density,
    bool is_convection,
    int64_t axisymmetric_type)
{
    prop_2d_forward(flags, f, phi_obs, f_new, rho, vel, inflow_vel, inflow_density, is_convection, axisymmetric_type);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("propagation_2d_forward", &propagation_2d_forward, "Forward: propagation in 2D");
}

TORCH_LIBRARY(propagation, m)
{
    m.def("propagation_2d_forward", propagation_2d_forward);
}