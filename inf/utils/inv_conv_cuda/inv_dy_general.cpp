#include <torch/extension.h>

#include <vector>

// CUDA forward declarations
std::vector<torch::Tensor> inv_conv_dy(
    torch::Tensor input,
    torch::Tensor kernel,
    torch::Tensor M,
    torch::Tensor output);

// C++ interface
// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> inv_conv_dL_dy(
    torch::Tensor input,
    torch::Tensor kernel,
    torch::Tensor M,
    torch::Tensor output) {
  CHECK_INPUT(input);
  CHECK_INPUT(kernel);
  CHECK_INPUT(M);
  CHECK_INPUT(output);

  return inv_conv_dy(input, kernel, M, output);
}

// # define TORCH_EXTENSION_NAME inv_conv_dL_dx_cuda_general
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dy", &inv_conv_dL_dy, "dL_dy for Inverse Conv (CUDA)");
}
// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//   m.def("dL_dy", &inv_conv_dL_dx, "dL_dy for Inverse Conv (CUDA)");
// }
