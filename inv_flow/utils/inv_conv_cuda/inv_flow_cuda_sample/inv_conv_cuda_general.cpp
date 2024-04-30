#include <torch/extension.h>

#include <vector>

// CUDA forward declarations

std::vector<torch::Tensor> inv_conv_cuda_inverse(
    torch::Tensor input,
    torch::Tensor kernel,
    torch::Tensor output);

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> inv_conv(
    torch::Tensor input,
    torch::Tensor kernel,
    torch::Tensor output) {
  CHECK_INPUT(input);
  CHECK_INPUT(kernel);
  CHECK_INPUT(output);

  return inv_conv_cuda_inverse(input, kernel, output);
}

std::vector<torch::Tensor> inv_conv_dL_dw_cuda_inverse(
    torch::Tensor input,
    torch::Tensor kernel,
    torch::Tensor loss,
    torch::Tensor M,
    torch::Tensor output);

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> inv_conv_dL_dw(
    torch::Tensor input,
    torch::Tensor kernel,
    torch::Tensor loss,
    torch::Tensor M,
    torch::Tensor output) {
  CHECK_INPUT(input);
  CHECK_INPUT(kernel);
  CHECK_INPUT(loss);
  CHECK_INPUT(M);
  CHECK_INPUT(output);

  return inv_conv_dL_dw_cuda_inverse(input, kernel, loss, M, output);
}

std::vector<torch::Tensor> inv_conv_dL_dx_cuda_inverse(
    torch::Tensor input,
    torch::Tensor kernel,
    torch::Tensor M,
    torch::Tensor output);

// C++ interface
// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> inv_conv_dL_dx(
    torch::Tensor input,
    torch::Tensor kernel,
    torch::Tensor M,
    torch::Tensor output) {
  CHECK_INPUT(input);
  CHECK_INPUT(kernel);
  CHECK_INPUT(M);
  CHECK_INPUT(output);

  return inv_conv_dL_dx_cuda_inverse(input, kernel, M, output);
}

// CUDA forward declarations

std::vector<torch::Tensor> inv_conv_fwd_cuda_inverse(
    torch::Tensor input,
    torch::Tensor kernel,
    torch::Tensor output);

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::vector<torch::Tensor> inv_conv_fwd(
    torch::Tensor input,
    torch::Tensor kernel,
    torch::Tensor output) {
  CHECK_INPUT(input);
  CHECK_INPUT(kernel);
  CHECK_INPUT(output);

  return inv_conv_fwd_cuda_inverse(input, kernel, output);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("inverse", &inv_conv, "inv_conv Inverse (CUDA)");
  m.def("dL_dw", &inv_conv_dL_dw, "dL_dw for Inverse Conv (CUDA)");
  m.def("dL_dx", &inv_conv_dL_dx, "dL_dx for Inverse Conv (CUDA)");
  m.def("forward", &inv_conv_fwd, "inv_conv_fwd Inverse (CUDA)");
}
