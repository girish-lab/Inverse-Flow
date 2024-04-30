from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='inv_conv_cuda',
    ext_modules=[
        CUDAExtension('inv_conv', [
            'inv_conv_cuda_general.cpp',
            'inv_conv_cuda_kernel_general.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })


# # load the cuda extension for inverse pass
# inv_conv_cuda = load(
#     'inv_conv_cuda', ['inv_conv_cuda/inv_conv_cuda_general.cpp', 
#                       'inv_conv_cuda/inv_conv_cuda_kernel_general.cu'], verbose=True)
# # load the cuda extension for forward pass
# inv_conv_fwd_cuda = load(
#     'inv_conv_fwd_cuda', ['inv_conv_cuda/inv_conv_fwd_cuda_general.cpp',
#                            'inv_conv_cuda/inv_conv_fwd_cuda_kernel_general.cu'], verbose=True)
# # load the cuda extension for backward pass dL/dx
# inv_conv_dL_dy_cuda = load(
#     'inv_conv_dL_dx_cuda', ['inv_conv_cuda/inv_conv_dL_dx_cuda_general.cpp',
#                              'inv_conv_cuda/inv_conv_dL_dx_cuda_kernel_general.cu'], verbose=True)
# # # load the cuda extension for backward pass dL/dw
# inv_conv_dL_dw_cuda = load(
#     'inv_conv_dL_dw_cuda', ['inv_conv_cuda/inv_conv_dL_dw_cuda_general.cpp',
#                              'inv_conv_cuda/inv_conv_dL_dw_cuda_kernel_general_2d.cu'], verbose=True)con