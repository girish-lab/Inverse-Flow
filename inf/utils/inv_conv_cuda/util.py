import numpy as np
import torch
import time

from torch.utils.cpp_extension import load

np.set_printoptions(precision=10, suppress=True, linewidth=120)


# load the cuda extension for inverse pass
inv_conv_cuda = load(
    'inv_conv_cuda', ['inv_cuda_general.cpp', 'inv_cuda_kernel_general.cu'], verbose=True)
# load the cuda extension for forward pass
inv_conv_fwd_cuda = load(
    'inv_conv_fwd_cuda', ['inv_fwd_cuda_general.cpp', 'inv_fwd_cuda_kernel_general.cu'], verbose=True)
# load the cuda extension for backward pass dL/dx
inv_conv_dL_dy_cuda = load(
    'inv_conv_dL_dx_cuda', ['inv_dL_dx_cuda_general.cpp', 'inv_dL_dx_cuda_kernel_general.cu'], verbose=True)
# # load the cuda extension for backward pass dL/dw
# inv_conv_dL_dw_cuda = load(
#     'inv_conv_dL_dw_cuda', ['inv_conv_dL_dw_cuda_general.cpp', 'inv_conv_dL_dw_cuda_kernel_general.cu'], verbose=True)


def test_inverse(input, kernel):
    '''
    input : 4D tensor of shape (m, c, n, n), image tensor.
    kernel : weights 4D tensor of shape (k, k), masked kernel.
    inv_conv_cuda.inverse(input, kernel, output) 
        computes the inverse of the convolution operation.
    '''
    print("------- ########## ----------")
    
    # print("Input (x): \n", np.array(input))
    # print("Kernel (W): \n", np.array(kernel))

    input = torch.tensor(input, dtype=torch.float).cuda()
    kernel = torch.tensor(kernel, dtype=torch.float).cuda()

    print("Input (x): \n",input)
    print("Kernel (W): \n", kernel)
    
    m = input.size()[0]
    n = input.size()[1]
    # k = kernel.size()[0]
    
    output0 = torch.zeros((m, n, n), dtype=torch.float).cuda()
    output1 = torch.zeros((m, n, n), dtype=torch.float).cuda()
    output2 = torch.zeros((m, n, n), dtype=torch.float).cuda()
    
    t = time.process_time()

    output_temp = inv_conv_cuda.inverse(input, kernel, output0) # y = output
    print("Output (y) : \n", output_temp[0])
    M = torch.zeros((m, n, n), dtype=torch.float).cuda()
    Xp = output_temp
    DXp = 2*(Xp[0]-input) # loss/error
    print("error/loss (DXp) : \n", DXp)

    dL_dx = inv_conv_dL_dy_cuda.dL_dx(DXp[0], kernel, M, output1) # dL/dx
    print("dL/dx : \n", dL_dx[0])
    # output_temp[0] = output_temp[0].pad((1,0,1,0), mode='constant', value=0)
    new_input = inv_conv_fwd_cuda.forward(Xp[0], kernel, output2) # x_ = input'
    t = time.process_time() - t
    print("  intput_recon (x`) : \n", new_input[0])

    # compute convolution of output with kernel and see if we get the input back
    error = (input.reshape(m,1,n,n) - new_input[0]).abs().sum().item()

    print(f"Error (x-x`) : {error}")
    print(f"Time : {t}s")

    