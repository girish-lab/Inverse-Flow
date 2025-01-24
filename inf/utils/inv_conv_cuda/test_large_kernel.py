# from util import *
import numpy as np
# from util import test_inverse
import torch
import time
# cuda load module
from torch.utils.cpp_extension import load
CUDA_LAUNCH_BLOCKING=1
# NOTE: important to mention in the paper also, that if kernel[-1,-1] is not 1, it can either result in 
# too small values in the output or large numerical errors.
# TODO: Need to arrive at a theoretical justification for above.
print('done importing cuda modules ...')
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
inv_conv_dL_dw_cuda = load(
    'inv_conv_dL_dw_cuda', ['inv_dL_dw_cuda_general.cpp', 'inv_dL_dw_cuda_kernel_general_2d.cu'], verbose=True)


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
    X = torch.tensor(input, dtype=torch.float).cuda()
    W = torch.tensor(kernel, dtype=torch.float).cuda()
    c_in = X.size()[0]
    n = X.size()[1]
    k = W.size()[0]

    print("Input (x): \n",X)
    print("Kernel (W): \n", W)
    
    
    # print("test_inverse-- m, n, k : ", m, n, k) // 1, 3, 2
    
    output0 = torch.zeros((c_in, n, n), dtype=torch.float).cuda()
    output2 = torch.zeros((c_in, n, n), dtype=torch.float).cuda()
    output3 = torch.zeros((k, k), dtype=torch.float).cuda() # output for dL/dw
    M_dw    = torch.zeros((k, k, n, n), dtype=torch.float).cuda()
    
    t = time.process_time()

    # output0 =  inv_conv_cuda.inverse(X, W, output0) # x = input
    # print("Output (y) : \n", output0[0])

    # output_x = inv_conv_fwd_cuda.forward(output0[0], W, output2) # y = output'
    # print("Output (x`) : \n", output_x[0])
    Y = inv_conv_fwd_cuda.forward(X, W, output2) # Y = output'
    print("Output (y) : \n", Y[0])

    # Wp = torch.randn((k,k), dtype=torch.float).cuda()
    Wp =  torch.tensor([[4,2],
                     [3,1]], dtype=torch.float).reshape(2,2).cuda()
    Wp = Wp/10.0
    Wp[-1][-1] = 1.0
    print("Wp : \n", Wp)

    # # temp = W
    # # W = Wp
    # # Wp = temp

    output_temp = inv_conv_cuda.inverse(Y[0], Wp, output0) # y = output
    print("Output (Xp) : \n", output_temp[0])
    Xp = output_temp
    DXp = 2*(Xp[0]-X) # loss/error
    print("error/loss (DXp) : \n", DXp)
    loss = DXp # temp variable to store the loss/error

    #         # dL/dx
    output1 = torch.zeros((c_in, n, n), dtype=torch.float).cuda()
    M_dy       = torch.zeros((c_in, n, n), dtype=torch.float).cuda()
    dL_dx = inv_conv_dL_dy_cuda.dL_dx(DXp, Wp, M_dy, output1)
    print("dL/dx : \n", dL_dx[0])

            # dL/dw
    dL_dw = inv_conv_dL_dw_cuda.dL_dw(X[0], Wp, loss[0], M_dw, output3)
    print("dL/dw : \n", dL_dw[0])

    output4 = torch.zeros((c_in, n, n), dtype=torch.float).cuda()
    X_recon = inv_conv_fwd_cuda.forward(Xp[0], Wp, output4) # x_ = input'
    print("X_recon (x`) : \n", X_recon[0])
    # Wp = Wp - 0.01*dL_dw[0]
    # new_input = inv_conv_fwd_cuda.forward(Xp[0], Wp, output2) # x_ = input'
    t = time.process_time() - t
    # print("  intput_recon (x`) : \n", new_input[0])

    # compute convolution of output with kernel and see if we get the input back
    # error = (input.reshape(m,1,n,n) - new_input[0]).abs().sum().item()

    # print(f"Error (x-x`) : {error}")
    print(f"Time : {t}s")

# if kernel is not normalized, the values in the output becomes very large 
# resulting in large numerical errors
kernel = (np.random.random((5, 5)) - 0.5)/25.0
kernel[-1, -1] = 1.0 # NOTE: its essential that this value is not too small. Othewise it results in large numerical errors
kernel = np.float32(kernel)

X = np.array([
    [2, 3],
    [8, 9]]).reshape(1, 2, 2)
# X = np.array([
#     [1, 2, 3],
#     [4,5,6],
#     [7,8,9]]).reshape( 3, 3)
# X = np.random.random((5, 5))

K = np.array([
    [1, 1],
    [1, 1] ])
# K = np.random.random((3, 3))
test_inverse(
    input = X, # np.random.random((2, 5, 5)) - 0.5,
    kernel= K #kernel
    )

# K1 = np.array([
#     [1, 1],
#     [1, 1]
# ])
# X2 = np.array([
#     [1, 2, 3],
#     [4,5,6],
#     [7,8,9]]).reshape(1, 3, 3)
# test_inverse(
#     input = X2, # np.random.random((2, 5, 5)) - 0.5,
#     kernel= K1 #kernel
#     )

# there some error with larger batchsize or image size. possibly some issues with blocks
# gradient of loss w.r.t y 
    # dL_dx = inv_conv_dL_dy_cuda.dL_dx(DXp, Wp, M_dy, output1) # dL/dx
    # print("dL/dx : \n", dL_dx[0])

    # gradient of loss w.r.t w
    # print(loss.size(), input.size(), kernel.size(), M_dw.size(), output3.size())
    # torch.Size([1, 3, 3]) torch.Size([1, 3, 3]) torch.Size([2, 2]) torch.Size([2, 2, 3, 3]) torch.Size([1, 2, 2])
# import torch

# class MyKernel(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input):
#         # Prepare your input, output tensors
#         output = torch.zeros_like(input)
#         # Assuming `my_cuda_kernel` is your custom kernel
#         # Ensure this is called once per forward pass
#         my_cuda_kernel.launch(input, output)
#         return output

#     @staticmethod
#     def backward(ctx, grad_output):
#         # Handle backward pass if necessary
#         pass

# # Usage
# input_tensor = torch.randn(10, device='cuda')
# result = MyKernel.apply(input_tensor)