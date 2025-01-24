
#%%# from util import *
import numpy as np
# from util import test_inverse
import torch
import time
# cuda load module

''' method-1: Ahead - of - time (AOT) compilation of cuda kernels   '''
# import inv_conv_with_bp


''' OR '''

''' method-2; Just - in - time (JIT) compilation of cuda kernels   '''
from torch.utils.cpp_extension import load
# CUDA_LAUNCH_BLOCKING=1
# # NOTE: important to mention in the paper also, that if kernel[-1,-1] is not 1, it can either result in \
# #  too small values in the output or large numerical errors.
# # TODO: Need to arrive at a theoretical justification for above. done 
# #%%
# print('done importing cuda modules ...')

# # load the cuda extension for inverse pass
inv_conv_inv = load(
    'inv_conv_inv', ['inv_general.cpp', 'inv_kernel_general.cu'], verbose=True)
inv_conv_fwd = load(
    'inv_conv_fwd', ['inv_fwd_general.cpp', 'inv_fwd_kernel_general.cu'], verbose=True)
inv_conv_dy = load(
    'inv_conv_dy', ['inv_dy_general.cpp', 'inv_dy_kernel_general.cu'], verbose=True)
inv_conv_dw = load(
    'inv_conv_dw', ['inv_dw_general.cpp', 'inv_dw_kernel_general.cu'], verbose=True)

# cinc_level2 = load(
#     'cinc_level2', ['cinc_level2.cpp', 'cinc_kernel_level2.cu'], verbose=True) # work same as inv_conv_cuda
# cinc_level1 = load(
#     'cinc_level1', ['cinc_level1.cpp', 'cinc_kernel_level1.cu'], verbose=True) # work for c_in = 1



def test_inverse(input, kernel, Wp, batch_size=1, c_in=4, n=2, k=2):
    '''
        input : 4D tensor of shape (m, c, n, n), image tensor.
        c : number of channels, 4*c () should be multiple of 4
        kernel : weights 4D tensor of shape (k, k), masked kernel.
        inv_conv_cuda.inverse(input, kernel, output) 
            computes the inverse of the convolution operation.
    '''
    print("-------########## ########## ----------")
    # print("Input (x): \n", np.array(input))
    # print("Kernel (W): \n", np.array(kernel))
    X = torch.tensor(input, dtype=torch.float).cuda()  # INPUT
    W = torch.tensor(kernel, dtype=torch.float).cuda() #  KERNEL
    Wp = torch.tensor(Wp, dtype=torch.float).cuda() # KERNEL ()


    print("Input (x): \n", X)
    print("-------########## ########## ----------")
    print("Kernel (W): \n", W)
    print("-------########## ########## ----------")
    print("Kernel (Wp): \n", Wp)
    # t = time.process_time()

    # forward pass
    output_fwd = torch.zeros((batch_size, c_in, n, n), dtype=torch.float).cuda()
    # Y = inv_conv.inverse(X, W, output0)
    Y = inv_conv_fwd.forward(X, W, output_fwd) # Y = output'            # fwd pass # JIT
    # Y = inv_conv_with_bp.forward(X, W, output_fwd) # Y = output'       # fwd pass # AOT
    print("------- ########## ########## ----------")
    print("Output (Y): \n", Y[0])

    # Wp =  torch.randn((c_in, c_in, k, k),dtype=torch.float).cuda()

    # inverse pass
    output_inv = torch.zeros((batch_size, c_in, n, n), dtype=torch.float).cuda()
    output_temp = inv_conv_inv.inverse(Y[0], Wp, output_inv)    # inv pass # JIT
    # output_temp = inv_conv_with_bp.inverse(Y[0], Wp, output_inv)        # inv pass # AOT
    print("-------########## ########## ----------")
    print("Output (Xp) : \n", output_temp[0])


    Xp = output_temp[0]
    DXp = 2*(Xp-X) # loss/error
    print("------- ########## ########## ----------")
    print("error/loss (DXp) : \n", DXp)
    loss = DXp

    # gradient of loss w.r.t y 
    output_dy = torch.zeros((batch_size, c_in, n, n), dtype=torch.float).cuda()
    M_dy = torch.zeros((batch_size, c_in, n, n), dtype=torch.float).cuda()
    dy = inv_conv_dy.dy(loss, Wp, M_dy, output_dy)            # dL/dy. JIT
    # dy = inv_conv_with_bp.dy(loss, Wp, M_dy, output_dy)         # dL/dy. AOT
    print("grad w.r.t y[] (): \n", dy[0]) # dL/dy
    print("-------########## ########## ----------")

    print("dy - DXp : ", (dy[0] - DXp))
    print("------- ########### ######### ----------")

    # gradient of loss w.r.t w
    M_dw = torch.zeros((batch_size, c_in, k, k, n, n), dtype=torch.float).cuda()
    output_dw = torch.zeros((batch_size, c_in, k, k), dtype=torch.float).cuda()
    dw , M_dw_out= inv_conv_dw.dw(Y[0], Wp, loss, M_dw, output_dw) # dL/dw # JIT
    # dw = inv_conv_with_bp.dw(Y[0], Wp, loss, M_dw, output_dw) # dL/dw # AOT
    print("grads dw : \n", dw[0])
    # print("M_dw_out : \n", M_dw_out[0])
    # dw = [[1260., -936.],
        # [786., 0.]]

    # c = 4 * c_in, level2
    # output2 = torch.zeros((batch_size, c_in, n, n), dtype=torch.float).cuda()
    # y = cinc_level2.inverse(X, W, output2)
    # print("Output (y): \n", y[0])
    # print(" Error level2 (y - Y)----", (y[0] - Y[0]).sum())

    # level1 c == 1, 2, 3....
    # output3 = torch.zeros((batch_size, c_in, n, n), dtype=torch.float).cuda()
    # y_ = cinc_level1.inverse(X, W, output3)
    # print("Output (y): \n", y_[0])
    # print(" Error level1 (y_ - Y)----", (y_[0] - Y[0]).sum())

    # t = time.process_time() - t

    # print(f"Error (x-x`) : {error}")
    # print(f"Time : {t} s")

batch_size  = 1
c_in        = 4 # input channels, should be multiple of 4, output channels
n           = 5 # input size (n, n)
k           = 4 # kernel size (k, k)

X = np.random.random((1, c_in, n, n)) # 1, 4, 2, 2``
K = np.random.random((c_in, c_in, k, k))
Wp = np.random.random((c_in, c_in, k, k))
print("X : \n", X.shape)
print("K shape : \n", K.shape)
print("Wp shape : \n", Wp.shape)
# print("K : \n", K)
# K[:, :, -1, -1] = 1.0
test_inverse(
    input = X, # np.random.random((2, 5, 5)) - 0.5,
    kernel= K, #kernel
    Wp = Wp,
    batch_size=batch_size, c_in=c_in, n=n, k=k
    )

# X = np.array([[[[4., 2.],
#                 [3., 1.]],
#                 [[4., 2.],
#                 [3., 1.]]],
#                 [[[4., 2.],
#                 [3., 1.]],
#                 [[4., 2.],
#                 [3., 1.]]]
#                 ])
# X =  np.array([[
#     [[2, 3],
#      [8, 9]],
     
#     [[2, 3],
#      [8, 9]],

#     [[2, 3],
#      [8, 9]],

#     [[2, 3],
#      [8, 9]]
#      ]])
# X =  np.array([[
#     [[2, -3],
#      [4, 5]],
     
#     [[2, -3],
#      [4, 5]],

#     [[2, -3],
#      [4, 5]],

#     [[2, -3],
#      [4, 5]]
#      ]])
# X =  np.array([[
#     [[2, -3],
#      [4, -5]],
     
#     [[2, -3],
#      [4, -5]],

#     [[2, -3],
#      [4, -5]],

#     [[2, -3],
#      [4, -5]],
#      ]])              
# K  = np.array([
#         [[[4., 2.],
#          [3., 1.]],
#         [[4., 2.],
#          [3., 1.]],
#         [[4., 2.],
#          [3., 1.]],
#         [[4., 2.],
#          [3., 1.]]],
#     [
#         [[4., 2.],
#          [3., 1.]],
#         [[4., 2.],
#          [3., 1.]],
#         [[4., 2.],
#          [3., 1.]],
#         [[4., 2.],
#          [3., 1.]]],
#     [
#         [[4., 2.],
#          [3., 1.]],
#         [[4., 2.],
#          [3., 1.]],
#         [[4., 2.],
#          [3., 1.]],
#         [[4., 2.],
#          [3., 1.]]],
#     [
#         [[4., 2.],
#         [3., 1.]],
#         [[4., 2.],
#          [3., 1.]],
#         [[4., 2.],
#          [3., 1.]],
#         [[4., 2.],
#          [3., 1.]]]]
#     )
# Wp  = np.array([
#         [[[1., 1.],
#          [1., 1.]],
#         [[1., 1.],
#          [1., 1.]],
#         [[1., 1.],
#          [1., 1.]],
#         [[1., 1.],
#          [1., 1.]], ],

#         [[[1., 1.],
#          [1., 1.]],
#         [[1., 1.],
#          [1., 1.]],
#         [[1., 1.],
#          [1., 1.]],
#         [[1., 1.],
#          [1., 1.]], ],

#         [[[1., 1.],
#          [1., 1.]],
#         [[1., 1.],
#          [1., 1.]],
#         [[1., 1.],
#          [1., 1.]],
#         [[1., 1.],
#          [1., 1.]], ],

#         [[[1., 1.],
#          [1., 1.]],
#         [[1., 1.],
#          [1., 1.]],
#         [[1., 1.],
#          [1., 1.]],
#         [[1., 1.],
#          [1., 1.]], ],
#     ]
#     )
# Wp  = np.array([
#         [[[2., 2.],
#          [2., 1.]],
#         [[2., 2.],
#          [2., 1.]],
#         [[2., 2.],
#          [2., 1.]],
#         [[2., 2.],
#          [2., 1.]], ],

#         [[[2., 2.],
#          [2., 1.]],
#         [[2., 2.],
#          [2., 1.]],
#         [[2., 2.],
#          [2., 1.]],
#         [[2., 2.],
#          [2., 1.]], ],

#         [[[2., 2.],
#          [2., 1.]],
#         [[2., 2.],
#          [2., 1.]],
#         [[2., 2.],
#          [2., 1.]],
#         [[2., 2.],
#          [2., 1.]], ],

#         [[[2., 2.],
#          [2., 1.]],
#         [[2., 2.],
#          [2., 1.]],
#         [[2., 2.],
#          [2., 1.]],
#         [[2., 2.],
#          [2., 1]], ],
#     ]
#     )
# Wp  = np.array([
#         [[[3., 3.],
#          [3., 1.]],
#         [[3., 3.],
#          [3., 1.]],
#         [[3., 3.],
#          [3., 1.]],
#         [[3., 3.],
#          [3., 1.]], ],

#         [[[3., 3.],
#          [3., 1.]],
#         [[3., 3.],
#          [3., 1.]],
#         [[3., 3.],
#          [3., 1.]],
#         [[3., 3.],
#          [3., 1.]], ],

#         [[[3., 3.],
#          [3., 1.]],
#         [[3., 3.],
#          [3., 1.]],
#         [[3., 3.],
#          [3., 1.]],
#         [[3., 3.],
#          [3., 1.]], ],

#         [[[3., 3.],
#          [3., 1.]],
#         [[3., 3.],
#          [3., 1.]],
#         [[3., 3.],
#          [3., 1.]],
#         [[3., 3.],
#          [3., 1.]], ],
#     ]
#     ) 
# Wp  = np.array([
#         [[[7,8],
# [6,1]],
#         [[7,8],
# [6,1]],
#         [[7,8],
# [6,1]],
#         [[7,8],
# [6,1]], ],

#         [[[7,8],
# [6,1]],
#         [[7,8],
# [6,1]],
#         [[7,8],
# [6,1]],
#         [[7,8],
# [6,1]], ],

#         [[[7,8],
# [6,1]],
#         [[7,8],
# [6,1]],
#         [[7,8],
# [6,1]],
#         [[7,8],
# [6,1]], ],

#         [[[7,8],
# [6,1]],
#         [[7,8],
# [6,1]],
#         [[7,8],
# [6,1]],
#         [[7,8],
# [6,1]], ],
#     ]
#     ) 
# 4, 2, 2, 2
# X = np.ones((batch_size, c_in, n, n)) # 1, 4, 2, 2``
# X = np.random.random((batch_size, c_in, n, n)) # 1, 4, 2, 2``
# K = np.ones((c_in, c_in, k, k))
# K = np.random.random((c_in, c_in, k, k))
# K = np.ones((c_in, c_in, k, k))
# Wp = np.random.random((c_in, c_in, k, k))
# Wp = np.ones((c_in, c_in, k, k))
# print("X : \n", X.shape)
# print("K shape : \n", K.shape)
# print("Wp shape : \n", Wp.shape)
# # print("K : \n", K)
# # K[:, :, -1, -1] = 1.0
# test_inverse(
#     input = X, # np.random.random((2, 5, 5)) - 0.5,
#     kernel= K, #kernel
#     Wp = Wp,
#     batch_size=1, c_in=4, n=n, k=k
#     )

# if kernel is not normalized, the values in the output becomes very large 
# resulting in large numerical errors
# kernel = (np.random.random((5, 5)) - 0.5)/25.0
# kernel[-1, -1] = 1.0 # NOTE: its essential that this value is not too small. Othewise it results in large numerical errors
# kernel = np.float32(kernel)

# X = np.array([
#     [[2, 3],
#     [8, 9],
#     [2, 3],
#     [8, 9]],
#     [[2, 3],
#     [8, 9],
#     [2, 3],
#     [8, 9]],
#     [[2, 3],
#     [8, 9],
#     [2, 3],
#     [8, 9]],
#     [[2, 3],
#     [8, 9],
#     [2, 3],
#     [8, 9]]]).reshape(1, 4, 2, 2)
# X = np.array([
#     [1, 2, 3],
#     [4,5,6],
#     [7,8,9]]).reshape( 3, 3)
# X = np.ones((1, 1, 2, 2))
# X = np.array([[[[2., 3.],
#           [8., 9.]]]])

# K = np.array([
#     [1, 1],
#     [1, 1] ])

# K = np.ones((1, 1, 2, 2))
# K = np.array([[[[4., 2.],
#                 [3., 1.]],
#                 [[4., 2.],
#                 [3., 1.]]],
#                 [[[4., 2.],
#                 [3., 1.]],
#                 [[4., 2.],
#                 [3., 1.]]],
#                 [[[4., 2.],
#                 [3., 1.]],
#                 [[4., 2.],
#                 [3., 1.]]],
#                 [[[4., 2.],
#                 [3., 1.]],
#                 [[4., 2.],
#                 [3., 1.]]]])

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

# input  = (B, C_in, H, W)
# K = (c_out, C_in, K, K)
# output = (B, c_out, H, W)
# %%
