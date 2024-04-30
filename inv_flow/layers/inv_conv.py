from functools import lru_cache
from itertools import product

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import wandb
from torch.utils.cpp_extension import load
CUDA_LAUNCH_BLOCKING=1

import math
import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

# import inv_conv # cuda load module for inv_conv 
import inv_conv_with_bp # cuda load module for inv_conv 

from snf.layers.flowlayer import FlowLayer, mark_expensive, \
    ModifiedGradFlowLayer
from snf.utils.toeplitz import get_sparse_toeplitz, get_toeplitz_idxs
# from snf.utils.convbackward import conv2d_backward

# @lru_cache(maxsize=128)
# def _compute_weight_multiple(wshape, output, x, padding, stride, dilation, 
#                                  groups, benchmark, deterministic):
#     batch_multiple =  conv2d_backward.backward_weight(wshape, 
#                                            torch.ones_like(output),
#                                            torch.ones_like(x),
#                                            padding, stride, dilation,
#                                            groups, benchmark, deterministic)
#     return batch_multiple / len(x)


def flip_kernel(W):
    return torch.flip(W, (2,3)).permute(1,0,2,3).clone()


class inv_conv_(autograd.Function):

    @staticmethod
    def forward(ctx, x, W): # inverse pass for the Inv-Flow
        # z = F.conv2d(x, W)
        # print('inv_conv_, fwd', x.shape, W.shape)
        output_x = torch.zeros_like(x).to(x.device)
        # x = x.resize(1, x.shape[2], x.shape[3])
        # W = W.resize(W.shape[2], W.shape[3])
        # z = inv_conv.inverse(x, W, output_x) # y = output
        z = inv_conv_with_bp.inverse(x, W, output_x) # y = output
        ctx.save_for_backward(x, W, z[0])
        # print('inv_conv_, fwd', f'x.shape: {x.shape},', 
        #       f'W.shape: {W.shape},', f'z[0].shape: {z[0].shape}')

        return z[0]

    @staticmethod
    def backward(ctx, output_grad):
        input, kernel, _ = ctx.saved_tensors
        # input = input.resize(1, input.shape[1], input.shape[2])
        # kernel = kernel.resize(kernel.shape[2], kernel.shape[3])
        b, c, n,m = input.shape
        c_out,c_in, k_h, k_w = kernel.shape
        # M_dk = torch.zeros((b, c, k_h,k_w, n,m), dtype=output_grad.dtype()).to(output_grad.device)
        M_dk = torch.zeros((b, c, k_h,k_w, n,m)).to(output_grad.device) # dtype=torch.float64)
        M_dy = torch.zeros_like(input).to(input.device) # torch.float64
        
        output_dy = torch.zeros_like(input).to(input.device)
        
        # input_grad = inv_conv.dy(output_grad, kernel, M_dy, output_dy)
        input_grad = inv_conv_with_bp.dy(output_grad, kernel, M_dy, output_dy)


        output_dk = torch.zeros_like(kernel).to(kernel.device)
        # grad_k = inv_conv.dw(input, kernel, output_grad, M_dk, output_dk)
        grad_k = inv_conv_with_bp.dw(input, kernel, output_grad, M_dk, output_dk)

        return input_grad[0] , grad_k[0]
    def clip_gradients(self):
        for param in self.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-self.grad_clip_value, self.grad_clip_value)


def inv_conv_4d(x, W):
    f = inv_conv_()
    return f.apply(x, W)


class inv_flow(FlowLayer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 sym_recon_grad=False,
                 only_R_recon=False,
                 recon_loss_weight=1.0,
                 recon_loss_lr=0.0,
                 recon_alpha=0.9):

        super().__init__()
        self.kernel_size = _pair(kernel_size)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.sym_recon_grad = sym_recon_grad
        self.only_R_recon = only_R_recon
        self.recon_loss_weight = recon_loss_weight
        self.recon_loss_lr = recon_loss_lr
        self.recon_loss_ema = None
        self.alpha = recon_alpha
        self.reset_parameters()

    def reset_parameters(self):
        self.logabsdet_dirty = True
        self.T_idxs, self.f_idxs = None, None

        w_shape = (self.out_channels, self.in_channels, *self.kernel_size)
        w_eye = nn.init.dirac_(torch.empty(w_shape))
        w_noise = nn.init.xavier_normal_(torch.empty(w_shape), gain=0.01)

        if self.kernel_size[0] == 1 and self.kernel_size[1] == 1:
            # If 1x1 convolution, use a random orthogonal matrix
            w_np = np.random.randn(self.out_channels, self.in_channels)
            w_init = torch.tensor(np.linalg.qr(w_np)[0]).to(torch.float).view(w_shape)
        else:
            # Otherweise init with identity + noise
            w_init = w_eye + w_noise

        self.weight_fwd = nn.Parameter(w_init)
        # self.weight_inv = nn.Parameter(flip_kernel(w_init))

        # b_small = torch.nn.init.normal_(torch.empty(self.out_channels), 
        #                                 std=w_noise.std())
        # self.bias_fwd = nn.Parameter(b_small) if self.use_bias else None

    def forward(self, input, context=None, compute_expensive=False):
        if self.training:
            self.logabsdet_dirty = True
        self.input = input

        if compute_expensive:
            self.output = F.conv2d(input, self.weight_fwd, self.bias_fwd, 
                self.stride, self.padding, self.dilation, self.groups)
            ldj = self.logdet(input, context)
        else:
            self.output = inv_conv_4d(input, self.weight_fwd)
            ldj = 0.
        return self.output, ldj
    def reset_gradients(self):
        '''
            Reset gradients for the weight matrix
            verifyt this function 
        '''
        self.mask = self.get_mask()
        self.weight_fwd.grad = self.weight_fwd.grad * self.mask.to(self.weight_fwd.grad.device)

        # self.conv.weight.grad = self.conv.weight.grad * self.mask.to(self.conv.weight.grad.device)
    def get_mask(self):
        mask = torch.ones_like(self.weight_fwd)
        for c_out in range(self.weight_fwd.data.shape[0]):
            mask[c_out, c_out, -1, -1] = 0.0
            mask[c_out, c_out+1:, -1, -1] = 0.0
        
        # if self.order == 'TR':
        #     mask = torch.flip(mask, [3])
        
        # elif self.order == 'BL':
        #     mask = torch.flip(mask, [2])
        
        # elif self.order == 'BR':
        #     mask = torch.flip(mask, [2, 3])
        
        return mask
    def reverse(self, input, context=None, compute_expensive=False): # fwd pass for the Inv-Flow, to generate samples
        # if self.bias_fwd is not None:
        #     input = input - self.bias_fwd.view(1, -1, 1, 1)

        if compute_expensive:
            # Use actual inverse
            T_sparse = self.sparse_toeplitz(input, context)
            rev = torch.matmul(T_sparse.to_dense().inverse().to(input.device),
                               input.flatten(start_dim=1).unsqueeze(-1))
            rev = rev.view(input.shape)
        else:
            ## SNF, Use inverse weights, SNF
            # rev = F.conv2d(input, self.weight_inv, None, self.stride, 
            #                 self.padding, self.dilation, self.groups)
            # inv_conv , Use actual fwd
            output_rev = torch.zeros_like(input).to(input.device)
            # rev = inv_conv.forward(input, self.kernel, output_rev) # Y = output', fwd pass for the Inv-Flow, 
            rev = inv_conv_with_bp.forward(input, self.weight_fwd, output_rev) # Y = output', fwd pass for the Inv
        return rev[0]

    def add_recon_grad(self, recon_loss_weight_update=None):
        # Compute ||x - RWx||^2
        x = self.input.detach() # have to compute z again w/ detached x
        z = F.conv2d(x, self.weight_fwd, None, 
                self.stride, self.padding, self.dilation, self.groups)
        if self.only_R_recon:
            z = z.detach()
        x_hat = F.conv2d(z, self.weight_inv, None, self.stride, 
                         self.padding, self.dilation, self.groups)
        recon_loss = (x - x_hat).pow(2).flatten(start_dim=1).sum(-1)

        # Compute ||z - WRz||^2
        if self.sym_recon_grad:
            zsym = z.detach()
            xsym = F.conv2d(z, self.weight_inv, None, 
                    self.stride, self.padding, self.dilation, self.groups)
            z_hat_sym = F.conv2d(xsym, self.weight_fwd, None, self.stride, 
                             self.padding, self.dilation, self.groups)
            recon_loss_sym = (zsym - z_hat_sym).pow(2).flatten(start_dim=1).sum(-1)
            recon_loss = (recon_loss + recon_loss_sym) / 2.0

        if recon_loss_weight_update is not None:
            self.recon_loss_weight = recon_loss_weight_update

        # Set NaN values to 0 for stability
        recon_loss[recon_loss != recon_loss] = 0.0
        recon_loss_weighted = self.recon_loss_weight * recon_loss.mean()

        # Using .backward call to add recon gradient
        recon_loss_weighted.backward()

        # If using GECO (i.e. recon_loss_lr > 0.0) update recon_loss_weight from moving average
        if self.recon_loss_lr > 0.0:
            with torch.no_grad():
                if self.recon_loss_ema is None:
                    self.recon_loss_ema = recon_loss.mean()
                else:
                    self.recon_loss_ema = self.alpha * self.recon_loss_ema + (1 - self.alpha) * recon_loss.mean()
                C_t = recon_loss.mean() + (self.recon_loss_ema - recon_loss.mean()).detach()
                delta_rlw = torch.exp(self.recon_loss_lr * C_t)
                self.recon_loss_weight = self.recon_loss_weight * delta_rlw

        return recon_loss_weighted

    def sparse_toeplitz(self, input, context=None):
        if self.T_idxs is None or self.f_idxs is None:
            self.T_idxs, self.f_idxs = get_toeplitz_idxs(
                self.weight_fwd.shape, input.shape[1:], self.stride, self.padding)

        T_sparse = get_sparse_toeplitz(self.weight_fwd, input.shape[1:],
                                       self.T_idxs, self.f_idxs)
        return T_sparse

    @mark_expensive
    def logdet(self, input, context=None):
        if self.logabsdet_dirty:
            T_sparse = self.sparse_toeplitz(input, context)
            self.logabsdet = torch.slogdet(T_sparse.to_dense())[1].to(input.device)
            self.logabsdet_dirty = False
        return self.logabsdet.view(1).expand(len(input))

    def plot_filters(self, layer_idx, max_s=10):
        name = 'SNF_L{}_{}'

        weights = {'fwd': self.weight_fwd.detach().cpu().numpy(),
                   'inv': self.weight_inv.detach().cpu().numpy()}       
        
        c_out, c_in, h, w = weights['fwd'].shape
        s = min(max_s, int(np.ceil(np.sqrt(c_out))))
        
        empy_weight = np.zeros_like(weights['fwd'][0,0,:,:])

        for direction in ['fwd', 'inv']:
            for c in range(c_in):
                f, axarr = plt.subplots(s,s)
                f.set_size_inches(7, 7)
                for s_h in range(s):
                    for s_w in range(s):
                        w_idx = s_h * s + s_w
                        if w_idx < c_out:
                            img = axarr[s_h, s_w].imshow(weights[direction][w_idx, c, :, :], cmap='PuBu_r')
                            axarr[s_h, s_w].get_xaxis().set_visible(False)
                            axarr[s_h, s_w].get_yaxis().set_visible(False)
                            f.colorbar(img, ax=axarr[s_h, s_w])
                        else:
                            img = axarr[s_h, s_w].imshow(empy_weight, cmap='PuBu_r')
                            axarr[s_h, s_w].get_xaxis().set_visible(False)
                            axarr[s_h, s_w].get_yaxis().set_visible(False)
                            f.colorbar(img, ax=axarr[s_h, s_w])
                # f.colorbar(img, ax=axarr.ravel().tolist())
            wandb.log({name.format(layer_idx, direction): wandb.Image(plt)}, commit=True)
            plt.close('all')