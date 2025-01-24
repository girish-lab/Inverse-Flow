__author__ = 'Sandeep Nagar, github.com/Naagar'

import torch
import torch.nn as nn

from torch.utils.cpp_extension import load
from inf.layers.inv_conv import inv_flow_with_pad, inv_flow_no_pad          # This is the important part for Invertible Convolution (Inv_flow)

# cinc_cuda_level2 = load(
#     'cinc_cuda_level2', ['snf/utils/fastflow_cuda_inverse/cinc_cuda_level2.cpp', 'snf/utils/fastflow_cuda_inverse/cinc_cuda_kernel_level2.cu'], verbose=True)


class Inv_FlowUnit(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Inv_FlowUnit, self).__init__()
        if isinstance(kernel_size, int) or len(kernel_size) == 1:
            kernel_size = (kernel_size, kernel_size)
        assert in_channels % 4 == 0, "Input channels have to be a multiple of 4" 

        # out_channels = in_channels // 4
        
        self.conv_tl = inv_flow_with_pad(out_channels, out_channels, kernel_size, order='TL')#, bias=True)
        self.conv_tr = inv_flow_with_pad(out_channels, out_channels, kernel_size, order='TR')#, bias=True)
        self.conv_bl = inv_flow_with_pad(out_channels, out_channels, kernel_size, order='BL')#, bias=True)
        self.conv_br = inv_flow_with_pad(out_channels, out_channels, kernel_size, order='BR')#, bias=True)
    
    def forward(self, x, context=None):

        logdet_accum = 0.0

        out_tl, logdet = self.conv_tl.forward(x)
        logdet_accum += logdet
        
        out_tr, logdet = self.conv_tr.forward(out_tl)
        logdet_accum += logdet
        
        out_bl, logdet = self.conv_bl.forward(out_tr)
        logdet_accum += logdet
        
        out_br, logdet = self.conv_br.forward(out_bl)
        logdet_accum += logdet
    
        return out_br, logdet_accum

    def reverse(self, x, context=None):
        out_br = self.conv_br.reverse(x)
        out_bl = self.conv_bl.reverse(out_br)
        out_tr = self.conv_tr.reverse(out_bl)
        
        out_tl = self.conv_tl.reverse(out_tr)

        return out_tl #self.reverse_level1(x)

    # def reverse_level1(self, x):

    #     out_tl = self.conv_tl.reverse(x)
    #     out_tr = self.conv_tr.reverse(out_tl)
    #     out_bl = self.conv_bl.reverse(out_tr)
    #     out_br = self.conv_br.reverse(out_bl)

    #     return out_br # out

    # def reverse_level2(self, x):
    #     k_tl = self.conv_tl.conv.weight.data
    #     k_tr = torch.flip(self.conv_tr.conv.weight.data, [3])
    #     k_bl = torch.flip(self.conv_bl.conv.weight.data, [2])
    #     k_br = torch.flip(self.conv_br.conv.weight.data, [2, 3])

    #     kernel = torch.cat([k_tl, k_tr, k_bl, k_br], dim=0)
    #     out_tl, out_tr, out_bl, out_br = torch.chunk(x, 4, dim=1) 
    #     out_tr = torch.flip(out_tr, [3])
    #     out_bl = torch.flip(out_bl, [2])
    #     out_br = torch.flip(out_br, [2, 3])

    #     x = torch.cat([out_tl, out_tr, out_bl, out_br], dim=1)
    #     y = torch.zeros_like(x).to(x.device)
    #     y = cinc_cuda_level2.inverse(x, kernel, y)[0]

    #     out_tl, out_tr, out_bl, out_br = torch.chunk(y, 4, dim=1) 
    #     out_tr = torch.flip(out_tr, [3])
    #     out_bl = torch.flip(out_bl, [2])
    #     out_br = torch.flip(out_br, [2, 3])

    #     y = torch.cat([out_tl, out_tr, out_bl, out_br], dim=1)
    #     return y

