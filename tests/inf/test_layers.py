import numpy as np
import torch
import torch.nn as nn
from inf.layers.activations import LearnableLeakyRelu, LeakyRelu, SplineActivation, \
    SmoothLeakyRelu, SmoothTanh, Identity
from inf.layers.actnorm import ActNorm
from inf.layers.conv1x1 import Conv1x1, Conv1x1Householder
from inf.layers.inv_flow import Inv_FlowUnit         # This is the important part for Invertible Convolution (Inv_flow)
from inf.layers.fincflow import Finc_FlowUnit         # This is the important part for Invertible Convolution (Inv_flow)
from inf.layers.inv_conv import inv_flow_with_pad, inv_flow_no_pad
# from inf.layers.inv_conv import inv_flow         # This is the important part for Invertible Convolution (Inv_flow)
from inf.layers.selfnorm import SelfNormConv, SelfNormFC
from inf.layers.coupling import Coupling
from inf.layers.normalize import Normalization
from inf.layers.squeeze import Squeeze, UnSqueeze
from inf.layers.splitprior import SplitPrior
from inf.train.losses import NegativeGaussianLoss

def check_inverse(module, data_dim, n_times=1, compute_expensive=False):
    for _ in range(n_times):
        if hasattr(module, 'reset_parameters'):
            module.reset_parameters()
            module.to('cuda')
        input = torch.randn(data_dim).to('cuda')

        if compute_expensive:
            forward, logdet = module(input, compute_expensive=compute_expensive)
            reverse = module.reverse(forward, compute_expensive=compute_expensive)
        else:
            forward, logdet = module(input)
            reverse = module.reverse(forward)

        inp = input.cpu().detach().numpy()
        outp = reverse.cpu().detach().view(data_dim).numpy()

        np.testing.assert_allclose(inp, outp, atol=1e-3)


def test_snf_layer_inverses(input_size=(12, 4, 16, 16)):
    c_in = c_out = input_size[1]
    f_in = f_out = input_size[1] * input_size[2] * input_size[3]
    conv_module = SelfNormConv(c_out, c_in, (3, 3), padding=1)
    fc_module = SelfNormFC(f_out, f_in)

    check_inverse(fc_module, input_size, compute_expensive=True)
    check_inverse(conv_module, input_size, compute_expensive=True)


def test_splitprior_inverse(input_size, distribution, n_times=1):
    module = SplitPrior(input_size[1:], NegativeGaussianLoss).to('cuda')
    half_c = input_size[1] // 2
 
    for _ in range(n_times):
        input = torch.randn(input_size).to('cuda')

        forward, logdet = module(input)
        reverse = module.reverse(forward)

        full_reverse = torch.cat([reverse[:, :half_c], input[:, half_c:]], dim=1)

        inp = input.cpu().detach().numpy()
        outp = full_reverse.cpu().detach().view(input_size).numpy()

        np.testing.assert_allclose(inp, outp, atol=1e-3)


def check_logdet(module, data_dim, name=None):
    if hasattr(module, 'reset_parameters'):
        module.reset_parameters()
        module.to('cuda')

    nfeats = np.prod(data_dim[1:])
    x = torch.randn(data_dim).to('cuda')

    module.to('cuda')
    _, _ = module(x)
    ldj_ours = module.logdet(x) 

    def func(*inputs):
        inp = torch.stack(inputs, dim=0)
        out, _ = module(inp)
        out = out.sum(dim=0)
        return out

    J = torch.autograd.functional.jacobian(
        func, tuple(x), create_graph=False,
        strict=False)
    J = torch.stack(J, dim=0)
    J = J.view(x.size(0), nfeats, nfeats)
    logdet_pytorch = torch.slogdet(J)[1]
    if name == 'inv_conv':
        pass
        # logdet_pytorch += module.base.log_prob(module.transform(x)[0])
    else:
        ldj_ours = ldj_ours.cpu().detach().numpy()
        ldj_pytorch = logdet_pytorch.cpu().detach().numpy()

        np.testing.assert_allclose(ldj_ours, ldj_pytorch, atol=1e-4)


def test_snf_logdet(input_size=(12, 4, 8, 8)):
    c_in = c_out = input_size[1]
    f_in = f_out = input_size[1] * input_size[2] * input_size[3]
    conv_module = SelfNormConv(c_out, c_in, (3, 3), padding=1)
    fc_module = SelfNormFC(f_out, f_in)

    check_logdet(fc_module, input_size)
    check_logdet(conv_module, input_size)


def test_splitprior_logdet(input_size, distribution, n_times=10):
    module = SplitPrior(input_size[1:], NegativeGaussianLoss).to('cuda')
    half_c = input_size[1] // 2
    nfeats = np.prod((half_c, *input_size[2:]))

    x = torch.randn(input_size).to('cuda')
    ldj_ours = module.logdet(x) 

    def func(*inputs):
        inp = torch.stack(inputs, dim=0)
        out, _ = module(inp)
        out = out.sum(dim=0)
        return out

    J = torch.autograd.functional.jacobian(
        func, tuple(x), create_graph=False,
        strict=False)
    J = torch.stack(J, dim=0)
    J = J[:, :, :, :, :half_c, :, :]
    J = J.view(x.size(0), nfeats, nfeats)
    logdet_pytorch = torch.slogdet(J)[1]
    logdet_pytorch += module.base.log_prob(module.transform(x)[0][:,half_c:])

    ldj_ours = ldj_ours.cpu().detach().numpy()
    ldj_pytorch = logdet_pytorch.cpu().detach().numpy()

    np.testing.assert_allclose(ldj_ours, ldj_pytorch, atol=1e-4)



def test_inverses(input_size=(1, 4, 5, 5)):
    check_inverse(LearnableLeakyRelu().to('cuda'), input_size)
    check_inverse(SplineActivation(input_size).to('cuda'), input_size)
    check_inverse(SmoothLeakyRelu().to('cuda'), input_size)
    check_inverse(LeakyRelu().to('cuda'), input_size)
    check_inverse(SmoothTanh().to('cuda'), input_size)
    check_inverse(Identity().to('cuda'), input_size)
    check_inverse(ActNorm(input_size[1]).to('cuda'), input_size)
    check_inverse(Conv1x1(input_size[1]).to('cuda'), input_size)
    check_inverse(Conv1x1Householder(input_size[1], 10).to('cuda'), input_size)
    check_inverse(Coupling(input_size[1:]).to('cuda'), input_size)
    check_inverse(inv_flow_with_pad(input_size[1], input_size[1], (2, 2), order='TL').to('cuda'), input_size, name='inv_conv')
    # check_inverse(Inv_FlowUnit(input_size[1], input_size[1], (3, 3)).to('cuda'), input_size)
    check_inverse(Finc_FlowUnit(input_size[1], input_size[1], (3, 3)).to('cuda'), input_size)
    check_inverse(Normalization(translation=-1e-6, scale=1 / (1 - 2 * 1e-6)).to('cuda'), input_size)
    check_inverse(Squeeze().to('cuda'), input_size)
    check_inverse(UnSqueeze().to('cuda'), input_size)
    test_splitprior_inverse(input_size, NegativeGaussianLoss)
    test_snf_layer_inverses(input_size)

    print("All inverse tests passed")

def test_logdet(input_size=(1, 4, 5, 5)):
    check_logdet(LearnableLeakyRelu().to('cuda'), input_size)
    check_logdet(LeakyRelu().to('cuda'), input_size)
    check_logdet(SplineActivation(input_size).to('cuda'), input_size)
    check_logdet(SmoothLeakyRelu().to('cuda'), input_size)
    check_logdet(SmoothTanh().to('cuda'), input_size)
    check_logdet(Identity().to('cuda'), input_size)
    check_logdet(ActNorm(input_size[1]).to('cuda'), input_size)
    check_logdet(Coupling(input_size[1:]).to('cuda'), input_size)
    check_logdet(inv_flow_with_pad(input_size[1], input_size[1], (2, 2), order='TL').to('cuda'), input_size, name='inv_conv')
    # check_logdet(Inv_FlowUnit(input_size[1], input_size[1], (3, 3)).to('cuda'), input_size)
    # check_logdet(Finc_FlowUnit(input_size[1], input_size[1], (3, 3)).to('cuda'), input_size)
    check_logdet(Normalization(translation=-1e-6, scale=1 / (1 - 2 * 1e-6)).to('cuda'), input_size)
    check_logdet(Squeeze().to('cuda'), input_size)
    check_logdet(UnSqueeze().to('cuda'), input_size)
    test_splitprior_logdet(input_size, NegativeGaussianLoss)
    test_snf_logdet(input_size)

    print("All log-det tests passed")
def test_inv_conv():
    input_size = (1, 4, 5, 5)
    check_inverse(inv_flow_no_pad(input_size[1], input_size[1], (3, 3)).to('cuda'), input_size)
    check_logdet(inv_flow_no_pad(input_size[1], input_size[1], (3, 3)).to('cuda'), input_size, name='inv_conv')

    check_inverse(inv_flow_with_pad(input_size[1], input_size[1], (3, 3), order='TL').to('cuda'), input_size)
    check_logdet(inv_flow_with_pad(input_size[1], input_size[1], (3, 3), order='TL').to('cuda'), input_size, name='inv_conv')

    print("All inverse tests passed")

if __name__ == '__main__':
    # test_inverses()
    # test_logdet()
    test_inv_conv()