import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR

from inf.layers import Dequantization, Normalization
from inf.layers.distributions.uniform import UniformDistribution
from inf.layers.flowsequential import FlowSequential
from inf.layers.selfnorm import SelfNormConv
from inf.layers.activations import SmoothLeakyRelu, SplineActivation, Identity
from inf.layers.squeeze import Squeeze
from inf.layers.transforms import LogitTransform
from inf.train.losses import NegativeGaussianLoss
from inf.train.experiment import Experiment
from inf.datasets.mnist import load_data
from inf.layers.inv_conv import inv_flow_with_pad, inv_flow_no_pad         # This is the important part for Invertible Convolution (Inv_flow)

activations = {
    'SLR':lambda size: SmoothLeakyRelu(alpha=0.3),
    'Spline': lambda size: SplineActivation(size, tail_bound=100, individual_weights=True),
}

def create_model(num_layers=9, sym_recon_grad=False, 
                 activation='Spline', recon_loss_weight=1.0,
                 num_blocks=3):
    block_size = int(num_layers / num_blocks)
    act = activations[activation]

    alpha = 1e-6
    layers = [
        Dequantization(UniformDistribution(size=(1, 28, 28))),
        Normalization(translation=0, scale=256),
        Normalization(translation=-alpha, scale=1 / (1 - 2 * alpha)),
        LogitTransform(),
    ]

    current_size = (1, 28, 28)

    for b in range(num_blocks):
        for l in range(block_size):
            # layers.append(SelfNormConv(current_size[0],current_size[0], (3, 3),
            #                            bias=True, stride=1, padding=1,
            #                            sym_recon_grad=sym_recon_grad, 
            #                            recon_loss_weight=recon_loss_weight))
            layers.append(inv_flow_no_pad(current_size[0], current_size[0], (3, 3)))
            
            if not (b == num_blocks - 1 and l == block_size - 1):
                # Dont place activation at end of last block
                layers.append(act(current_size))

        if not (b == num_blocks - 1):
            # Only squeeze between blocks
            layers.append(Squeeze())
            current_size = (current_size[0]*4, 
                            current_size[1]//2,
                            current_size[2]//2)

    return FlowSequential(NegativeGaussianLoss(size=current_size), 
                         *layers)


def main():
    config = {
        'name': '9L SNF Conv Exact Spline MNIST',
        'eval_epochs': 1,
        'sample_epochs': 1,
        'log_interval': 100,
        'lr': 1e-4,
        'num_layers': 9,
        'batch_size': 1000,
        'verbose': True,
        'modified_grad': False,
        'add_recon_grad': False,
        'sym_recon_grad': False,
        'activation': 'Spline',
        'recon_loss_weight': 0.0,
        'log_timing': True,
        'grad_clip_norm': False,
        'grad_clip': 0.01,
        'grad_clip_value': 0.01,
    }

    train_loader, val_loader, test_loader = load_data(batch_size=config['batch_size'])

    model = create_model(num_layers=config['num_layers'], 
                         sym_recon_grad=config['sym_recon_grad'],
                         activation=config['activation'],
                         recon_loss_weight=config['recon_loss_weight']).to('cuda')
    print(model)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print('Total_params Inv_flow , :', pytorch_total_params/1e6, 'M')
    print('config:', config)


    optimizer = optim.Adam(model.parameters(), lr=config['lr'], betas=(0.9, 0.999))
    scheduler = StepLR(optimizer, step_size=1, gamma=1.0)

    experiment = Experiment(model, train_loader, val_loader, test_loader,
                            optimizer, scheduler, **config)

    experiment.run()