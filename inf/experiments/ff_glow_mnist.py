import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR, ExponentialLR

from inf.layers import Dequantization, Normalization
from inf.layers.distributions.uniform import UniformDistribution
from inf.layers.splitprior import SplitPrior
from inf.layers.flowsequential import FlowSequential
from inf.layers.selfnorm import SelfNormConv
from inf.layers.actnorm import ActNorm
from inf.layers.squeeze import Squeeze
from inf.layers.transforms import LogitTransform
from inf.layers.coupling import Coupling
from inf.train.losses import NegativeGaussianLoss
from inf.train.experiment import Experiment
from inf.datasets.mnist import load_data
from inf.layers.fincflow import Finc_FlowUnit
from inf.layers.inv_flow import Inv_FlowUnit
from inf.layers.activations import SmoothLeakyRelu, SplineActivation, Identity

activations = {
    'SLR':lambda size: SmoothLeakyRelu(alpha=0.3),
    'Spline': lambda size: SplineActivation(size, n_bins = 7, tail_bound=20, individual_weights=True),
}

def create_model(num_blocks=2, block_size=16, sym_recon_grad=False, 
                 actnorm=False, activation='Spline', split_prior=False, recon_loss_weight=100.0):
    alpha = 1e-6
    if activation in ['Spline', 'SLR']:
        act = activations[activation]
    layers = [
        Dequantization(UniformDistribution(size=(1, 28, 28))),
        Normalization(translation=0, scale=256),
        Normalization(translation=-alpha, scale=1 / (1 - 2 * alpha)),
        LogitTransform(),
    ]

    current_size = (1, 28, 28)

    for l in range(num_blocks):
        layers.append(Squeeze())
        current_size = (current_size[0]*4, current_size[1]//2, current_size[2]//2)

        for k in range(block_size):
            if actnorm:
                layers.append(ActNorm(current_size[0]))
            
            # layers.append(SelfNormConv(current_size[0], current_size[0], (1, 1), 
            #                            bias=True, stride=1, padding=0,
            #                            sym_recon_grad=sym_recon_grad, 
            #                            recon_loss_weight=recon_loss_weight))
            # FincFlow unit
            layers.append(Finc_FlowUnit(current_size[0], current_size[0], (3, 3)))
            if activation in ['Spline', 'SLR']:
                if not (l == num_blocks - 1 and k == block_size - 1):
                    # Dont place activation at end of last block
                    layers.append(act(current_size))

            layers.append(Coupling(current_size))

        if split_prior and l < num_blocks - 1:
            layers.append(SplitPrior(current_size, NegativeGaussianLoss))
            current_size = (current_size[0] // 2, current_size[1], current_size[2])

    return FlowSequential(NegativeGaussianLoss(size=current_size), 
                         *layers)


def main():
    # set cuda device 1 to train the model
    # torch.cuda.set_device(1)
    config = {
        'name': '2L-16K_FF_Glow_M',
        'eval_epochs': 1,
        'sample_epochs': 1,
        'log_interval': 100,
        'lr': 1e-5,
        'num_blocks': 2,
        'block_size': 16,
        'batch_size': 100,
        'modified_grad': True,
        'add_recon_grad': True,
        'sym_recon_grad': True,
        'actnorm': True, # True,
        'split_prior': True,
        'activation': 'Spline',
        'recon_loss_weight': 10.0,
        'sample_true_inv': True,
        'plot_recon': True,
        'vis_epochs': 1,
        'grad_clip_norm': True,
        'grad_clip': 00.01,
        'grad_clip_value': 0.02,


        'wandb_project': 'ff_mnist',
        'wandb_entity': 'carlobob031',

        'log_timing': True,
        'wandb': True,
    }

    train_loader, val_loader, test_loader = load_data(data_aug=True, batch_size=config['batch_size'])

    model = create_model(num_blocks=config['num_blocks'],
                         block_size=config['block_size'], 
                         sym_recon_grad=config['sym_recon_grad'],
                         actnorm=config['actnorm'],
                         activation=config['activation'],
                         split_prior=config['split_prior'],
                         recon_loss_weight=config['recon_loss_weight']).to('cuda')
    print(model)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print('Total_params Inv_flow , :', pytorch_total_params/1e6, 'M')
    print('config:', config)

    optimizer = optim.Adam(model.parameters(), lr=config['lr'], betas=(0.9, 0.999))
    # scheduler = StepLR(optimizer, step_size=1, gamma=1.0)
    scheduler = ExponentialLR(optimizer, gamma=0.9997, last_epoch=-1, verbose=True)

    experiment = Experiment(model, train_loader, val_loader, test_loader,
                            optimizer, scheduler, **config)

    experiment.run()