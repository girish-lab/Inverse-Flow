import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

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
from inf.datasets.cifar10 import load_data
from inf.layers.activations import SplineActivation, SmoothLeakyRelu
activations = {
    'SLR':lambda size: SmoothLeakyRelu(alpha=0.3),
    'Spline': lambda size: SplineActivation(size, n_bins=10, tail_bound=20, individual_weights=False),
}
def create_model(num_blocks=3, block_size=32, sym_recon_grad=False, 
                 activation='Spline',
                 actnorm=True, split_prior=True, recon_loss_weight=1000.0):
    current_size = (3, 32, 32)

    if activation in ['Spline', 'SLR', 'BSpline']:
        act = activations[activation]

    alpha = 1e-6
    layers = [
        Dequantization(UniformDistribution(size=(3, 32, 32))),
        Normalization(translation=0, scale=256),
        Normalization(translation=-alpha, scale=1 / (1 - 2 * alpha)),
        LogitTransform(),
    ]

    for l in range(num_blocks):
        layers.append(Squeeze())
        current_size = (current_size[0]*4, current_size[1]//2, current_size[2]//2)

        for k in range(block_size):
            if actnorm:
                layers.append(ActNorm(current_size[0]))
            
            layers.append(SelfNormConv(current_size[0], current_size[0], (1, 1), 
                                       bias=True, stride=1, padding=0,
                                       sym_recon_grad=sym_recon_grad, 
                                       recon_loss_weight=recon_loss_weight))
            if activation in ['Spline', 'SLR', 'BSpline']:
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
    config = {
        'name': '2L-4K_Glow_SNF_C',
        'eval_epochs': 1,
        'sample_epochs': 1,
        'log_interval': 100,
        'lr': 1e-3,
        'num_blocks': 2,
        'block_size': 4,
        'batch_size': 100,
        'n_samples': 100,
        'modified_grad': True,
        'add_recon_grad': True,
        'sym_recon_grad': True,
        'actnorm': True,
        'split_prior': True,
        'activation': 'None',
        'recon_loss_weight': 1000.0,
        'sample_true_inv': True,
        'plot_recon': True,
        'vis_epochs': 10_000,

        'grad_clip_norm': False,
        'grad_clip': 0.001,
        'grad_clip_value': 0.001,

        # 'grad_clip_norm': 1.1,
        'warmup_epochs': 2,
	    'step_epochs': 1,
        'step_gamma': 1.0,
        'wandb_project': 'snf_cifar10',
        'wandb_entity': 'carlobob031',
        'wandb': True,
        'log_timing': True,
        

    }

    train_loader, val_loader, test_loader = load_data(data_aug=True, batch_size=config['batch_size'])

    model = create_model(num_blocks=config['num_blocks'],
                         block_size=config['block_size'], 
                         sym_recon_grad=config['sym_recon_grad'],
                        activation=config['activation'],
                         actnorm=config['actnorm'],
                         split_prior=config['split_prior'],
                         recon_loss_weight=config['recon_loss_weight']).to('cuda')
    print(model)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print('Total_params Inv_flow , :', pytorch_total_params/1000000, 'Million')
    print('config:', config)

    optimizer = optim.Adam(model.parameters(), lr=config['lr'], betas=(0.9, 0.999))
    scheduler = StepLR(optimizer, step_size=config['step_epochs'], gamma=config['step_gamma'])

    experiment = Experiment(model, train_loader, val_loader, test_loader,
                            optimizer, scheduler, **config)
    experiment.run()
