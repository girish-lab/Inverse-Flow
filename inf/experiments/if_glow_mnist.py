import torch
import torch.nn as nn

from torch import optim
from torch.optim.lr_scheduler import StepLR, ExponentialLR, MultiStepLR 

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
from datetime import datetime
### 
from inf.layers.inv_flow import Inv_FlowUnit
from inf.layers.fincflow import Finc_FlowUnit
from inf.layers.inv_conv import inv_flow_with_pad, inv_flow_no_pad
from inf.layers.activations import SmoothLeakyRelu, SplineActivation, Identity
from inf.layers.splines.bspline import ConditionalBSplineTransformer

# activations = {
#     'SLR':lambda size: SmoothLeakyRelu(alpha=0.3),
#     'Spline': lambda size: SplineActivation(size, n_bins = 10, tail_bound=30, individual_weights=True),

# }

def create_model(snf=False, ff=False,inv_conv=False, inv_flow=True, inv_conv_no_pad=False,
                 if_kernel_size=3,
                 coupling_width = 512,
                 num_blocks=2, block_size=16,
                 tail_bound=30, n_bins=10,
                 sym_recon_grad=False, 
                 actnorm=False, activation='Spline', split_prior=False, recon_loss_weight=100.0):
    alpha = 1e-7
    activations = {
                'SLR':lambda size: SmoothLeakyRelu(alpha=0.3),
                'Spline': lambda size: SplineActivation(size, n_bins = n_bins, tail_bound=tail_bound, individual_weights=True),
                'BSpline': lambda size: ConditionalBSplineTransformer(size),
                }
    if activation in ['Spline', 'SLR', 'BSpline']:
        act = activations[activation]#(tail_bound=tail_bound, bins=bins)
    # act_if = activations['SLR']
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
            if snf:
                layers.append(SelfNormConv(current_size[0], current_size[0], (1, 1), 
                                           bias=True, stride=1, padding=0,
                                           sym_recon_grad=sym_recon_grad, 
                                           recon_loss_weight=recon_loss_weight))
            # FincFlow unit
            if ff:
                layers.append(Finc_FlowUnit(current_size[0], current_size[0], (3, 3)))
            if inv_conv:
                # layers.append(Inv_FlowUnit(current_size[0], current_size[0], (3, 3)))
                # Inv Flow with padding
                layers.append(inv_flow_with_pad(current_size[0], current_size[0], (3, 3), order='TL'))
            # Inv Flow with padding
            if inv_flow:
                layers.append(inv_flow_with_pad(current_size[0], current_size[0], (if_kernel_size, if_kernel_size), order='TL'))
                # layers.append(nn.BatchNorm2d(100))
                # layers.append(nn.BatchNorm3d(100))
                # layers.append(nn.LayerNorm([current_size[2], current_size[1], current_size[0]]))
                # layers.append(nn.ReLU())
                # layers.append(ActNorm(current_size[0]))
                # layers.append(act_if(current_size))

                # layers.append(inv_flow_with_pad(current_size[0], current_size[0], (if_kernel_size, if_kernel_size), order='TR'))
                # layers.append(nn.BatchNorm2d(100))
                # layers.append(nn.ReLU())
                # layers.append(ActNorm(current_size[0]))
                # ActNorm(current_size[0]),
                # layers.append(act_if(current_size))

                # layers.append(inv_flow_with_pad(current_size[0], current_size[0], (if_kernel_size, if_kernel_size), order='BL'))
                # layers.append(nn.BatchNorm2d(100))
                # layers.append(nn.ReLU())
                # layers.append(ActNorm(current_size[0]))
                # ActNorm(current_size[0]),
                # layers.append(act_if(current_size))

                # layers.append(inv_flow_with_pad(current_size[0], current_size[0], (if_kernel_size, if_kernel_size), order='BR'))
                # layers.append(nn.ReLU())
                # layers.append(nn.BatchNorm2d(100))
                # layers.append(nn.Sequential(inv_flow_with_pad(current_size[0], current_size[0], (2, 2), order='BR'),
                # # nn.ReLU(),
                # ActNorm(current_size[0]),
                # inv_flow_with_pad(current_size[0], current_size[0], (2, 2), order='BL'),
                # # nn.ReLU(),
                # ActNorm(current_size[0]),
                # inv_flow_with_pad(current_size[0], current_size[0], (2, 2), order='TR'),
                # # nn.ReLU(),
                # ActNorm(current_size[0]),
                # inv_flow_with_pad(current_size[0], current_size[0], (2, 2), order='TL'),))
            
            # Inv Flow without padding
            if inv_conv_no_pad:
                layers.append(inv_flow_no_pad(current_size[0], current_size[0], (2, 2)))

            if activation in ['Spline', 'SLR', 'BSpline']:
                # if not (l == num_blocks - 1 and k == block_size - 1):
                    # Dont place activation at end of last block
                layers.append(act(size=current_size))

            layers.append(Coupling(current_size, width=coupling_width))
            # layers.append(nn.BatchNorm2d(100))

        if split_prior and l < num_blocks - 1:
            layers.append(SplitPrior(current_size, NegativeGaussianLoss))
            current_size = (current_size[0] // 2, current_size[1], current_size[2])

    return FlowSequential(NegativeGaussianLoss(size=current_size), 
                         *layers)


def main():
    # set cuda device 1 to train the model
    # torch.cuda.set_device(1)
    now = datetime.now()
    # dd/mm/YY HH/MM/SS
    date_time = now.strftime("%d:%m:%Y_%H:%M:%S")
    run_name = f'_{date_time}'
    torch.cuda.empty_cache()
    config = {
        'name': f'2L-16K_IF_Glow_M{run_name}', 
        'eval_epochs': 1,
        'sample_epochs': 1,
        'max_eval_ex': float('inf'),

        'log_interval': 100,
        'lr': 1e-5,
        # 'gamma': 0.96170,
        'gamma': 0.96170,
        'epochs': 2000,
        'warmup_epochs': 1,

        'num_blocks': 2,
        'block_size': 16,
        'coupling_width': 512, # for coupling layer
        'batch_size': 100,
        'grad_clip_norm': True,
        'grad_clip': 00.01,
        'grad_clip_value': 0.02,
        'actnorm': True, # True, for avoiding vanishing gradient 'error: ' File "/home/sandeep.nagar/SelfNormalizingFlows/snf/layers/splines/rational_quadratic.py", line 77, in rational_quadratic_spline     if torch.min(inputs) < left or torch.max(inputs) > right: RuntimeError: operation does not have an identity.
        'split_prior': True, # True, not helps reconstruction
        'activation': 'Spline', # 'Spline', 'SLR', 'None
        'n_bins': 5, # 5 for spline activation
        'tail_bound': 20, # for spline activation 20 for best results

        'modified_grad': True,
        'add_recon_grad': True,
        'sym_recon_grad': True,
        
        'recon_loss_weight': 0.0,
        'sample_true_inv': True,
        'plot_recon': True,
        'vis_epochs': 10_000,
        'eval_train': True, # False evaluate on train set
        'wandb_project': 'inv_flow_mnist_02_10',
        'wandb_entity': 'carlobob031',

        'log_timing': True,
        'wandb': True,

        # model specific
        'snf': False, # SelfNormConv
        'ff': False, # Finc_FlowUnit
        'inv_flow': False, # Inv_FlowUnit
        'if_kernel_size': 3,
        'inv_conv': False, # Inv_Flow with padding
        'inv_conv_no_pad': True, # Inv_Flow without padding

        'optimizer_name': 'Adam', # Adam, Adamax, SGD,
        'scheduler_name': 'ExponentialLR', # StepLR, MultiStepLR, CosineAnnealingLR, ExponentialLR, CosineAnnealingWarmRestarts, expo_lr
        # 'checkpoints': './wandb/run-20240723_133448-ttf1m44b/files/checkpoint.tar'
    }

    train_loader, val_loader, test_loader = load_data(data_aug=False, batch_size=config['batch_size'])

    model = create_model(snf=config['snf'],
                        ff=config['ff'],
                        inv_flow=config['inv_flow'], 
                        inv_conv=config['inv_conv'],
                        inv_conv_no_pad=config['inv_conv_no_pad'],
                        if_kernel_size=config['if_kernel_size'],
                        coupling_width=config['coupling_width'],
                        num_blocks=config['num_blocks'],
                        block_size=config['block_size'],
                        tail_bound=config['tail_bound'],
                        n_bins=config['n_bins'],
                        sym_recon_grad=config['sym_recon_grad'],
                        actnorm=config['actnorm'],
                        activation=config['activation'],
                        split_prior=config['split_prior'],
                        recon_loss_weight=config['recon_loss_weight']).to('cuda')
    print(model)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print('Total_params Inv_flow , :', pytorch_total_params/1e6, 'M')
    print('config:', config)

    if config['optimizer_name'] is 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=config['lr'], betas=(0.9, 0.999)) # ,

    if config['optimizer_name'] is 'Adamax':
        optimizer = optim.Adamax(model.parameters(), lr=config['lr'], betas=(0.9, 0.999)) # ,

    if config['optimizer_name'] is 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=0.95, weight_decay=1e-5) # ,

    if config['scheduler_name'] is 'StepLR':
        scheduler = StepLR(optimizer, step_size=25, gamma=config['gamma'], verbose=True) # 0.9999997
    if config['scheduler_name'] is 'MultiStepLR':
        scheduler = MultiStepLR(optimizer, milestones=[2, 4, 50, 80, 240], gamma=config['gamma'], verbose=True)
    if config['scheduler_name'] is 'ExponentialLR':
        scheduler = ExponentialLR(optimizer, gamma=config['gamma'], last_epoch=-1, verbose=True)
    if config['scheduler_name'] is 'CosineAnnealingLR':                                            #```'''Todo '''```
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 900, verbose=True) # max_iter =5000
    if config['scheduler_name'] is 'CosineAnnealingWarmRestarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=1, eta_min=5e-8, verbose=True) # T_0=100, T_mult=1, eta_min=1e-6
        # scheduler = None

    # checkpoint = torch.load(config['checkpoints'])
    # model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    experiment = Experiment(model, train_loader, val_loader, test_loader,
                            optimizer, scheduler, **config)

    experiment.run()

#```'''Todo '''```
# CosineAnnealingLR
# 1. Add CosineAnnealingLR scheduler
# 2. Add two layers of iflow
