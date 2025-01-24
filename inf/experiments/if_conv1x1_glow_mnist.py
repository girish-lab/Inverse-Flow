import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR, MultiStepLR, ReduceLROnPlateau, ExponentialLR
from inf.layers import Dequantization, Normalization
from inf.layers.distributions.uniform import UniformDistribution
from inf.layers.splitprior import SplitPrior
from inf.layers.flowsequential import FlowSequential
from inf.layers.conv1x1 import Conv1x1
from inf.layers.actnorm import ActNorm
from inf.layers.squeeze import Squeeze
from inf.layers.transforms import LogitTransform
from inf.layers.coupling import Coupling
from inf.train.losses import NegativeGaussianLoss
from inf.train.experiment import Experiment
from inf.datasets.mnist import load_data
from inf.layers.activations import SmoothLeakyRelu, SplineActivation, Identity
from collections import OrderedDict
from datetime import datetime
# from inf.layers.inv_flow import Inv_FlowUnit    # This is the important part for Invertible Convolution (Inv_flow)
# from inf.layers.inv_conv import inv_flow        # This is the important part for Invertible Convolution (Inv_flow)
from inf.layers.inv_conv import inv_flow_with_pad, inv_flow_no_pad        # This is the important part for Invertible Convolution (Inv_flow)
activations = {
    'SLR':lambda size: SmoothLeakyRelu(alpha=0.3),
    'Spline': lambda size: SplineActivation(size, n_bins = 5, tail_bound=20, individual_weights=True),
}
# 
def create_model(num_blocks=2, block_size=16, sym_recon_grad=False, 
                 activation='Spline',actnorm=False, split_prior=False, recon_loss_weight=1.0):

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
            # if actnorm:
            #     layers.append(ActNorm(current_size[0]))
    
            # layers.append(Inv_FlowUnit(current_size[0], current_size[0], (3, 3)))#, order='TR'))
            layers.append(inv_flow_no_pad(current_size[0], current_size[0], (3, 3)))
            # layers.append( inv_flow(current_size[0],current_size[0], (3, 3), order='TR',))
            # layers.append( inv_flow(current_size[0],current_size[0], (3, 3), order='BL',))
            # layers.append( inv_flow(current_size[0],current_size[0], (3, 3), order='BR',))

            # layers.append(Conv1x1(current_size[0]))
            layers.append(Coupling(current_size))

            if activation in ['Spline', 'SLR']:
                if not (l == num_blocks - 1 and k == block_size - 1):
                    # Dont place activation at end of last block
                    layers.append(act(current_size))

        if split_prior and l < num_blocks - 1:
            layers.append(SplitPrior(current_size, NegativeGaussianLoss))
            current_size = (current_size[0] // 2, current_size[1], current_size[2])

    return FlowSequential(NegativeGaussianLoss(size=current_size), 
                         *layers)


def main():
    now = datetime.now()
    # dd/mm/YY HH/MM/SS
    date_time = now.strftime("%d:%m:%Y_%H:%M:%S")
    run_name = f'_{date_time}'

    config = {
        
        # Training Options (important first)
        'name': f'2L-16K_IF_Glow_M{run_name}',
        'eval_epochs': 1,
        'lr': 1e-5,
        'batch_size': 100,
        'epochs': 200,
        'max_eval_ex': float('inf'),
        'recon_loss_weight': 0.0,
        'grad_clip': 0.01,
        'grad_clip_norm': True,
        'grad_clip_value': 0.001,
        'warmup_epochs': 1,

        'modified_grad': True,
        'add_recon_grad': True,
        'sym_recon_grad': True,

        # Logging Options
        'log_timing': True,
        'wandb': True,
        'wandb_project': 'inv_flow_mnist',
        'wandb_entity': 'carlobob031',
        'sample_epochs': 1,
        'log_interval': 100,
        'vis_epochs': 10_000,
        'eval_train': False,
        'n_samples': 100,
        'sample_dir': 'samples',
        'sample_true_inv': True,
        'plot_recon': True,
        'checkpoint_path': None,

        # Model Architecutre Options
        'actnorm': False, 
        'num_blocks': 2, # 'num_blocks': 3, 'block_size': 16,
        'block_size': 16,
        'split_prior': True, # helps Invertible Convolution
        'activation': 'Spline', # 'activation': 'Spline', 'activation': 'SLR',
    }

    train_loader, val_loader, test_loader = load_data(data_aug=False, batch_size=config['batch_size'])

    model = create_model(num_blocks=config['num_blocks'],
                         block_size=config['block_size'], 
                         sym_recon_grad=config['sym_recon_grad'],
                         activation=config['activation'],
                         actnorm=config['actnorm'],
                         split_prior=config['split_prior'],
                         recon_loss_weight=config['recon_loss_weight']).to('cuda')
    
    print(model)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print('Total_params Inv_flow , :', pytorch_total_params/1000000, 'Millions')
    # wandb.log({"total_params": pytorch_total_params})
    print('config:', config)

    optimizer = optim.Adam(model.parameters(), lr=config['lr'], betas=(0.9, 0.999))
    # scheduler = StepLR(optimizer, step_size=1, gamma=1.0)
    scheduler = ExponentialLR(optimizer, gamma=0.9997, last_epoch=-1)

    experiment = Experiment(model, train_loader, val_loader, test_loader,
                            optimizer, scheduler, **config)

    experiment.run()