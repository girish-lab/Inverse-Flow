import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR, MultiStepLR, ReduceLROnPlateau, ExponentialLR

from layers import Dequantization, Normalization
from layers.distributions.uniform import UniformDistribution
from layers.splitprior import SplitPrior
from layers.flowsequential import FlowSequential
from layers.conv1x1 import Conv1x1
from snf.layers.activations import SmoothLeakyRelu, SplineActivation, Identity
from layers.actnorm import ActNorm, ActNormFC, ActNorm_glow
from layers.squeeze import Squeeze
from layers.transforms import LogitTransform
from layers.coupling import Coupling
from train.losses import NegativeGaussianLoss
from train.experiment import Experiment
from datasets.mnist import load_data


# from layers.conv import PaddedConv2d#, Conv1x1
# from fastflow import FastFlowUnit
from layers.inv_conv import inv_flow         # This is the important part for Invertible Convolution (Inv_flow)


from datetime import datetime

activations = {
    'SLR':lambda size: SmoothLeakyRelu(alpha=0.3),
    'Spline': lambda size: SplineActivation(size, n_bins=20, tail_bound=25, individual_weights=True), # (tail_bound>=100.0), individual_weights=True),
}

now = datetime.now()


# dd/mm/YY HH/MM/SS
run_name = now.strftime("%d:%m:%Y %H:%M:%S")
optimizer_ = "Adam" # SGD, Adam, AdamW
scheduler_ = "Exp_0.997"
lr = 1e-5 # , 1e-3, 1e-4*, 1e-5, 1e-6, 

# look at the log scale, translation  
def create_model(num_blocks=2, block_size=16, activation='Spline', sym_recon_grad=False, 
                 actnorm=False, split_prior=False, recon_loss_weight=1.0, current_size=(1, 28, 28)):

    current_size = current_size
    act = activations[activation]

    alpha = 1e-2
    layers = [
        Dequantization(UniformDistribution(size=current_size)),
        Normalization(translation=0, scale=256),
        Normalization(translation=-alpha, scale=1 / (1 - 2 * alpha)),
        LogitTransform(),
    ]

    for b in range(num_blocks):
        layers.append(Squeeze())
        current_size = (current_size[0]*4, current_size[1]//2, current_size[2]//2)

        for k in range(block_size):
            layers.append(inv_flow(current_size[0], current_size[0], (3, 3)))
            
            if actnorm :
                if not  (b == num_blocks - 1 and k == block_size - 1):
                    layers.append(ActNorm(current_size[0])) # ActNormFC(current_size[0])) 
                    # layers.append(ActNorm_glow(current_size[0])) # ActNormFC(current_size[0])) 
            # layers.append(Conv1x1(current_size[0])) # ... (san)
            layers.append(Coupling(current_size, width=512)) # width=512 (san)
            if not (b == num_blocks - 1 and k == block_size - 1):
                # Dont place activation at end of last block
                layers.append(act(current_size))

        if split_prior and b < num_blocks - 1:
            # layers.append(SplitPrior(current_size, NegativeGaussianLoss))
            current_size = (current_size[0] // 2, current_size[1], current_size[2])

    return FlowSequential(NegativeGaussianLoss(size=current_size), 
                         *layers)
    # return NegativeGaussianLoss(size=current_size), *layers 

def main():
    config = {
        'name': f'a_IF_MNIST_{optimizer_}_{scheduler_}_{lr}_{run_name}',
        'eval_epochs': 1,
        'sample_epochs': 1,
        'log_interval': 10,
        'lr': lr,
        'num_blocks': 1,
        'block_size': 8,
        'epochs': 120,
        'batch_size': 1000,
        'grad_clip': 0.01, # for grad clipping, lower value is better (low bpd) (san) (0.01)
        'grad_clip_norm': True, # 'inf', 'l2, imporved the bpd (san)
        'modified_grad': False,
        'add_recon_grad': False,
        'sym_recon_grad': False,
        'activation': 'Spline', # 'SLR', 'Spline, (Spline is better)
        'actnorm': False, # (False)
        'split_prior': False,
        'recon_loss_weight': 1.0,
        'sample_true_inv': True,
        'plot_recon': True,
        'dataset': 'MNIST',
        'run_name': f'{run_name}',
        'Optimizer': optimizer_,
        'Scheduler': scheduler_
    }

    train_loader, val_loader, test_loader = load_data(data_aug=False, batch_size=config['batch_size'])

    model = create_model(num_blocks=config['num_blocks'],
                         block_size=config['block_size'], 
                         activation=config['activation'],
                         sym_recon_grad=config['sym_recon_grad'],
                         actnorm=config['actnorm'],
                         split_prior=config['split_prior'],
                         recon_loss_weight=config['recon_loss_weight']).to('cuda')
    # print(model.parameters())

    # print('config:', config)
    print(model)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print('Total_params Inv_flow , :', pytorch_total_params)
    #
    print('<<<<<<<<<<<<<<    config    >>>>>>>>>>>>>>', config)


    # optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9)
    # optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    # optimizer = optim.Adamax(model.parameters(), lr=config['lr'])
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=0.01) #( 0.011) # 0.01, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001

    # scheduler = StepLR(optimizer, step_size=1, gamma=1.0)
    # scheduler = None
    # scheduler = MultiStepLR(optimizer, milestones=[5, 10, 20, 50], gamma=0.1)   
    # scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10, threshold=1.0) 
    scheduler = ExponentialLR(optimizer, gamma=0.99, last_epoch=-1, verbose=True) # gamma= 0.997, 0.998, 0.999, 0.9999, 0.99999
    experiment = Experiment(model, train_loader, val_loader, test_loader,
                            optimizer, scheduler, **config)
    ## try KAN
    experiment.run()

if __name__ == '__main__':
    main()

    # python inv_flow_mnist.py 
    # L= 4, K= 32: # params = 32,0,76,428 (32M) 
    # L= 2, K= 32: # params = 19,7,47,972 (19M) 