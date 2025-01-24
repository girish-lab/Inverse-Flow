import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ExponentialLR, MultiStepLR

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
from inf.datasets.imagenet import load_data
# from inf.datasets.cifar10 import load_data
from inf.layers.activations import SmoothLeakyRelu, SplineActivation, Identity
from inf.layers.inv_flow import Inv_FlowUnit         # This is the important part for Invertible Convolution (Inv_flow)
from inf.layers.inv_conv import inv_flow_with_pad, inv_flow_no_pad        # This is the important part for Invertible Convolution (Inv_flow)
from inf.layers.fincflow import Finc_FlowUnit         # This is the important part for Invertible Convolution (Inv_flow)
from datetime import datetime

class MemoryTracker:
    def __init__(self):
        """ Initializes the MemoryTracker class to track GPU memory usage. """
        self.initial_allocated = torch.cuda.memory_allocated()
        self.initial_reserved = torch.cuda.memory_reserved()

    def track_memory(self):
        """ Prints the current memory usage on the GPU. """
        allocated = torch.cuda.memory_allocated()  # Memory used by tensors
        reserved = torch.cuda.memory_reserved()    # Memory reserved by caching allocator

        print(f"Memory Allocated: {allocated / 1024**3:.3f} GB")
        print(f"Memory Reserved: {reserved / 1024**3:.3f} GB")
        
    def track_difference(self):
        """ Tracks the difference in memory usage since the object was created. """
        allocated = torch.cuda.memory_allocated() - self.initial_allocated
        reserved = torch.cuda.memory_reserved() - self.initial_reserved
        
        print(f"Memory Allocated Change: {allocated / 1024**3:.3f} GB")
        print(f"Memory Reserved Change: {reserved / 1024**3:.3f} GB")

    def reset(self):
        """ Resets the initial memory values. """
        self.initial_allocated = torch.cuda.memory_allocated()
        self.initial_reserved = torch.cuda.memory_reserved()
activations = {
    'SLR':lambda size: SmoothLeakyRelu(alpha=0.3),
    'Spline': lambda size: SplineActivation(size, n_bins=10, tail_bound=20, individual_weights=False),
}

def create_model(snf=False, ff=False,inv_conv=False, inv_flow=True, inv_conv_no_pad=False,
                num_blocks=3, block_size=32,
                coupling_width=512,
                 if_kernel_size=3, sym_recon_grad=False, 
                 tail_bound=30, n_bins=10,

                 activation='Spline', actnorm=True, split_prior=True, recon_loss_weight=1000.0):
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
            # layers.append(SelfNormConv(current_size[0], current_size[0], (1, 1), 
            #                            bias=True, stride=1, padding=0,
            #                            sym_recon_grad=sym_recon_grad, 
            #                            recon_loss_weight=recon_loss_weight))
            # if k % 2 == 0:
            #     layers.append(inv_flow(current_size[0],current_size[0], (3, 3), order='TR',))
                layers.append(inv_flow_with_pad(current_size[0], current_size[0], (if_kernel_size, if_kernel_size), order='TL'))
                # layers.append(inv_flow_with_pad(current_size[0], current_size[0], (if_kernel_size, if_kernel_size), order='TR'))
                # layers.append(inv_flow_with_pad(current_size[0], current_size[0], (if_kernel_size, if_kernel_size), order='BL'))
                # layers.append(inv_flow_with_pad(current_size[0], current_size[0], (if_kernel_size, if_kernel_size), order='BR'))  
            # layers.append(inv_flow_no_pad(current_size[0], current_size[0], (3, 3)))

            # Inv Flow without padding
            if inv_conv_no_pad:
                layers.append(inv_flow_no_pad(current_size[0], current_size[0], (3, 3)))

            if activation in ['Spline', 'SLR', 'BSpline']:
                if not (l == num_blocks - 1 and k == block_size - 1):
                    # Dont place activation at end of last block
                    layers.append(act(current_size))
            # layers.append(Inv_FlowUnit(current_size[0], current_size[0], (3, 3)))#, order='TR'))
            # layers.append(Finc_FlowUnit(current_size[0], current_size[0], (3, 3)))#, order='TR'))
            layers.append(Coupling(current_size, width=coupling_width))
            

        if split_prior and l < num_blocks - 1:
            layers.append(SplitPrior(current_size, NegativeGaussianLoss))
            current_size = (current_size[0] // 2, current_size[1], current_size[2])

    return FlowSequential(NegativeGaussianLoss(size=current_size), 
                         *layers)


def main():
    # clear the cache
    torch.cuda.empty_cache()
    now = datetime.now()
    # dd/mm/YY HH/MM/SS
    date_time = now.strftime("%d:%m:%Y_%H:%M:%S")
    multi_gpu = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        multi_gpu = True
    run_name = f'_{date_time}'
    config = {
        'name': f'3L-48K_IF_Glow_ImgageNet_{run_name}',
        'multi_gpu': multi_gpu,
        'eval_epochs': 1,
        'sample_epochs': 1,
        'log_interval': 100,
        'lr': 1e-4,
        'gamma': 0.99997,
        'num_blocks': 2,
        'block_size': 32,

        'coupling_width': 256, # for coupling layer
        'batch_size': 100,

        'grad_clip_norm': True,
        'grad_clip': 0.01,       # imporant for Inv_Flow model
        'grad_clip_value': 0.1,
        # 'grad_clip_norm': 05.06,

        'modified_grad': False, # 
        'add_recon_grad': False, # 
        'sym_recon_grad': False, #

        # Logging Options
        'log_timing': True,
        'wandb': True,
        'wandb_project': 'IF_glow_imagenet32',
        'wandb_entity': 'carlobob031',
        'log_interval': 100,
        'vis_epochs': 2, 
    
        'n_samples': 100,
        'sample_dir': 'samples',
        'sample_true_inv': True,
        'plot_recon': True,
        'checkpoint_path': None,
        
        # model options
        'activation': 'Spline', # Spline, SLR, BSpline
        'actnorm': False, # False for Inv_Flow model, True for snf
        'split_prior': True,
        'recon_loss_weight': 0.0, # not imp for Inv_Flow model, for snf
        'n_bins': 7, # 5 for spline activation
        'tail_bound': 10, # for spline activation 20 for best results
        'warmup_epochs': 2,
        'step_epochs': 1,
        'step_gamma': 1.0,
        'eval_train': True, # False evaluate on train set
        
        'resolution': 32, #imagenet32, imagenet64


        # model specific
        'snf': False, # SelfNormConv
        'ff': False, # Finc_FlowUnit
        'inv_flow': False, # Inv_FlowUnit
        'if_kernel_size': 2,
        'inv_conv': False, # Inv_Flow with padding
        'inv_conv_no_pad': True, # Inv_Flow without padding

        'optimizer_name': 'Adam', # Adam, Adamax, SGD,
        'scheduler_name': 'ExponentialLR', # StepLR, MultiStepLR, CosineAnnealingLR, ExponentialLR, CosineAnnealingWarmRestarts, expo_lr
        # 'checkpoints': './wandb/run-20240723_133448-ttf1m44b/files/checkpoint.tar'
        'test': False, # True for testing the model setup 
        'input_size': (3, 32, 32), # input size of the image
        'multi_gpu': False,

    }

    # train_loader, val_loader, test_loader = load_data(data_aug=True, batch_size=config['batch_size'])
    data_dir = '/scratch/imagenet32'
    train_loader, val_loader, test_loader = load_data(data_aug=True, 
                                                      batch_size=config['batch_size'],
                                                      resolution=config['resolution'],
                                                      data_dir=data_dir)
    memory_tracker = MemoryTracker()
    print("Initial Memory Usage:")
    memory_tracker.track_memory()
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
                        activation=config['activation'],
                        actnorm=config['actnorm'],
                        split_prior=config['split_prior'],
                        recon_loss_weight=config['recon_loss_weight'])
    if config['multi_gpu']:
        model = torch.nn.DataParallel(model)
    model = model.to('cuda')
    
    # print(model)
    pytorch_total_params = sum(p.numel() for p in model.parameters()) 
    print('Total_params Inv_flow , :', pytorch_total_params/1e6, 'M ')
    print('config:', config) 

    if config['optimizer_name'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=config['lr'], betas=(0.9, 0.999)) # ,

    if config['optimizer_name'] == 'Adamax':
        optimizer = optim.Adamax(model.parameters(), lr=config['lr'], betas=(0.9, 0.999)) # ,

    if config['optimizer_name'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=0.95, weight_decay=1e-5) # ,

    if config['scheduler_name'] == 'StepLR':
        scheduler = StepLR(optimizer, step_size=25, gamma=config['gamma'], verbose=True) # 0.9999997
    if config['scheduler_name'] == 'MultiStepLR':
        scheduler = MultiStepLR(optimizer, milestones=[2, 4, 10, 80, 240], gamma=config['gamma'], verbose=True)
    if config['scheduler_name'] == 'ExponentialLR':
        scheduler = ExponentialLR(optimizer, gamma=config['gamma'], last_epoch=-1, verbose=True)
    if config['scheduler_name'] == 'CosineAnnealingLR':                                            #```'''Todo '''```
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 100, verbose=True) # max_iter =5000
    if config['scheduler_name'] == 'CosineAnnealingWarmRestarts':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=30, T_mult=1, eta_min=5e-9, verbose=True) # T_0=100, T_mult=1, eta_min=1e-6
        # scheduler = None

    # checkpoint = torch.load(config['checkpoints'])
    # model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    experiment = Experiment(model, train_loader, val_loader, test_loader,
                            optimizer, scheduler, memory_tracker, **config)
    experiment.run()