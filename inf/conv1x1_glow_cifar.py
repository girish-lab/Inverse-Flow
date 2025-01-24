import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR

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
from inf.datasets.cifar10 import load_data

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

def create_model(num_blocks=3, block_size=32, sym_recon_grad=False, 
                 actnorm=False, split_prior=False, recon_loss_weight=1.0):
    current_size = (3, 32, 32)

    alpha = 1e-6
    layers = [
        Dequantization(UniformDistribution(size=current_size)),
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
            # layers.append(Conv1x1(current_size[0]))
            layers.append(Coupling(current_size))

        if split_prior and l < num_blocks - 1:
            layers.append(SplitPrior(current_size, NegativeGaussianLoss))
            current_size = (current_size[0] // 2, current_size[1], current_size[2])

    return FlowSequential(NegativeGaussianLoss(size=current_size), 
                         *layers)


def main():
    config = {
        'name': '2L-4K_Glow_Exact_C',
        'eval_epochs': 1,
        'sample_epochs': 1,
        'log_interval': 100,
        'lr': 1e-3,
        'num_blocks': 2,
        'block_size': 32,
        'batch_size': 100,
        'n_samples': 100,
        'modified_grad': True,
        'add_recon_grad': True,
        'sym_recon_grad': True,
        'actnorm': True,
        'split_prior': True,
        'activation': 'None',
        'recon_loss_weight': 0.0,
        'sample_true_inv': True,
        'plot_recon': True,
        'grad_clip_norm': None,
        'warmup_epochs': 10,
        'wandb_project': 'snf_mnist',
        'wandb_entity': 'carlobob031',
        'wandb': True,
        'wandb_project': 'snf_cifar10',
        'wandb_entity': 'carlobob031',
        'grad_clip': 1.1,
        'grad_clip_value': 1.1,
        'grad_clip_norm': True,
    }

    train_loader, val_loader, test_loader = load_data(data_aug=True, batch_size=config['batch_size'])
    memory_tracker = MemoryTracker()
    print("Initial Memory Usage:")
    memory_tracker.track_memory()
    model = create_model(num_blocks=config['num_blocks'],
                         block_size=config['block_size'], 
                         sym_recon_grad=config['sym_recon_grad'],
                         actnorm=config['actnorm'],
                         split_prior=config['split_prior'],
                         recon_loss_weight=config['recon_loss_weight']).to('cuda')

    optimizer = optim.Adam(model.parameters(), lr=config['lr'], betas=(0.9, 0.999))
    scheduler = StepLR(optimizer, step_size=1, gamma=1.0)

    experiment = Experiment(model, train_loader, val_loader, test_loader,
                            optimizer, scheduler, memory_tracker **config)

    experiment.run()
