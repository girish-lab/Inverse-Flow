import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR

from inf.layers import Dequantization, Normalization
from inf.layers.distributions.uniform import UniformDistribution
from inf.layers.flowsequential import FlowSequential
from inf.layers.emerging.emerging_module import Emerging
from inf.layers.activations import SmoothLeakyRelu, SplineActivation, Identity
from inf.layers.squeeze import Squeeze
from inf.layers.transforms import LogitTransform
from inf.train.losses import NegativeGaussianLoss
from inf.train.experiment import Experiment
from inf.datasets.mnist import load_data

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
    'Spline': lambda size: SplineActivation(size, n_bins=10, tail_bound=70, individual_weights=True),
}

def create_model(num_layers=9, sym_recon_grad=False, 
                 activation='Spline', recon_loss_weight=1.0, num_blocks=2):
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
            layers.append(Emerging(current_size[0]))
            
            if not (b == num_blocks - 1 and l == block_size - 1):
                layers.append(act(current_size))
        
        if not (b == num_blocks - 1):
            layers.append(Squeeze())
            current_size = (current_size[0]*4,
                            current_size[1]//2,
                            current_size[2]//2)

    return FlowSequential(NegativeGaussianLoss(size=current_size), 
                         *layers)


def main():
    config = {
        'name': '9 layers Conv Emerging Spline MNIST',
        'eval_epochs': 1,
        'sample_epochs': 1,
        'log_interval': 100,
        'lr': 1e-5,
        'num_layers': 9,
        'batch_size': 100,
        'modified_grad': False,
        'add_recon_grad': False,
        'sym_recon_grad': False,
        'activation': 'Spline',
        'recon_loss_weight': 0.0,
        'log_timing': True,
        'sample_true_inv': True,
        'plot_recon': True,

        'grad_clip_norm': True,
        'grad_clip': 0.01,
        'grad_clip_value': 0.01,
    }

    train_loader, val_loader, test_loader = load_data(batch_size=config['batch_size'])
    memory_tracker = MemoryTracker()
    print("Initial Memory Usage:")
    memory_tracker.track_memory()
    model = create_model(num_layers=config['num_layers'], 
                         sym_recon_grad=config['sym_recon_grad'],
                         activation=config['activation'],
                         recon_loss_weight=config['recon_loss_weight']).to('cuda')
    print('config:', config)
    print(model)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print('Total_params Inv_flow , :', pytorch_total_params)

    optimizer = optim.Adam(model.parameters(), lr=config['lr'], betas=(0.9, 0.999))
    scheduler = StepLR(optimizer, step_size=1, gamma=1.0)

    experiment = Experiment(model, train_loader, val_loader, test_loader,
                            optimizer, scheduler, memory_tracker, **config)

    experiment.run()