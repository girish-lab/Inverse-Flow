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
from inf.datasets.imagenet import load_data

# Dummy dataset for random image generation
class RandomImageDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples, image_size=(3, 32, 32)):
        self.num_samples = num_samples
        self.image_size = image_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image = torch.randn(self.image_size)  # Random image
        return image



def create_model(num_blocks=3, block_size=48, sym_recon_grad=False, 
                 actnorm=True, split_prior=True, recon_loss_weight=1000.0):
    current_size = (3, 32, 32)

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
            layers.append(Coupling(current_size))

        if split_prior and l < num_blocks - 1:
            layers.append(SplitPrior(current_size, NegativeGaussianLoss))
            current_size = (current_size[0] // 2, current_size[1], current_size[2])

    return FlowSequential(NegativeGaussianLoss(size=current_size), 
                         *layers)


def main():
    config = {
        'name': '3L-48K Glow SNF Recon1000 ImageNet',
        'eval_epochs': 1,
        'sample_epochs': 1,
        'log_interval': 100,
        'lr': 1e-3,
        'num_blocks': 3,
        'block_size': 48,
        'batch_size': 64,
        'modified_grad': True,
        'add_recon_grad': True,
        'sym_recon_grad': False,
        'actnorm': True,
        'split_prior': True,
        'activation': 'None',
        'recon_loss_weight': 1000.0,
        'sample_true_inv': True,
        'plot_recon': True,
        'vis_epochs': 10_000,
        'grad_clip_norm': 10_000,
        'warmup_epochs': 0,
        'step_epochs': 1,
        'step_gamma': 1.0,
        
        'test': True,
    }
    # Use RandomImageDataset for testing
    train_loader = torch.utils.data.DataLoader(RandomImageDataset(1000), batch_size=config['batch_size'], shuffle=True)
    val_loader = train_loader  # Using train_loader as placeholder for validation
    test_loader = train_loader  # Using train_loader as placeholder for test
    # train_loader, val_loader, test_loader = load_data(data_aug=False, resolution=32, 
    #           data_dir='data/imagenet32', batch_size=config['batch_size'])
    #         #   data_dir='...../share1/sandeep.nagar/imagenet32', batch_size=config['batch_size'])
            

    model = create_model(num_blocks=config['num_blocks'],
                         block_size=config['block_size'], 
                         sym_recon_grad=config['sym_recon_grad'],
                         actnorm=config['actnorm'],
                         split_prior=config['split_prior'],
                         recon_loss_weight=config['recon_loss_weight']).to('cuda')
    print(model)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print('Total_params Inv_flow , :', pytorch_total_params/1e6, 'M')
    print('config:', config)
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], betas=(0.9, 0.999))
    scheduler = StepLR(optimizer, step_size=config['step_epochs'], gamma=config['step_gamma'])

    experiment = Experiment(model, train_loader, val_loader, test_loader,
                            optimizer, scheduler, test=True, **config)

    experiment.run()
    
if __name__ == "__main__":
    main()
