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
from torchvision import transforms
from inf.train.datatransforms import ToTensorNoNorm
import math
import torch
from torch.utils.data import Dataset


class ToTensorNoNorm(transforms.ToTensor):
    """Custom ToTensor transformation that does not normalize"""
    def __call__(self, pic):
        return torch.tensor(pic).float()

class NormalizingFlowImageDataset(Dataset):
    def __init__(self, 
                 num_samples=1000, 
                 image_size=(12, 16, 16),  # Changed to match your original input shape
                 distribution='normal', 
                 normalize=True, 
                 transform=None):
        """
        Enhanced dataset generator for Normalizing Flow models
        
        Args:
            num_samples (int): Number of samples to generate
            image_size (tuple): Dimensions of the image (channels, height, width)
            distribution (str): Type of distribution to generate from
            normalize (bool): Whether to normalize the generated data
            transform (callable, optional): A function/transform to apply to the samples
        """
        self.num_samples = num_samples
        self.image_size = image_size
        self.normalize = normalize
        self.transform = transform
        
        # Generate data based on specified distribution
        if distribution == 'normal':
            self.data = self._generate_normal_data()
        elif distribution == 'uniform':
            self.data = self._generate_uniform_data()
        else:
            raise ValueError(f"Unsupported distribution: {distribution}")
        
        # Normalize if specified
        if normalize:
            self.data = self._normalize_data(self.data)
    
    def _generate_normal_data(self):
        """Generate data from a normal distribution"""
        data = torch.randn(self.num_samples, *self.image_size)
        return data
    
    def _generate_uniform_data(self):
        """Generate data from a uniform distribution"""
        data = torch.rand(self.num_samples, *self.image_size)
        return data
    
    def _normalize_data(self, data):
        """
        Normalize data to have zero mean and unit variance
        
        Args:
            data (torch.Tensor): Input data tensor
        
        Returns:
            torch.Tensor: Normalized data
        """
        # Compute mean and std along all dimensions except the first (samples)
        mean = data.mean(dim=(0, 2, 3), keepdim=True)
        std = data.std(dim=(0, 2, 3), keepdim=True)
        
        # Add small epsilon to prevent division by zero
        normalized_data = (data - mean) / (std + 1e-7)
        return normalized_data
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Apply the transformation if provided
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
    @property
    def tensors(self):
        """Return data tensor as a tuple for compatibility"""
        return (self.data,)
    
    @property
    def dataset(self):
        """Return self for dataset access"""
        return self
    
    def get_data_stats(self):
        """
        Compute and print statistics about the generated data
        
        Returns:
            dict: Statistics of the data
        """
        data_stats = {
            'shape': self.data.shape,
            'mean': self.data.mean(),
            'std': self.data.std(),
            'min': self.data.min(),
            'max': self.data.max()
        }
        
        print("Dataset Statistics:")
        for key, value in data_stats.items():
            print(f"{key}: {value}")
        
        return data_stats

# Example usage
def test_dataset():
    # Generate a dataset with 64 samples of shape (12, 16, 16)
    dataset = NormalizingFlowImageDataset(
        num_samples=64, 
        image_size=(12, 16, 16), 
        distribution='normal',
        normalize=True
    )
    
    # Print dataset statistics
    stats = dataset.get_data_stats()
    
    # Verify dimensionality
    print("\nDimensionality Check:")
    sample = dataset[0]
    print(f"Sample shape: {sample.shape}")
    print(f"Total dataset samples: {len(dataset)}")
    
    return dataset



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
        'lr': 1e-4,
        'gamma': 0.1097170,
        'num_blocks': 1, #3
        'block_size': 4, #48
        'coupling_width': 128, # for coupling layer
        
        'batch_size': 16,
        
        'grad_clip_norm': True,
        'grad_clip': 0.01,       # imporant for Inv_Flow model
        'grad_clip_value': 0.1,
        # 'grad_clip_norm': 05.06,
        
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
        # 'grad_clip_norm': 10_000,
        'warmup_epochs': 0,
        'step_epochs': 1,
        'step_gamma': 1.0,
        
        'test': True,
    }

    # Define your transformations
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Pad(int(math.ceil(32 * 0.04)), padding_mode='edge'),
        transforms.RandomAffine(degrees=0, translate=(0.04, 0.04)),
        transforms.CenterCrop(32),
        transforms.RandomHorizontalFlip(),
        ToTensorNoNorm()
    ])

    test_transform = transforms.Compose([
        ToTensorNoNorm()
    ])

    # Use NormalizingFlowImageDataset for training
    dataset = NormalizingFlowImageDataset(num_samples=1000, 
                                        image_size=(3, 32, 32), 
                                        distribution='normal', 
                                        normalize=True, 
                                        transform=train_transform)

    # DataLoader setup
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    # Use DataLoader for validation and test if needed
    val_loader = train_loader  # Placeholder for validation
    test_loader = train_loader  # Placeholder for testing
    
    
    # train_loader, val_loader, test_loader = load_data(data_aug=False, resolution=32, 
    #           data_dir='../data/imagenet32', batch_size=config['batch_size'])
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
                            optimizer, scheduler, **config)

    experiment.run()
    
if __name__ == "__main__":
    # test_dataset()
    main()
