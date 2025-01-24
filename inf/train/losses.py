import torch
import torch.nn as nn
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal
from torch import sparse

class NegativeLogLaplaceLoss(nn.Module):
    """
    Centered Negative LogLaplace Likelihood with std=1, constant terms
    are ignored.
    """
    def forward(self, input):
        return torch.abs(input).sum() * 1.4142
    
def clip_tensor(x, min_value=-10, max_value=10):
    return torch.clamp(x, min=min_value, max=max_value)
def clean_tensor(x):
    # Check for NaNs and replace them with zeros (or another strategy)
    x = torch.nan_to_num(x, nan=0.0, posinf=1e10, neginf=-1e10)
    return x
class NegativeGaussianLoss(nn.Module):
    """
    Standard Normal Likelihood (negative)
    """
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.dim = dim = int(np.prod(size))
        self.N = MultivariateNormal(torch.zeros(dim, device='cuda'),
                                    torch.eye(dim, device='cuda'),
                                    validate_args=None)

    def forward(self, input, context=None):
        return -self.log_prob(input, context).sum(-1)

    def log_prob(self, input, context=None, sum=True):
        device = input.device
        input = input.to(self.N.loc.device)
        try: 
            p = self.N.log_prob(input.view(-1, self.dim))
        except RuntimeError:
            p = self.N.log_prob(input.reshape(-1, self.dim))
        p = p.to(device)
        return p

    def sample(self, n_samples, context=None):
        x = self.N.sample((n_samples,)).view(n_samples, *self.size)
        log_px = self.log_prob(x, context)
        return x, log_px

class NegativeGaussianLoss_test(nn.Module):
    """
    Standard Normal Likelihood (negative)
    """
    def __init__(self, size):
        super().__init__()
        # self.device = device

        self.size = size
        self.dim =dim= int(np.prod(size))
        self.mean = torch.zeros(size)  # Example of initializing the mean attribute
        self.mean = torch.zeros(dim, device='cuda')
        self.log_std = torch.zeros(dim, device='cuda')
        # self.N = MultivariateNormal(torch.zeros(dim, device='cuda'),
        #                      torch.eye(dim, device='cuda') * 1e-3)  # Diagonal covariance with small variance
        # cov_matrix = torch.eye(self.dim, device='cuda')  # Creates a sparse identity matrix
        # self.N = MultivariateNormal(torch.zeros(self.dim, device='cuda'),
        #                      cov_matrix)
        # self.N = MultivariateNormal(torch.zeros(self.dim, device='cuda'),
                                    # torch.eye(self.dim, device='cuda'))

    def forward(self, input, context=None):
        return -self.log_prob(input, context).sum(-1)

    # def log_prob(self, input, context=None, sum=True):
    #     try: 
    #         # print("input.shape: ", input.shape)
    #         # print("self.dim: ", self.dim)
    #         # print("input.view(-1, self.dim).shape: ", input.view(-1, self.dim).shape)
    #         # print("input.reshape(-1, self.dim).shape: ", input.reshape(-1, self.dim).shape)
    #         # print("input view : ", input)
    #         # input = clean_tensor(input)
    #         # input = clip_tensor(input)
    #         p = self.N.log_prob(input.view(-1, self.dim))
    #     except RuntimeError:
    #         input = clean_tensor(input)
    #         input = clip_tensor(input)
    #         p = self.N.log_prob(input.reshape(-1, self.dim))
    #     return p
    # def log_prob(self, x):
        # Ensure mean and log_std are reshaped to match the shape of x
        # mean = self.mean.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # Adjust as needed
        # log_std = self.log_std.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # Adjust as needed
        
        # # Now calculate the log probability using broadcasting
        # normal_dist = torch.distributions.Normal(mean, torch.exp(log_std))
        # return torch.sum(normal_dist.log_prob(x), dim=-1)  # Sum over the last dimension (spatial locations)
    def log_prob(self, input):
        # log P(x) = sum(log P(x_i))
        input = clean_tensor(input)
        input = clip_tensor(input)
        return torch.sum(torch.distributions.Normal(self.mean, torch.exp(self.log_std)).log_prob(input.reshape(-1,  self.dim)), dim=-1)

    def sample(self, n_samples, context=None):
        x = self.N.sample((n_samples,)).view(n_samples, *self.size)
        log_px = self.log_prob(x, context)
        return x, log_px

    def sample(self, n_samples, context=None):
    # Sample from the Normal distribution
        normal_dist = torch.distributions.Normal(self.mean, torch.exp(self.log_std))
        x = normal_dist.sample((n_samples,))  # Sample n_samples from the distribution
        
        # Reshape the sampled tensor to match the expected shape
        x = x.view(n_samples, *self.size)  # Shape: [n_samples, *self.size]
        
        # Compute the log-probability of the samples
        log_px = self.log_prob(x)  # Assuming log_prob computes the log-likelihood of the samples
    
        return x, log_px


class LogGaussian(NegativeGaussianLoss):
    """
    Standard Normal Likelihood 
    """
    def forward(self, input, context=None):
        return self.log_prob(input, context).sum(-1)

class DiagonalGaussian:
    def __init__(self, dim, device='cuda'):
        self.dim = dim
        self.device = device
        self.mean = torch.zeros(dim, device=device)
        self.log_std = torch.zeros(dim, device=device)
    
    def forward(self, input, context=None):
        return -self.log_prob(input, context).sum(-1)

    def log_prob(self, x):
        # log P(x) = sum(log P(x_i))
        return torch.sum(torch.distributions.Normal(self.mean, torch.exp(self.log_std)).log_prob(x), dim=-1)
    
    def sample(self, n_samples, context=None):
        return torch.distributions.Normal(self.mean, torch.exp(self.log_std)).sample()
