import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import time
import numpy as np
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

# Dummy dataset for random image generation
# Dummy dataset for random image generation
class RandomImageDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples, image_size=(3, 32, 32)):
        self.num_samples = num_samples
        self.image_size = image_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Random image generation
        image = torch.randn(self.image_size)  # Random image
        return image

    @property
    def dataset(self):
        # Return the dataset object itself for compatibility with Experiment
        return self
def create_model(num_blocks=3, block_size=48, sym_recon_grad=False, 
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
            layers.append(Conv1x1(current_size[0]))
            layers.append(Coupling(current_size))

        if split_prior and l < num_blocks - 1:
            layers.append(SplitPrior(current_size, NegativeGaussianLoss))
            current_size = (current_size[0] // 2, current_size[1], current_size[2])

    return FlowSequential(NegativeGaussianLoss(size=current_size), 
                         *layers)

# Function to check GPU memory usage
def get_gpu_memory():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024**3)  # in GB
    return 0

# Main testing function
def main():
    config = {
        'name': '3L-48K Glow Exact Imagenet32',
        'eval_epochs': 1,
        'sample_epochs': 1,
        'log_interval': 100,
        'lr': 1e-3,
        'num_blocks': 1, # 3
        'block_size': 4, # 48
        'batch_size': 64,
        'modified_grad': False,
        'add_recon_grad': False,
        'sym_recon_grad': False,
        'actnorm': True,
        'split_prior': True,
        'activation': 'None',
        'recon_loss_weight': 0.0,
        'sample_true_inv': False,
        'plot_recon': False,
        'grad_clip_norm': 10_000,
        'warmup_epochs': 0
    }

    # Use RandomImageDataset for testing
    train_loader = torch.utils.data.DataLoader(RandomImageDataset(1000), batch_size=config['batch_size'], shuffle=True)
    val_loader = train_loader  # Using train_loader as placeholder for validation
    test_loader = train_loader  # Using train_loader as placeholder for test

    # Create the model and move it to GPU
    model = create_model(num_blocks=config['num_blocks'],
                         block_size=config['block_size'], 
                         sym_recon_grad=config['sym_recon_grad'],
                         actnorm=config['actnorm'],
                         split_prior=config['split_prior'],
                         recon_loss_weight=config['recon_loss_weight']).to('cuda')

    optimizer = optim.Adam(model.parameters(), lr=config['lr'], betas=(0.9, 0.999))
    scheduler = StepLR(optimizer, step_size=1, gamma=1.0)

    # Training loop
    start_time = time.time()
    model.train()

    # Check initial GPU memory usage
    print(f"Initial GPU memory usage: {get_gpu_memory()} GB")

    for epoch in range(config['eval_epochs']):
        running_loss = 0.0
        for i, images in enumerate(train_loader):
            images = images.cuda()  # Move data to GPU

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass: Compute log-likelihood (negative log probability)
            log_prob = model(images)
            loss = -log_prob.mean()  # Negate the log likelihood

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % config['log_interval'] == 0:
                print(f"Epoch [{epoch+1}/{config['eval_epochs']}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

        print(f"Epoch {epoch+1} finished. Average loss: {running_loss / len(train_loader):.4f}")

    # Measure GPU memory usage after training
    print(f"GPU memory usage after training: {get_gpu_memory()} GB")

    # Measure total time taken
    end_time = time.time()
    print(f"Time taken for {config['eval_epochs']} epochs: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()












# import torch
# import torch.optim as optim
# from torch.optim.lr_scheduler import StepLR
# import time
# import numpy as np
# from inf.layers import Dequantization, Normalization
# from inf.layers.distributions.uniform import UniformDistribution
# from inf.layers.splitprior import SplitPrior
# from inf.layers.flowsequential import FlowSequential
# from inf.layers.conv1x1 import Conv1x1
# from inf.layers.actnorm import ActNorm
# from inf.layers.squeeze import Squeeze
# from inf.layers.transforms import LogitTransform
# from inf.layers.coupling import Coupling
# from inf.train.losses import NegativeGaussianLoss
# from inf.train.experiment import Experiment
# from inf.datasets.imagenet import load_data

# # Dummy dataset for random image generation
# class RandomImageDataset(torch.utils.data.Dataset):
#     def __init__(self, num_samples, image_size=(3, 32, 32)):
#         self.num_samples = num_samples
#         self.image_size = image_size

#     def __len__(self):
#         return self.num_samples

#     def __getitem__(self, idx):
#         image = torch.randn(self.image_size)  # Random image
#         label = torch.randint(0, 10, (1,))  # Random label
#         return image, label

# def create_model(num_blocks=3, block_size=48, sym_recon_grad=False, 
#                  actnorm=False, split_prior=False, recon_loss_weight=1.0):
#     current_size = (3, 32, 32)

#     alpha = 1e-6
#     layers = [
#         Dequantization(UniformDistribution(size=current_size)),
#         Normalization(translation=0, scale=256),
#         Normalization(translation=-alpha, scale=1 / (1 - 2 * alpha)),
#         LogitTransform(),
#     ]

#     for l in range(num_blocks):
#         layers.append(Squeeze())
#         current_size = (current_size[0]*4, current_size[1]//2, current_size[2]//2)

#         for k in range(block_size):
#             if actnorm:
#                 layers.append(ActNorm(current_size[0]))
#             layers.append(Conv1x1(current_size[0]))
#             layers.append(Coupling(current_size))

#         if split_prior and l < num_blocks - 1:
#             layers.append(SplitPrior(current_size, NegativeGaussianLoss))
#             current_size = (current_size[0] // 2, current_size[1], current_size[2])

#     return FlowSequential(NegativeGaussianLoss(size=current_size), 
#                          *layers)

# # Function to check GPU memory usage
# def get_gpu_memory():
#     if torch.cuda.is_available():
#         return torch.cuda.memory_allocated() / (1024**3)  # in GB
#     return 0

# # Main testing function
# def main():
#     config = {
#         'name': '3L-48K Glow Exact Imagenet32',
#         'eval_epochs': 1,
#         'sample_epochs': 1,
#         'log_interval': 100,
#         'lr': 1e-3,
#         'num_blocks': 3,
#         'block_size': 48,
#         'batch_size': 64,
#         'modified_grad': False,
#         'add_recon_grad': False,
#         'sym_recon_grad': False,
#         'actnorm': True,
#         'split_prior': True,
#         'activation': 'None',
#         'recon_loss_weight': 0.0,
#         'sample_true_inv': False,
#         'plot_recon': False,
#         'grad_clip_norm': 10_000,
#         'warmup_epochs': 0
#     }

#     # Use RandomImageDataset for testing
#     train_loader = torch.utils.data.DataLoader(RandomImageDataset(1000), batch_size=config['batch_size'], shuffle=True)
#     val_loader = train_loader  # Using train_loader as placeholder for validation
#     test_loader = train_loader  # Using train_loader as placeholder for test

#     # Create the model and move it to GPU
#     model = create_model(num_blocks=config['num_blocks'],
#                          block_size=config['block_size'], 
#                          sym_recon_grad=config['sym_recon_grad'],
#                          actnorm=config['actnorm'],
#                          split_prior=config['split_prior'],
#                          recon_loss_weight=config['recon_loss_weight']).to('cuda')

#     optimizer = optim.Adam(model.parameters(), lr=config['lr'], betas=(0.9, 0.999))
#     scheduler = StepLR(optimizer, step_size=1, gamma=1.0)

#     # Training loop
#     start_time = time.time()
#     model.train()

#     # Check initial GPU memory usage
#     print(f"Initial GPU memory usage: {get_gpu_memory()} GB")

#     for epoch in range(config['eval_epochs']):
#         running_loss = 0.0
#         for i, (images, labels) in enumerate(train_loader):
#             images, labels = images.cuda(), labels.cuda()  # Move data to GPU

#             # Zero the parameter gradients
#             optimizer.zero_grad()

#             # Forward pass
#             outputs = model(images)
#             loss = torch.nn.functional.cross_entropy(outputs, labels)

#             # Backward pass and optimization
#             loss.backward()
#             optimizer.step()

#             running_loss += loss.item()

#             if i % config['log_interval'] == 0:
#                 print(f"Epoch [{epoch+1}/{config['eval_epochs']}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

#         print(f"Epoch {epoch+1} finished. Average loss: {running_loss / len(train_loader):.4f}")

#     # Measure GPU memory usage after training
#     print(f"GPU memory usage after training: {get_gpu_memory()} GB")

#     # Measure total time taken
#     end_time = time.time()
#     print(f"Time taken for {config['eval_epochs']} epochs: {end_time - start_time:.2f} seconds")

# if __name__ == "__main__":
#     main()


































# # import torch
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# import time
# from torch.utils.data import Dataset, DataLoader

# # Define a custom dataset that generates random images
# class RandomImageDataset(Dataset):
#     def __init__(self, num_samples, image_size, num_classes):
#         self.num_samples = num_samples
#         self.image_size = image_size
#         self.num_classes = num_classes

#     def __len__(self):
#         return self.num_samples

#     def __getitem__(self, idx):
#         # Generate a random image of the given size
#         image = torch.from_numpy(np.random.rand(*self.image_size).astype(np.float32))
#         # Generate a random label
#         label = torch.randint(0, self.num_classes, (1,)).item()
#         return image, label

# # Simple Convolutional Neural Network (CNN) for testing
# class SimpleCNN(nn.Module):
#     def __init__(self, num_classes=10):
#         super(SimpleCNN, self).__init__()
#         self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
#         self.fc1 = nn.Linear(64*32*32, 512)
#         self.fc2 = nn.Linear(512, num_classes)

#     def forward(self, x):
#         x = torch.relu(self.conv1(x))
#         x = torch.max_pool2d(x, 2)
#         x = torch.relu(self.conv2(x))
#         x = torch.max_pool2d(x, 2)
#         x = x.view(x.size(0), -1)  # Flatten
#         x = torch.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x

# # Function to check GPU memory usage
# def get_gpu_memory():
#     if torch.cuda.is_available():
#         return torch.cuda.memory_allocated() / (1024**3)  # in GB
#     return 0

# # Main function to test speed and memory
# def test_model_on_random_data(num_samples=1000, image_size=(3, 64, 64), num_classes=10, batch_size=64, epochs=1):
#     # Create dataset and dataloader
#     dataset = RandomImageDataset(num_samples, image_size, num_classes)
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#     # Initialize model, optimizer, and loss function
#     model = SimpleCNN(num_classes=num_classes).cuda()  # Move model to GPU
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.001)

#     # Measure initial GPU memory usage
#     print(f"Initial GPU memory usage: {get_gpu_memory()} GB")

#     # Track time for one epoch
#     start_time = time.time()

#     # Loop through the dataset for training
#     model.train()
#     for epoch in range(epochs):
#         running_loss = 0.0
#         for i, (images, labels) in enumerate(dataloader):
#             images, labels = images.cuda(), labels.cuda()  # Move data to GPU

#             # Zero the parameter gradients
#             optimizer.zero_grad()

#             # Forward pass
#             outputs = model(images)
#             loss = criterion(outputs, labels)

#             # Backward pass and optimization
#             loss.backward()
#             optimizer.step()

#             running_loss += loss.item()

#             if i % 100 == 0:  # Print loss every 100 batches
#                 print(f"Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item():.4f}")

#         print(f"Epoch {epoch+1} finished. Average loss: {running_loss / len(dataloader):.4f}")

#     # Measure GPU memory after training
#     print(f"GPU memory usage after training: {get_gpu_memory()} GB")

#     # Measure time taken for the epoch
#     end_time = time.time()
#     print(f"Time taken for {epochs} epochs: {end_time - start_time:.2f} seconds")

# # Run the test with random data
# test_model_on_random_data(num_samples=1000, image_size=(3, 64, 64), batch_size=64, epochs=1)
