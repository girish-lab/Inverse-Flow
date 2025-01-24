import torch
from torch.utils.data import DataLoader
from torch import optim
from torch.optim.lr_scheduler import StepLR

from inf.layers.flowsequential import FlowSequential
from inf.layers.selfnorm import SelfNormConv, SelfNormFC
from inf.train.losses import NegativeGaussianLoss
from inf.train.experiment import Experiment
from inf.layers.inv_conv import inv_flow_with_pad, inv_flow_no_pad        # This is the important part for Invertible Convolution (Inv_flow)


def create_model(data_size, layer='conv'):
    layers = []

    c_in = data_size[0]
    h = data_size[1]
    w = data_size[2]

    if layer == 'fc':
        size = c_in * h * w
        # layers.append(SelfNormFC(size, size, bias=True,
        #                          sym_recon_grad=False, 
        #                          only_R_recon=False))
        # layers.append(inv_flow_with_pad(c_in, c_in, (3,3)))
        layers.append(inv_flow_with_pad(c_in, c_in, (3, 3), order='TR'))
        # layers.append(inv_flow_no_pad(c_in, c_in, (3, 3)))
        layers.append( inv_flow_with_pad(c_in, c_in, (3, 3), order='TR',))
        layers.append( inv_flow_with_pad(c_in, c_in, (3, 3), order='BL',))
        layers.append( inv_flow_with_pad(c_in, c_in, (3, 3), order='BR',))

        model = FlowSequential(NegativeGaussianLoss(size=(size,)), *layers)
    
    elif layer == 'conv':
        # layers.append(SelfNormConv(c_in, c_in, (3,3), bias=True,
        #                          stride=1, padding=1,
        #                          sym_recon_grad=False, 
        #                          only_R_recon=False))  
        # layers.append(inv_flow_with_pad(c_in, c_in, (3,3)))
        # layers.append(inv_flow_no_pad(current_size[0], current_size[0], (3, 3)))
        layers.append(inv_flow_with_pad(c_in, c_in, (3, 3), order='TR'))
        # layers.append(inv_flow_no_pad(c_in, c_in, (3, 3)))
        layers.append( inv_flow_with_pad(c_in, c_in, (3, 3), order='TR',))
        layers.append( inv_flow_with_pad(c_in, c_in, (3, 3), order='BL',))
        layers.append( inv_flow_with_pad(c_in, c_in, (3, 3), order='BR',))

        model = FlowSequential(NegativeGaussianLoss(size=data_size), *layers)

    return model


def load_data(batch_size=100, im_size=(1,28,28), n_train=60_000, n_val=10_0000, n_test=10_000):
    trainx = torch.randn(n_train, *im_size)
    testx = torch.randn(n_test, *im_size)

    trainy = torch.zeros(n_train)
    testy = torch.zeros(n_test) 

    trainvalset = torch.utils.data.TensorDataset(trainx, trainy)
    testset = torch.utils.data.TensorDataset(testx, testy)

    trainset = torch.utils.data.Subset(trainvalset, range(0, n_train - n_val))
    valset = torch.utils.data.Subset(trainvalset, range(n_train - n_val, n_train))

    train_loader = DataLoader(trainset, batch_size=batch_size)
    val_loader = DataLoader(valset, batch_size=batch_size)
    test_loader = DataLoader(testset, batch_size=batch_size)

    return train_loader, val_loader, test_loader

def run_timing_experiment(name, snf_name, config, sz, m, results):    
    train_loader, val_loader, test_loader = load_data(batch_size=config['batch_size'], im_size=sz,
                                                      n_train=50_000, n_val=100, n_test=100)
    model = create_model(data_size=sz, layer=m).to('cuda')
    print(model)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print('Total_params Inv_flow , :', pytorch_total_params)
    # wandb.log({"total_params": pytorch_total_params})
    print('config:', config)

    optimizer = optim.Adam(model.parameters(), lr=config['lr'], betas=(0.9, 0.999))
    scheduler = StepLR(optimizer, step_size=1, gamma=1.0)

    experiment = Experiment(model, train_loader, val_loader, test_loader,
                            optimizer, scheduler, **config)
    experiment.run()
    mean_time = experiment.summary['Batch Time Mean']
    std_time = experiment.summary['Batch Time Std']

    print(f"{name}: {mean_time} +/- {std_time}")

    results[f'{m} {snf_name}']['n_params'].append(sz[0] * sz[1] * sz[2])
    results[f'{m} {snf_name}']['mean'].append(mean_time)
    results[f'{m} {snf_name}']['std'].append(std_time)  

    return results

def main():
    image_sizes = [(1, x*32, 1) for x in range(1, 130, 3)]
    model_type = ['conv'] # ['fc', 'conv']
    self_normalized = [True, False]

    name = 'Timing Experiment '

    results = {}
    for m in model_type:
        for snf in self_normalized:
            if snf:
                snf_name = 'IF'
            else:
                snf_name = 'Reg'

            results[f'{m} {snf_name}'] = {
                'n_params': [],
                'mean': [],
                'std': []
            }

            for sz in image_sizes:
                name = f'Timing Experiment {m} {snf_name} {sz}'

                config = {
                    'name': name,
                    'eval_epochs': 1,
                    'sample_epochs': 1000,
                    'log_interval': 10000,
                    'lr': 1e-4,
                    'batch_size': 128,
                    'modified_grad': snf,
                    'add_recon_grad': snf,
                    'sym_recon_grad': False,
                    'only_R_recon': False,
                    'actnorm': False,
                    'split_prior': True,
                    'activation': 'None',
                    'log_timing': True,
                    'epochs': 10,
                    'log_timing': True,
                    'wandb': True,
                    'wandb_project': 'inv_flow_timescaling',
                    'wandb_entity': 'carlobob031',
                    'grad_clip_value': 0.001,
                    'grad_clip_norm': 0.1,
                    'grad_clip': False,

                }
                results = run_timing_experiment(name, snf_name, config, sz, m, results)
                print(results[f'{m} {snf_name}'])
            print(results)
        print(results)
    print(results)