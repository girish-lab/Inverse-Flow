# [Inverse-Flow: Parallel Backpropagation for Inverse of a Convolution with Application to Normalizing Flows](https://openreview.net/forum?id=9e0yjZevPe)

Paper: [arxiv](https://arxiv.org/abs/2410.14634)

Project Page: [Inverse-Flow](https://naagar.github.io/projectPage_InverseFlow/)
## Getting Started
#### Install requirements with Anaconda:
`conda env create -f conda_environment.yml`

Ensure you have GCC 9 or later installed. You can install GCC 9 on Ubuntu using the following commands: 

        sudo apt update
        sudo apt install gcc-9 g++-9
        sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 60 --slave /usr/bin/g++ g++ /usr/bin/g++-9
        sudo update-alternatives --config gcc
#### Install inf package
Install the inf package locally for development. This allows you to run experiments with the `inf` command. At the root of the project directory run:
`pip install -e .`

Note that this package requires Ninja to build the C++ extensions required to efficiently compute the Self-Normalizing Flow gradient, however this should be installed automatically with pytorch.

#### (Optional) Setup Weights & Biases:
This repository uses Weight & Biases for experiment tracking. By deafult this is set to off. However, if you would like to use this (highly recommended!) functionality, all you have to do is install weight & biases as follows, and then set `'wandb': True`,  `'wandb_project': YOUR_PROJECT_NAME`, and `'wandb_entity': YOUR_ENTITY_NAME` in the default experiment config at the top of inf/train/experiment.py.

To install Weights & Biases follow the [quickstart guide here](https://docs.wandb.com/quickstart), or, simply run `pip install wandb` (with the pip associated to your conda environment), followed by: `wandb login`

Make sure that you have first [created a weights and biases account](https://app.wandb.ai/login?signup=true), and filled in the correct `wandb_project` and `wandb_entity` in the experiment config.

## Running an experiment
<!-- To rerun the experiments from table 2, you can run the following commands: -->
<!-- - `inf --name 'selfnorm_fc_mnist'` -->
<!-- - `inf --name 'exact_fc_mnist'` -->
<!-- - `inf --name 'selfnorm_cnn_mnist'` -->
<!-- - `inf --name 'exact_cnn_mnist'` -->
<!-- - `inf --name 'emerging_cnn_mnist'` -->
<!-- - `inf --name 'exponential_cnn_mnist'` -->
<!-- - `inf --name 'conv1x1_glow_mnist'` -->
<!-- - `inf --name 'selfnorm_glow_mnist'` -->
<!-- - `inf --name 'if_cnn_mnist'` -->
<!-- - `inf --name 'if_exact_cnn_mnist'` -->


To rerun the experiments from table 2:
<!-- - `inf --name 'if_conv1x1_glow_cifar'` -->
- `inf --name 'emerging_cnn_mnist'`
<!-- - `inf --name 'emerging_cnn_mnist'` -->
- `inf --name 'ff_glow_mnist'`
- `inf --name 'selfnorm_glow_mnist'`
- `inf --name 'if_glow_mnist'`

<!-- - `inf --name 'if_selfnorm_glow_cifar'`
- `inf --name 'if_conv1x1_glow_imagenet'`
- `inf --name 'if_glow_imagenet'` -->
<!-- - `inf --name 'selfnorm_cnn_mnist'` -->

To rerun the experiments from table 3, you can run the following commands after setting L=2, K=16 (except Mintnet):
<!-- - `inf --name 'emerging_cnn_mnist'` -->
- `inf --name 'selfnorm_glow_mnist'`
- `inf --name 'ff_glow_mnist'`
- `inf --name 'emerging_cnn_mnist'`
- `inf --name 'if_glow_mnist'`

To rerun the experiments from table 4, you can run the following commands, first set the number of layers:
- `inf --name 'exact_fc_mnist'` 
- `inf --name 'exponential_cnn_mnist'`
- `inf --name 'emerging_cnn_mnist'`
- `inf --name 'selfnorm_cnn_mnist'`
- `inf --name 'if_cnn_mnist'`

<!-- To recreate the results of figure 4, run:
- `inf --name 'inf_timescaling'` -->

<!-- To recreate the results of Table A.5 (using improved constrained optimization), run:
- `inf --name 'geco_selfnorm_glow_mnist` -->

## Basics of the framework
- All models are built using the `FlowSequential` module (see inf/layers/flowsequential.py)
    - This module iterates through a list of `FlowLayer` or `ModifiedGradFlowLayer` modules, repeatedly transforming the input, while simultaneously accumulating the log-determinant of the jacobian of each transformation along the way.
    - Ultimately, this layer returns the total normalized log-probability of input by summing the log-probability of the transformed input under the base distribuiton, and the accumulated sum of log jacobian determinants (i.e. using the change of variables rule).
- The `Experiment` class (see inf/train/experiment.py) handles running the training iterations, evaluation, sampling, and model saving & loading.
- All experiments can be found in `inf/experiments/`, and require the specification of a model, optimizer, dataset, and config dictionary. See below for the currently implemented options for the config.  
- All layer implementations can be found in `inf/layers/` including the self-normalizing layer found at inf/layers/selfnorm.py

## Overview of Config options
The configuration dictionary is mainly used to modify the training procedure specified in the Experiment class, but it can also be used to modify the model architecture if desired. A non-exhaustive list of config options and descriptions are given below, note that config options which modify model architecture may not have any effect if they are not explicitly incorporated in the `create_model()` function of the experiments. The default configuration options are specified at the top of the experiment file. The configuration dictionary passed to the Experiment class initalizer then overwrites these defaults if the key is present.


#### Training Options (important first)
- `'modified_grad'`: *bool*, if True, use the self-normalized gradient in place of the true exact gradient. This also causes self-normlizing flow layers to return 0 for their log-jacobian-determinant during training. During evaluation, this option has no effect, and log-likelihoods are computed exactly.  
- `'add_recon_grad'`: *bool*, if True, for all layers which inherit from `SelfNormConv` (includes `SelfNormFC`), compute the layer-wise reconstruction loss and add it to the standard gradient.
- `'lr'`: *float*, learning rate
- `'epochs'`: *int*, total training epochs
- `'eval_epochs'`: *int*, number of epochs between computing true log-likelihood on validation set.
- `'eval_train'`:  *bool*, if True, also compute true-log-likelihood on train set during eval phase.
- `'max_eval_ex'`: *int*, maximum number of examples to run evaluation on (default `inf`). Useful for models with extremely computationally expensive inference procedures. 
- `'warmup_epochs'`: *int*, number of epochs over which to linearly increase learning rate from $0$ to `lr`.
- `'batch_size'`: *int*, number of samples per batch
- `'recon_loss_weight'`: *float*, Value of $\lambda$ weight on reconstruction gradient. 
- `'sym_recon_grad'`: Bool, if True, and `add_recon_grad` is true, use a symmetric version of the reconstruction loss for increased stability.
- `'grad_clip_norm'`: *float*, maximum magnitude which to scale gradients to if greater than.

#### Model Architecutre Options
- `'activation'`: *str*: Name of activation function to use for FC and CNN models (one of `SLR, LLR, Spline, SELU`).
- `'num_blocks'`: *int*, Number of blocks for glow-like models
- `'block_size'`: *int*, Number of setps-of-flow per block in glow-like models
- `'num_layers'`: *int*, Number of layers for CNN and FC models
- `'actnorm'`: *bool*, if True, use ActNorm in glow-like models
- `'split_prior'`: *bool*, if True, use Split Priors between each block in glow-like models

#### Logging Options
- `'wandb'`: *bool*, if True, use weights & biases logging
- `'wandb_project'`: *str*, Name of weights & biases project to log to.
- `'wandb_entity'`: *str*, username or team name for weights and biases.
- `'name'`: *str*, experiment name file-saving, and for weights & biases
- `'notes'`: *str*, experiment notes for weights & biases
- `'sample_epochs'`: *int*, epochs between generating samples.
- `'n_samples'`: *int*, number of samples to generate 
- `'sample_dir'`: *str*, directory to save samples
- `'checkpoint_path'`: *str*, path of saved checkpoints (at best validation likelihood). If unspecified, and using weights and biases, this defaults to `checkpoint.tar` in the WandB run directory. If not using weights and biases, this defaults to `f"./{str(self.config['name']).replace(' ', '_')}_checkpoint.tar"`.
- `'log_interval'`: *int*, number of batches between printing training loss.
- `'verbose'`: *bool*, if True, log the log-jacobian-determinant and reconstruction loss per layer separately to weights and biases.
- `'sample_true_inv'`: *bool*, if True, generate samples from the true inverse of a self-normalizing flow model, in addition to samples from the approximate (fast) inverse.
- `'plot_recon'`: *bool*, if True, plot reconstruction of training images.
- `'log_timing'`: *bool*, if True, compute mean and std. of time per batch and time per sample. Print to screen and save as summary statistic of experiment.

## Acknowledgements
iHub-IIITH is acknowledged for financial support.

## Citation: 

        @InProceedings{nagar2025inverseflow,
        title = {Parallel Backpropagation for Inverse of a Convolution with Application to Normalizing Flows},
        author = Nagar, Sandeep and Varma, Girish}
        booktitle = {Proceedings of The 28th International Conference on Artificial Intelligence and Statistics},
        year = {2025},
        volume = {239},
        series = {Proceedings of Machine Learning Research},
        month = {03--05 May},
        publisher = {PMLR},
        pdf = {https://arxiv.org/pdf/2410.14634},
        url = {https://arxiv.org/abs/2410.14634},
      }
