import sys
import argparse
from inf.experiments import (selfnorm_fc_mnist, exact_fc_mnist,
    selfnorm_cnn_mnist, exact_cnn_mnist,
    emerging_cnn_mnist, exponential_cnn_mnist,
    selfnorm_glow_mnist, conv1x1_glow_mnist, 
    selfnorm_glow_cifar, conv1x1_glow_cifar,
    selfnorm_glow_imagenet, conv1x1_glow_imagenet,
    geco_selfnorm_glow_mnist,
    snf_timescaling,
    if_cnn_mnist, if_exact_cnn_mnist, if_conv1x1_glow_mnist, if_glow_mnist, if_glow_cifar,
    ff_glow_mnist, ff_glow_cifar, if_timescaling,
    if_glow_imagenet32)

parser = argparse.ArgumentParser(description='')
parser.add_argument('--name', type=str, help='experiment name')

def main():
    args = parser.parse_args()
    module_name = 'inf.experiments.{}'.format(args.name)
    experiment = sys.modules[module_name]
    experiment.main()
