#!/bin/bash
#SBATCH -A research
#SBATCH --job-name=inv_flow
#SBATCH -n 20
#SBATCH --gres=gpu:2
#SBATCH --mem-per-cpu=3000
#SBATCH --output=output_files/inv_flow_MNIST_%j.txt       # Output file.
#SBATCH --mail-type=END                # Enable email about job finish
#SBATCH --mail-user=sandeep.nagar@research.iiit.ac.in    # Where to send mail
#SBATCH --time=4-00:00:00

# module load cudnn/8-cuda-10.2
# module load cuda/8.0
# module load cudnn/7-cuda-8.0

# module load cudnn/7.6-cuda-11.2
# source activate inv_flow 		# to activate the conda virtual environment 
# source venv/bin/activate # to activate the python virtual environment

# rm -r /scratch/sandeep.nagar
# mkdir /scratch/sandeep.nagar
# rsync -aP sandeep.nagar@ada.iiit.ac.in:/share1/sandeep.nagar/celeba/train.tar /scratch/sandeep.nagar
# rsync -aP sandeep.nagar@ada.iiit.ac.in:/share1/sandeep.nagar/celeba/validation.tar /scratch/sandeep.nagar

# scp ada:/share1/$aditya.kalappa/train /scratch
# comment # cinc_cuda_level1
# fastflow/layers/conv

# cd fastflow2
echo "Training the inv_flow model on MNIST"
python inv_flow_mnist_multiGPUs.py
# python fastflow_cifar.py

# /run/user/1000/gvfs/sftp:host=ada.iiit.ac.in,user=sandeep.nagar/home/sandeep.nagar/inv_flow_ff/inv_flow/inv_flow_cifar_multiGPU.py
