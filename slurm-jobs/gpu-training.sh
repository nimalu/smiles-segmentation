#!/bin/bash
#SBATCH --job-name=smiles-gpu-training
#SBATCH --partition=gpu_a100_short
#SBATCH --nodes=1    
#SBATCH --time=00:30:00              
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=niklas.lutze@student.kit.edu
#SBATCH --output=slurm-logs/gpu_training_%j.log


uv run python segmentation/02_train.py 
