#!/bin/bash
#SBATCH --job-name=mavt
#SBATCH --output=logs/test_%j.txt
#SBATCH --error=logs/err_%j.txt
#SBATCH --gres=gpu:1
#SBATCH --time=14-00:00:00             # đặt tối đa được phép ở cụm bạn
#SBATCH --requeue
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
nvidia-smi