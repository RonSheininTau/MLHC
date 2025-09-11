#! /bin/sh

#SBATCH --job-name=inet
#SBATCH --output=slurm/ablation_%A.out
#SBATCH --error=slurm/ablation_%A.err
#SBATCH --partition=gpu-roded
#SBATCH --time=5-00:00:00
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1


cd /home/bnet/kupershmidt/HML
source ~/.bashrc
conda activate spider

export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1


python3 ablation_study.py



