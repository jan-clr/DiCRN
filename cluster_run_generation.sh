#!/bin/bash -l
#SBATCH --gres=gpu:1
#SBATCH --time=14:00:00

# Prepare the environment
module load cuda/11.8.0
module load python/3.12-conda

cd src || exit 1
conda activate digress

# Constants/Arguments
runname=$1
# Run training
python generate_samples.py +experiment=switches.yaml dataset=switches

