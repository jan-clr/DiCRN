#!/bin/bash -l
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00

# Prepare the environment
module load cuda/11.8.0
module load python/3.12-conda

cd src || exit 1
conda activate digress

# Constants/Arguments
runname=$1
guidance_target=$2

# Run training
python guidance/train_switches_regressor.py +experiment=regressor_model.yaml dataset=switches general.guidance_target=$guidance_target dataset.reduced_reactions=True dataset.undirected=False
