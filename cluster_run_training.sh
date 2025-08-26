#!/bin/bash -l
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00

# Prepare the environment
module load cuda/11.8.0
module load python/conda-3.12

cd src || exit 1
conda activate digress

# Constants/Arguments
outdir=$WORK/gnn-generation/runs
runname=$1
# Run training
python main.py +experiment=switches.yaml dataset=switches

