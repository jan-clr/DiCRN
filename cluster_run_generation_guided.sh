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
target_value=$2
experiment=${3}

echo $experiment
echo $runname
echo $target_value

# Run training
python guidance/main_guidance.py +experiment=$experiment dataset=switches "general.target_value=$target_value"

