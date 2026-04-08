#!/bin/bash -l
#SBATCH --gres=gpu:1
#SBATCH --time=14:00:00

# Prepare the environment
module load cuda/11.8.0
module load python/3.12-conda

cd src || exit 1
conda activate digress

batch_size=$1
name=$2
reduced_reactions=$3

# Constants/Arguments
# Run training
#python main.py +experiment=switches.yaml dataset=switches dataset.reduced_reactions=True dataset.undirected=True general.name="switches_undirected" train.n_epochs=150
python main.py +experiment=switches.yaml dataset=switches dataset.reduced_reactions=${reduced_reactions} dataset.undirected=False general.name=${name} train.batch_size=${batch_size} train.n_epochs=150

