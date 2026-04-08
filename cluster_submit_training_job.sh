#!/bin/bash -l

cd $HOME/repos/DiCRN

mkdir -p logs

timestamp=$(date +%Y%m%d%H%M%S)


export http_proxy=http://proxy:80
export https_proxy=http://proxy:80

batch_size=256
name=switches_reduced
reduced_reactions=True
runname="${name}_training_${timestamp}"

sbatch.tinygpu --job-name=$runname --output=logs/$runname.log --mail-user='mail@jan-claar.de' --mail-type=ALL cluster_run_training.sh $batch_size $name $reduced_reactions

batch_size=128
name=switches_full
reduced_reactions=False
runname="${name}_training_${timestamp}"

sbatch.tinygpu --job-name=$runname --output=logs/$runname.log --mail-user='mail@jan-claar.de' --mail-type=ALL cluster_run_training.sh $batch_size $name $reduced_reactions
