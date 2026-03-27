#!/bin/bash -l

cd $HOME/repos/DiCRN

mkdir -p logs

export http_proxy=http://proxy:80
export https_proxy=http://proxy:80

timestamp=$(date +%Y%m%d%H%M%S)
runname="digress_regressor_${timestamp}"
guidance_target="nr_species"

sbatch.tinygpu --job-name=$runname --output=logs/$runname.log --mail-user='mail@jan-claar.de' --mail-type=ALL cluster_run_regressor_training.sh $runname "$guidance_target"

timestamp=$(date +%Y%m%d%H%M%S)
runname="digress_regressor_${timestamp}"
guidance_target="avg_degree"

sbatch.tinygpu --job-name=$runname --output=logs/$runname.log --mail-user='mail@jan-claar.de' --mail-type=ALL cluster_run_regressor_training.sh $runname "$guidance_target"

timestamp=$(date +%Y%m%d%H%M%S)
runname="digress_regressor_${timestamp}"
guidance_target="species+degree"

sbatch.tinygpu --job-name=$runname --output=logs/$runname.log --mail-user='mail@jan-claar.de' --mail-type=ALL cluster_run_regressor_training.sh $runname "$guidance_target"

#timestamp=$(date +%Y%m%d%H%M%S)
#runname="digress_regressor_${timestamp}"
#guidance_target="binary_propensity"
#
#sbatch.tinygpu --job-name=$runname --output=logs/$runname.log --mail-user='mail@jan-claar.de' --mail-type=ALL cluster_run_regressor_training.sh $runname "$guidance_target"
