#!/bin/bash -l

cd $HOME/repos/DiCRN

mkdir -p logs

timestamp=$(date +%Y%m%d%H%M%S)

runname="digress_${timestamp}"

sbatch.tinygpu --job-name=$runname --output=logs/$runname.log --mail-user='mail@jan-claar.de' --mail-type=ALL cluster_run_training.sh $runname
