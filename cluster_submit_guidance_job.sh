#!/bin/bash -l

cd $HOME/repos/DiCRN

mkdir -p logs

timestamp=$(date +%Y%m%d%H%M%S)

runname="dicrn_guidance_${timestamp}"

export http_proxy=http://proxy:80
export https_proxy=http://proxy:80

sbatch.tinygpu --job-name=$runname --output=logs/$runname.log --mail-user='mail@jan-claar.de' --mail-type=ALL cluster_run_generation_guided.sh $runname
