#!/bin/bash -l

cd $HOME/repos/DiCRN

mkdir -p logs

timestamp=$(date +%Y%m%d%H%M%S)

export http_proxy=http://proxy:80
export https_proxy=http://proxy:80

#runname="guidance_binary_${timestamp}"
#target_value=1.0
#experiment="guidance_binary_propensity_cluster.yaml"
#
#sbatch.tinygpu --job-name=$runname --output=logs/$runname.log --mail-user='mail@jan-claar.de' --mail-type=ALL cluster_run_generation_guided.sh $runname "$target_value" "$experiment"
#
#runname="guidance_cube_${timestamp}"
#target_value=1.0
#experiment="guidance_cube_propensity_cluster.yaml"
#
#sbatch.tinygpu --job-name=$runname --output=logs/$runname.log --mail-user='mail@jan-claar.de' --mail-type=ALL cluster_run_generation_guided.sh $runname "$target_value" "$experiment"
#
#runname="guidance_nr_species_${timestamp}"
#target_value=9.0
#experiment="guidance_nr_species_cluster.yaml"
#
#sbatch.tinygpu --job-name=$runname --output=logs/$runname.log --mail-user='mail@jan-claar.de' --mail-type=ALL cluster_run_generation_guided.sh $runname "$target_value" "$experiment"
#
#runname="guidance_avg_degree_${timestamp}"
#target_value=4.0
#experiment="guidance_avg_degree_cluster.yaml"
#
#sbatch.tinygpu --job-name=$runname --output=logs/$runname.log --mail-user='mail@jan-claar.de' --mail-type=ALL cluster_run_generation_guided.sh $runname "$target_value" "$experiment"
#
runname="guidance_species_degree_${timestamp}"
target_value='[5.0, 4.0]'
experiment="guidance_species_degree_cluster.yaml"

sbatch.tinygpu --job-name=$runname --output=logs/$runname.log --mail-user='mail@jan-claar.de' --mail-type=ALL cluster_run_generation_guided.sh $runname "$target_value" "$experiment"
