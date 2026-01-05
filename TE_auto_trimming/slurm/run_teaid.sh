#!/bin/bash

#SBATCH -o run_teaid.out
#SBATCH -e run_teaid.err
#SBATCH --mail-type END
#SBATCH --mail-user agonzalezpalo@uoc.edu
#SBATCH -J run_teaid
#SBATCH --time 1-00:00:00
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --mem 200GB

source /shared/home/sorozcoarias/anaconda3/bin/activate autotrim_env

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

/shared/ifbstor1/projects/tagua_gen_ec/anaconda3_homesimon/envs/teaid_agp/bin/python \
    Auto_trimming.py \
    --mode teaid \
    --input_fasta ./data_generation/simulated_data_merged.fasta 
