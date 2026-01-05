#!/bin/bash

#SBATCH -o create_dataset.out
#SBATCH -e create_dataset.err
#SBATCH --mail-type END
#SBATCH --mail-user agonzalezpalo@uoc.edu
#SBATCH -J create_dataset
#SBATCH --time 1-00:00:00
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --mem 200GB

source /shared/home/sorozcoarias/anaconda3/bin/activate autotrim_env

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

/shared/ifbstor1/projects/tagua_gen_ec/anaconda3_homesimon/envs/autotrim_env/bin/python \
    Auto_trimming.py \
    --mode dataset \
    --input_fasta ./data_generation/simulated_data_merged.fasta \
    --dataset_dir dataset_autotrim
