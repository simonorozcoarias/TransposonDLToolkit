#!/bin/bash

#SBATCH -o train_model.out
#SBATCH -e train_model.err
#SBATCH --mail-type END
#SBATCH --mail-user agonzalezpalo@uoc.edu
#SBATCH -J train_model
#SBATCH --time 1-00:00:00
#SBATCH --partition gpu
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --mem 400GB

module load tensorflow-gpu/2.6.2

/shared/ifbstor1/projects/tagua_gen_ec/anaconda3_homesimon/envs/teaid_agp/bin/python3.9 \
    Auto_trimming.py \
    --mode train \
    --input_fasta ./data_generation/simulated_data_merged.fasta \
    --dataset_dir ./new_dataset
