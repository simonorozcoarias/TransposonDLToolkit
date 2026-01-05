#!/bin/bash

#SBATCH -o auto_trimming.out
#SBATCH -e auto_trimming.err
#SBATCH --mail-type END
#SBATCH --mail-user agonzalezpalo@uoc.edu
#SBATCH -J auto_trimming
#SBATCH --time 1-00:00:00
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --mem 200GB

module load tensorflow-gpu/2.6.2

/shared/ifbstor1/projects/tagua_gen_ec/anaconda3_homesimon/envs/autotrim_env/bin/python3.9 \
    Auto_trimming.py \
    --mode trimming \
    --input_fasta ./data_generation/simulated_data_merged.fasta \
    --model ./models/model_trained_9218/trained_model.h5 \
    --scaler ./models/model_trained_9218/scalerX.bin \
    --dataset_dir ./datasets/dataset_autotrim
