#!/bin/bash

#SBATCH -o test_model.out
#SBATCH -e test_model.err
#SBATCH --mail-type END
#SBATCH --mail-user agonzalezpalo@uoc.edu
#SBATCH -J test_model
#SBATCH --time 1-00:00:00
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --mem 200GB

module load tensorflow-gpu/2.6.2

/shared/ifbstor1/projects/tagua_gen_ec/anaconda3_homesimon/envs/autotrim_env/bin/python3.9 \
    Auto_trimming.py \
    --mode test \
    --model ./models/model_trained_9218/trained_model.h5 \
    --scaler ./models/model_trained_9218/scalerX.bin \
    --dataset_testing ./datasets/dataset_testing
