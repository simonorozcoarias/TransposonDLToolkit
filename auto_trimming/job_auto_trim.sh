#!/bin/bash

#SBATCH -o auto_trim.out
#SBATCH -e auto_trim.err
#SBATCH --mail-type END
#SBATCH --mail-user simon.orozco.arias@gmail.com
#SBATCH -J auto_trim
#SBATCH --time 3-00:00:00
#SBATCH --partition gpu
#SBATCH --nodelist=gpu-node-03
#SBATCH -n 2
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --mem 200GB

module load tensorflow-gpu/2.6.2
source /shared/home/sorozcoarias/anaconda3/bin/activate gpu

~/anaconda3/envs/gpu/bin/python3 NN_trainingV2.py train caso_todos/features_data.npy caso_todos/labels_data.npy
~/anaconda3/envs/gpu/bin/python3 NN_trainingV2.py test trained_model.h5 scalerX.bin X_test.npy Y_test.npy