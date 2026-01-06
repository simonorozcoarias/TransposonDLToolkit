#!/bin/bash
#SBATCH -o simple_eval_limits.out
#SBATCH -e simple_eval_limits.err
#SBATCH --mail-type END
#SBATCH --mail-user msuevos@uoc.edu
#SBATCH -J simple_eval_limits
#SBATCH --time 2-00:00:00
#SBATCH --partition gpu
#SBATCH -n 2
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --mem 200GB

source /shared/home/sorozcoarias/anaconda3/bin/activate clasificacion_sintetico
/shared/home/sorozcoarias/anaconda3/envs/clasificacion_sintetico/bin/python3 new_evaluate_augmentation_limit.py --model_name wgan_c_generator_best_600_100_32.weights.h5 --data_file ../../../r.1.5_all.fasta


