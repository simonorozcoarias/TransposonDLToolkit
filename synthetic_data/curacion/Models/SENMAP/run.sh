#!/bin/bash

#SBATCH -o inpactor2.out
#SBATCH -e inpactor2.err
#SBATCH --mail-type END
#SBATCH --mail-user msuevos@uoc.edu
#SBATCH -J inpactor2
#SBATCH --time 3-00:00:00
#SBATCH --partition gpu
#SBATCH -n 2
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --mem 200GB


source /shared/home/sorozcoarias/anaconda3/bin/activate Inpactor2
/shared/home/sorozcoarias/anaconda3/envs/Inpactor2/bin/python3 Inpactor2.py -f ../../datasynth/results/combined_dataset.fasta -o inpactor2_results