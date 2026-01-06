#!/bin/bash

#SBATCH -o imperf_inpactor2.out
#SBATCH -e imperf_inpactor2.err
#SBATCH --mail-type END
#SBATCH --mail-user msuevos@uoc.edu
#SBATCH -J imperf_inpactor2
#SBATCH --time 3-00:00:00
#SBATCH --partition gpu
#SBATCH -n 2
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --mem 300GB

source /shared/home/sorozcoarias/anaconda3/bin/activate Inpactor2
/shared/home/sorozcoarias/anaconda3/envs/Inpactor2/bin/python3 inpacto_model.py train --input ../../new_datasynth/imperfection_out/combined_dataset.fasta --output imperfection_output
