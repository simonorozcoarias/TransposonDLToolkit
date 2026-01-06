#!/bin/bash

#SBATCH -o synth_curation.out
#SBATCH -e synth_curation.err
#SBATCH --mail-type=END
#SBATCH --mail-user=msuevos@uoc.edu
#SBATCH -J synth_curation.err
#SBATCH --partition=long
#SBATCH -n 60
#SBATCH -N 1
#SBATCH --mem=350GB
#SBATCH --export=NONE

source /shared/home/sorozcoarias/anaconda3/bin/activate curation_synth
/shared/home/sorozcoarias/anaconda3/envs/curation_synth/bin/python3 main.py --fasta ../../Clasificacion/r.1.5_all.fasta --output_dir results
