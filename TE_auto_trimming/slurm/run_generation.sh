#!/bin/bash

#SBATCH -o agp_datos.out
#SBATCH -e agp_datos.err
#SBATCH --mail-type END
#SBATCH --mail-user agonzalezpalo@uoc.edu
#SBATCH -J agp_datos
#SBATCH --time 1-00:00:00
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --mem 200GB

/shared/ifbstor1/projects/tagua_gen_ec/anaconda3_homesimon/envs/auto_trimming_agp/bin/python3.9 GenerationData.py --fasta ./r.1.5_all.fasta --seq_per_case 30
