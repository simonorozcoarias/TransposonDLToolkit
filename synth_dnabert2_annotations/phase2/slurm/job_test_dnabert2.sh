#!/bin/bash
#SBATCH -o test_dnabert2.out
#SBATCH -e test_dnabert2.err
#SBATCH --mail-type END
#SBATCH --mail-user jgilbaja@uoc.edu
#SBATCH -J test_dnabert2
#SBATCH --time 3-00:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --account=inpactor3

module load cuda-toolkit/12.9.1
source ~/anaconda3/bin/activate DNABERT2

stdbuf -oL -eL python3 -u ~/tagua_gen_ec/TransposonDLToolkit/auto_detection/phase2/scripts/test_dnabert2_installation.py
