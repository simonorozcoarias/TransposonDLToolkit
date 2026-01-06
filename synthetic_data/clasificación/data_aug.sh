  GNU nano 7.2                                job_train.sh                                          
#!/bin/bash
#SBATCH -o dataaug.out
#SBATCH -e dataaug.err
#SBATCH --mail-type END
#SBATCH --mail-user msuevos@uoc.edu
#SBATCH -J dataaug

source /shared/home/sorozcoarias/anaconda3/bin/activate clasificacion_sintetico
/shared/home/sorozcoarias/anaconda3/envs/clasificacion_sintetico/bin/python3 data_aug.py --fasta_file ../../../r.1.5_all.fasta
