#!/bin/bash

#SBATCH -o outputs/build_species_index.out
#SBATCH -e outputs/build_species_index.err
#SBATCH --mail-type END
#SBATCH --mail-user jgilbaja@uoc.edu
#SBATCH -J build_species_index
#SBATCH --time 0-04:00:00
#SBATCH --partition fast
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --mem 16GB

# Cargar módulos necesarios si es requerido por el cluster
# (Descomentar si tu cluster requiere cargar Python/Anaconda)
# module load anaconda3

# Directorio de trabajo
cd ~/tagua_gen_ec/TransposonDLToolkit/auto_detection/phase1/scripts

# Ejecutar script con output sin buffering
stdbuf -oL -eL python3 -u build_species_index.py \
    --input ../../datasets/r.1.5_all.fasta \
    --species-gc ../results/species_gc_data_v2.csv \
    --output ../results/species_index_v2.json

echo "✅ Índice de especies generado correctamente"
