#!/bin/bash

#SBATCH -o get_species_gc.out
#SBATCH -e get_species_gc.err
#SBATCH --mail-type END
#SBATCH --mail-user jgilbaja@uoc.edu
#SBATCH -J get_species_gc
#SBATCH --time 1-00:00:00
#SBATCH --partition fast
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --mem 8GB

# Cargar módulos necesarios si es requerido por el cluster
# (Descomentar si tu cluster requiere cargar Python/Anaconda)
# module load anaconda3

# Activar entorno conda con ncbi-datasets-cli
source ~/anaconda3/bin/activate ncbi_datasets

# Directorio de trabajo
cd ~/tagua_gen_ec/TransposonDLToolkit/auto_detection/phase1/scripts

# Ejecutar script con output sin buffering
stdbuf -oL -eL python3 -u get_species_gc_optimized.py \
    --input ../results/species_list_v2.csv \
    --output ../results/species_gc_data_v2.csv \
    --rate-limit 1.0

echo "✅ Job completado"