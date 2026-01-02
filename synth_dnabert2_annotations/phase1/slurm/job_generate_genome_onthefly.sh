#!/bin/bash

#SBATCH -o outputs/generate_genome_onthefly.out
#SBATCH -e outputs/generate_genome_onthefly.err
#SBATCH --mail-type END
#SBATCH --mail-user jgilbaja@uoc.edu
#SBATCH -J generate_genome_onthefly
#SBATCH --time 0-08:00:00
#SBATCH --partition fast
#SBATCH -n 8
#SBATCH -N 1
#SBATCH --mem 64GB

################################################################################
# Script para generar genoma sintético individual
#
# Factor de escalado: Ajusta max/min copies para compensar fragmentación
# en TEgenomeSimulator. Por defecto 1.0 (sin ajuste).
# Ejemplo: SCALING_FACTOR=1.5 aumenta copies en 50%
################################################################################

# Configuración
SCALING_FACTOR=${1:-1.0}

# Cargar módulos necesarios si es requerido por el cluster
# (Descomentar si tu cluster requiere cargar Python/Anaconda)
# module load anaconda3

# Activar entorno conda con TEgenomeSimulator
source ~/anaconda3/bin/activate TEgenomeSimulator

# Directorio de trabajo
cd ~/tagua_gen_ec/TransposonDLToolkit/auto_detection/phase1/scripts

echo "========================================================================="
echo "  Generación de genoma sintético: Oryza sativa"
echo "========================================================================="
echo "Scaling factor: $SCALING_FACTOR"
echo "========================================================================="
echo ""

# Ejecutar script con output sin buffering
stdbuf -oL -eL python3 -u generate_genome_onthefly.py \
    --species "Oryza sativa" \
    --input-fasta ../../datasets/r.1.5_all.fasta \
    --species-index ../results/species_index_v2.json \
    --species-gc-data ../results/species_gc_data_v2.csv \
    --output-dir ../results/pilot_outputs/Oryza_sativa/ \
    --simulator-dir "../TEgenomeSimulator" \
    --scaling-factor "$SCALING_FACTOR"

echo "✅ Genoma sintético generado correctamente para Oryza sativa"