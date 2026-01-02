#!/bin/bash

#SBATCH -o outputs/genome_array_%A_%a.out
#SBATCH -e outputs/genome_array_%A_%a.err
#SBATCH --mail-type END
#SBATCH --mail-user jgilbaja@uoc.edu
#SBATCH -J genome_array
#SBATCH --time 0-04:00:00
#SBATCH --partition fast
#SBATCH -n 4
#SBATCH -N 1
#SBATCH --mem 16GB

################################################################################
# Script de generación de genomas usando SLURM Job Arrays
#
# SLURM Job Arrays permite lanzar múltiples jobs indexados automáticamente.
# Cada tarea del array procesa una especie diferente.
#
# Uso:
#   1. Generar archivo de índice de especies:
#      bash prepare_species_array.sh <START> <END> [SCALING]
#
#   2. Lanzar job array (automático con límite de concurrencia):
#      sbatch --array=1-100%20 job_array_generate_genomes.sh species_list.txt 1.0
#             ↑           ↑
#             │           └── Max 20 jobs concurrentes
#             └────────────── Rango de especies a procesar
#
# Parámetros:
#   SPECIES_LIST_FILE: Archivo con lista de especies (generado por prepare_species_array.sh)
#   SCALING_FACTOR: Factor de escalado (default: 1.0)
#
# Variables automáticas de SLURM:
#   SLURM_ARRAY_JOB_ID: ID del job array completo
#   SLURM_ARRAY_TASK_ID: Índice de esta tarea específica (1, 2, 3, ...)
#   SLURM_ARRAY_TASK_COUNT: Total de tareas en el array
#
# Ejemplo completo:
#   bash prepare_species_array.sh 1 100 1.5
#   sbatch --array=1-100%20 job_array_generate_genomes.sh ../results/species_list_1_100.txt 1.5
#
# Ventajas:
#   - No requiere job gestor (SLURM maneja la coordinación)
#   - Control nativo de concurrencia con %N
#   - No necesita terminal abierta
#   - Usa partición fast
#   - Puede cancelar/reanudar tareas individuales
#
# Autor: Jorge González Gilbaja
# TFM - UOC
################################################################################

# Parámetros de entrada
SPECIES_LIST_FILE="$1"
SCALING_FACTOR="${2:-1.0}"

# Validar parámetros
if [ -z "$SPECIES_LIST_FILE" ]; then
    echo "❌ ERROR: Debe proporcionar el archivo de lista de especies"
    echo "Uso: sbatch --array=1-N%M $0 <SPECIES_LIST_FILE> [SCALING_FACTOR]"
    exit 1
fi

if [ ! -f "$SPECIES_LIST_FILE" ]; then
    echo "❌ ERROR: No se encuentra el archivo $SPECIES_LIST_FILE"
    exit 1
fi

# Variables del array
ARRAY_JOB_ID=$SLURM_ARRAY_JOB_ID
TASK_ID=$SLURM_ARRAY_TASK_ID
TASK_COUNT=$SLURM_ARRAY_TASK_COUNT

echo "========================================================================"
echo "  Generación de genoma sintético - Job Array"
echo "========================================================================"
echo "Array Job ID: $ARRAY_JOB_ID"
echo "Task ID: $TASK_ID / $TASK_COUNT"
echo "Task Job ID: $SLURM_JOB_ID"
echo "Scaling factor: $SCALING_FACTOR"
echo "Species list: $SPECIES_LIST_FILE"
echo "Fecha inicio: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================================"
echo ""

# Activar entorno conda
source ~/anaconda3/bin/activate TEgenomeSimulator

# Directorio de trabajo
WORKDIR=~/tagua_gen_ec/TransposonDLToolkit/auto_detection/phase1
cd $WORKDIR/scripts

# Archivos de entrada
INPUT_FASTA="../../datasets/r.1.5_all.fasta"
SPECIES_INDEX_FILE="../results/species_index_v2.json"
SPECIES_GC_DATA="../results/species_gc_data_v2.csv"
OUTPUT_BASE_DIR="../results/pilot_outputs"
SIMULATOR_DIR="../TEgenomeSimulator"

# Leer la línea correspondiente a este TASK_ID del archivo de especies
SPECIES_LINE=$(sed -n "${TASK_ID}p" "$SPECIES_LIST_FILE")

if [ -z "$SPECIES_LINE" ]; then
    echo "❌ ERROR: No se encontró la especie para TASK_ID=$TASK_ID"
    exit 1
fi

# Parsear la línea (formato: INDEX|SPECIES|NUM_SEQ)
IFS='|' read -r SPECIES_INDEX SPECIES NUM_SEQ <<< "$SPECIES_LINE"

echo "📋 Procesando:"
echo "  Especie: $SPECIES"
echo "  Índice: $SPECIES_INDEX"
echo "  Secuencias: $NUM_SEQ"
echo ""

# Directorio de salida para esta especie
SPECIES_OUTPUT_DIR="${OUTPUT_BASE_DIR}/${SPECIES// /_}"

# Directorio de logs individuales
LOG_DIR="../results/array_logs"
mkdir -p "$LOG_DIR"

RUNTIME_LOG="${LOG_DIR}/runtime_${ARRAY_JOB_ID}_${TASK_ID}_${SPECIES// /_}.log"
EXECUTION_LOG="${LOG_DIR}/execution_${ARRAY_JOB_ID}_${TASK_ID}_${SPECIES// /_}.csv"

# Timestamp de inicio
START_TIME=$(date +%s)
START_TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')

echo "🕐 Inicio: $START_TIMESTAMP"
echo ""

# Ejecutar generate_genome_onthefly.py con logging en tiempo real
stdbuf -oL -eL python3 -u generate_genome_onthefly.py \
    --species "$SPECIES" \
    --input-fasta "$INPUT_FASTA" \
    --species-index "$SPECIES_INDEX_FILE" \
    --species-gc-data "$SPECIES_GC_DATA" \
    --output-dir "$SPECIES_OUTPUT_DIR" \
    --simulator-dir "$SIMULATOR_DIR" \
    --scaling-factor "$SCALING_FACTOR" 2>&1 | tee -a "$RUNTIME_LOG"

EXIT_CODE=${PIPESTATUS[0]}

# Timestamp de fin
END_TIME=$(date +%s)
END_TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
EXECUTION_TIME=$((END_TIME - START_TIME))

echo ""
echo "🕐 Fin: $END_TIMESTAMP"
echo "⏱️  Tiempo ejecución: ${EXECUTION_TIME}s ($(($EXECUTION_TIME/60))m $(($EXECUTION_TIME%60))s)"

# Extraer max_copies y min_copies del log
EXTRACTED_MAX=$(grep -oP 'max_copies_per_chr:\s*\K\d+' "$RUNTIME_LOG" | tail -1 | tr -d '\r\n')
EXTRACTED_MIN=$(grep -oP 'min_copies_per_chr:\s*\K\d+' "$RUNTIME_LOG" | tail -1 | tr -d '\r\n')

# Calcular tiempo en minutos y horas
EXECUTION_TIME_MINUTES=$(echo "scale=2; $EXECUTION_TIME / 60" | bc)
EXECUTION_TIME_HOURS=$(echo "scale=2; $EXECUTION_TIME / 3600" | bc)

# Determinar estado
if [ $EXIT_CODE -eq 0 ]; then
    STATUS="SUCCESS"
    echo "✅ ÉXITO"
else
    STATUS="FAILED"
    echo "❌ FALLO (exit code: $EXIT_CODE)"
fi

# Registrar ejecución en CSV individual
echo "species,index,array_job_id,task_id,execution_time_seconds,execution_time_minutes,execution_time_hours,max_copies,min_copies,status,start_time,end_time" > "$EXECUTION_LOG"
echo "\"$SPECIES\",$SPECIES_INDEX,$ARRAY_JOB_ID,$TASK_ID,$EXECUTION_TIME,$EXECUTION_TIME_MINUTES,$EXECUTION_TIME_HOURS,$EXTRACTED_MAX,$EXTRACTED_MIN,$STATUS,\"$START_TIMESTAMP\",\"$END_TIMESTAMP\"" >> "$EXECUTION_LOG"

echo ""
echo "========================================================================"
echo "  RESUMEN"
echo "========================================================================"
echo "Estado: $STATUS"
echo "Tiempo total: ${EXECUTION_TIME}s (${EXECUTION_TIME_MINUTES}m / ${EXECUTION_TIME_HOURS}h)"
echo ""
echo "Archivos de log generados:"
echo "  - Runtime log: $RUNTIME_LOG"
echo "  - Execution log: $EXECUTION_LOG"
echo ""
echo "Fecha fin: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================================"

exit $EXIT_CODE
