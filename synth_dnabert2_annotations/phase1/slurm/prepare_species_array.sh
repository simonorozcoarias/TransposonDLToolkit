#!/bin/bash

################################################################################
# Script preparador de lista de especies para Job Arrays
#
# Genera un archivo de texto con la lista de especies a procesar.
# Cada línea corresponde a una especie que será procesada por una tarea del array.
#
# Uso:
#   bash prepare_species_array.sh <START_INDEX> <END_INDEX> [SCALING_FACTOR]
#
# Parámetros:
#   START_INDEX: Índice inicial de especie
#   END_INDEX: Índice final de especie
#   SCALING_FACTOR: (Opcional) Factor de escalado (default: 1.0, solo para info)
#
# Salida:
#   ../results/species_list_<START>_<END>.txt
#   Formato: INDEX|SPECIES_NAME|NUM_SEQUENCES
#
# Ejemplo:
#   bash prepare_species_array.sh 1 100
#   # Genera: ../results/species_list_1_100.txt
#
# Luego usar con:
#   sbatch --array=1-100%20 job_array_generate_genomes.sh ../results/species_list_1_100.txt 1.0
#
# Autor: Jorge González Gilbaja
# TFM - UOC
################################################################################

# Parámetros de entrada
START_INDEX=${1:-1}
END_INDEX=${2:-10}
SCALING_FACTOR=${3:-1.0}

# Validar parámetros
if ! [[ "$START_INDEX" =~ ^[0-9]+$ ]] || ! [[ "$END_INDEX" =~ ^[0-9]+$ ]]; then
    echo "❌ ERROR: Los índices deben ser números enteros positivos"
    echo "Uso: bash $0 <START_INDEX> <END_INDEX> [SCALING_FACTOR]"
    exit 1
fi

if [ "$START_INDEX" -gt "$END_INDEX" ]; then
    echo "❌ ERROR: START_INDEX debe ser menor o igual a END_INDEX"
    exit 1
fi

echo "========================================================================"
echo "  Preparación de lista de especies para Job Array"
echo "========================================================================"
echo "Rango: especies $START_INDEX a $END_INDEX"
echo "Scaling factor: $SCALING_FACTOR (informativo)"
echo "Fecha: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================================================"
echo ""

# Directorio de trabajo
WORKDIR=~/tagua_gen_ec/TransposonDLToolkit/auto_detection/phase1
cd $WORKDIR/scripts

# Archivos de entrada
SPECIES_GC_DATA="../results/species_gc_data_v2.csv"
OUTPUT_LIST="../results/species_list_${START_INDEX}_${END_INDEX}.txt"
SKIPPED_LOG="../results/skipped_species_prepare_${START_INDEX}_${END_INDEX}.csv"

# Verificar que existe el archivo de datos
if [ ! -f "$SPECIES_GC_DATA" ]; then
    echo "❌ ERROR: No se encuentra el archivo $SPECIES_GC_DATA"
    exit 1
fi

echo "📊 Filtrando especies con num_sequences >= 10..."

# Inicializar log de saltadas
echo "species,num_sequences,reason,timestamp" > "$SKIPPED_LOG"

# Generar lista de especies
awk -F',' -v start="$START_INDEX" -v end="$END_INDEX" -v skipped_log="$SKIPPED_LOG" '
    NR==1 {next}  # Saltar encabezado
    {
        if ($2 < 10) {
            # Registrar especies saltadas
            timestamp = strftime("%Y-%m-%d %H:%M:%S")
            print "\"" $1 "\"," $2 ",\"num_sequences < 10\",\"" timestamp "\"" >> skipped_log
        } else {
            # Contar especies válidas
            count++
            if (count >= start && count <= end) {
                # Formato: INDEX|SPECIES|NUM_SEQUENCES
                print count "|" $1 "|" $2
            }
        }
    }
' "$SPECIES_GC_DATA" > "$OUTPUT_LIST"

# Contar especies generadas
TOTAL_SPECIES=$(wc -l < "$OUTPUT_LIST")
SKIPPED_COUNT=$(awk 'NR>1' "$SKIPPED_LOG" | wc -l 2>/dev/null || echo 0)

if [ "$TOTAL_SPECIES" -eq 0 ]; then
    echo "❌ ERROR: No se encontraron especies en el rango especificado"
    exit 1
fi

echo "✅ Lista generada exitosamente"
echo ""
echo "========================================================================"
echo "  RESUMEN"
echo "========================================================================"
echo "Total de especies en la lista: $TOTAL_SPECIES"
echo "Especies saltadas (< 10 seqs): $SKIPPED_COUNT"
echo ""
echo "Archivo generado:"
echo "  $OUTPUT_LIST"
echo ""
echo "Especies saltadas registradas en:"
echo "  $SKIPPED_LOG"
echo ""
echo "========================================================================"
echo "  SIGUIENTE PASO: LANZAR JOB ARRAY"
echo "========================================================================"
echo ""
echo "Para lanzar el job array con control de concurrencia:"
echo ""
echo "  # Máximo 20 jobs concurrentes (recomendado)"
echo "  sbatch --array=1-${TOTAL_SPECIES}%20 job_array_generate_genomes.sh \\"
echo "         $OUTPUT_LIST $SCALING_FACTOR"
echo ""
echo "  # Máximo 30 jobs concurrentes (agresivo)"
echo "  sbatch --array=1-${TOTAL_SPECIES}%30 job_array_generate_genomes.sh \\"
echo "         $OUTPUT_LIST $SCALING_FACTOR"
echo ""
echo "  # Sin límite de concurrencia (NO RECOMENDADO para >50 especies)"
echo "  sbatch --array=1-${TOTAL_SPECIES} job_array_generate_genomes.sh \\"
echo "         $OUTPUT_LIST $SCALING_FACTOR"
echo ""
echo "========================================================================"
echo "  COMANDOS ÚTILES POST-LANZAMIENTO"
echo "========================================================================"
echo ""
echo "Ver estado del array:"
echo "  squeue -u \$USER -j <ARRAY_JOB_ID>"
echo ""
echo "Cancelar todo el array:"
echo "  scancel <ARRAY_JOB_ID>"
echo ""
echo "Cancelar tarea específica del array:"
echo "  scancel <ARRAY_JOB_ID>_<TASK_ID>"
echo ""
echo "Relanzar tareas fallidas (obtener lista primero):"
echo "  sbatch --array=5,12,23 job_array_generate_genomes.sh $OUTPUT_LIST $SCALING_FACTOR"
echo ""
echo "Consolidar resultados:"
echo "  cat ../results/array_logs/execution_*.csv | awk 'NR==1 || !/^species,/' > ../results/all_array_executions.csv"
echo ""
echo "========================================================================"

exit 0
