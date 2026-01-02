#!/bin/bash
#SBATCH -o logs/combine_datasets_%j.out
#SBATCH -e logs/combine_datasets_%j.err
#SBATCH --mail-type END
#SBATCH --mail-user jgilbaja@uoc.edu
#SBATCH -J combine_datasets
#SBATCH --time=2-00:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --partition long
#SBATCH --account=inpactor3

# Script optimizado para SOLO combinar los splits de especies ya generados
# Este script asume que los datasets individuales de cada especie ya existen

source ~/anaconda3/bin/activate DNABERT2

# Define base paths
BASE_DIR=~/inpactor3/auto_detection/phase2
COMBINE_SCRIPT="${BASE_DIR}/scripts/combine_datasets.py"
DATASETS_DIR="${BASE_DIR}/datasets"
COMBINED_OUTPUT_DIR="${BASE_DIR}/datasets/combined_all_species_splits"

echo "=========================================="
echo "COMBINING SPECIES SPLITS (Optimized)"
echo "=========================================="
echo "Script version: Optimized with max_shard_size + num_proc"
echo ""
echo "Input directory:  ${DATASETS_DIR}"
echo "Output directory: ${COMBINED_OUTPUT_DIR}"
echo "CPUs available:   8"
echo "Memory:           128G"
echo ""

# Verificar que el script de combinación existe
if [[ ! -f "${COMBINE_SCRIPT}" ]]; then
    echo "❌ ERROR: Script not found: ${COMBINE_SCRIPT}"
    exit 1
fi

# Verificar que el directorio de datasets existe
if [[ ! -d "${DATASETS_DIR}" ]]; then
    echo "❌ ERROR: Datasets directory not found: ${DATASETS_DIR}"
    exit 1
fi

# Contar especies disponibles (directorios con splits train/val/test)
echo "Verificando especies disponibles..."
SPECIES_COUNT=0
for SPECIES_DIR in "${DATASETS_DIR}"/*; do
    if [[ -d "${SPECIES_DIR}" ]]; then
        SPECIES=$(basename "${SPECIES_DIR}")
        # Skip combined directory if exists
        if [[ "${SPECIES}" == "combined_all_species_splits" ]]; then
            continue
        fi

        # Check if it has splits
        if [[ -d "${SPECIES_DIR}/train" && -d "${SPECIES_DIR}/val" && -d "${SPECIES_DIR}/test" ]]; then
            echo "  ✓ Found: ${SPECIES}"
            ((SPECIES_COUNT++))
        fi
    fi
done

if [[ ${SPECIES_COUNT} -eq 0 ]]; then
    echo ""
    echo "❌ ERROR: No se encontraron especies con splits train/val/test"
    echo "Verifica que los datasets individuales se hayan generado correctamente"
    exit 1
fi

echo ""
echo "Total especies encontradas: ${SPECIES_COUNT}"
echo ""
echo "=========================================="
echo "INICIANDO COMBINACIÓN"
echo "=========================================="
echo ""

# Ejecutar el script de combinación optimizado
stdbuf -oL -eL python3 -u "${COMBINE_SCRIPT}" \
    "${DATASETS_DIR}" \
    "${COMBINED_OUTPUT_DIR}" \
    --seed 42

EXIT_CODE=$?

echo ""
echo "=========================================="
if [[ ${EXIT_CODE} -eq 0 ]]; then
    echo "✅ COMBINACIÓN COMPLETADA EXITOSAMENTE"
    echo "=========================================="
    echo ""
    echo "Splits combinados disponibles en:"
    echo "  Train: ${COMBINED_OUTPUT_DIR}/train/"
    echo "  Val:   ${COMBINED_OUTPUT_DIR}/val/"
    echo "  Test:  ${COMBINED_OUTPUT_DIR}/test/"
    echo ""
    echo "NOTA: Los splits han sido shuffleados para mezclar especies"
    echo "      y evitar sesgo de entrenamiento"
    echo ""

    # Mostrar tamaño de los directorios generados
    if command -v du &> /dev/null; then
        echo "Tamaño de los splits generados:"
        du -sh "${COMBINED_OUTPUT_DIR}/train" 2>/dev/null && echo "  Train: $(du -sh ${COMBINED_OUTPUT_DIR}/train | cut -f1)"
        du -sh "${COMBINED_OUTPUT_DIR}/val" 2>/dev/null && echo "  Val:   $(du -sh ${COMBINED_OUTPUT_DIR}/val | cut -f1)"
        du -sh "${COMBINED_OUTPUT_DIR}/test" 2>/dev/null && echo "  Test:  $(du -sh ${COMBINED_OUTPUT_DIR}/test | cut -f1)"
        echo ""
    fi
else
    echo "❌ ERROR EN LA COMBINACIÓN"
    echo "=========================================="
    echo ""
    echo "Exit code: ${EXIT_CODE}"
    echo "Revisa los logs para más detalles"
fi

echo "JOB FINALIZADO"
echo "=========================================="
