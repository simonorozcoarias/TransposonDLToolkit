#!/bin/bash
#SBATCH -o logs/prepare_dnabert2_data_%j.out
#SBATCH -e logs/prepare_dnabert2_data_%j.err
#SBATCH --mail-type END
#SBATCH --mail-user jgilbaja@uoc.edu
#SBATCH -J prepare_dnabert2_data
#SBATCH --time=3-00:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition long
#SBATCH --account=inpactor3

# module load cuda-toolkit/12.9.1
source ~/anaconda3/bin/activate DNABERT2

# Define base paths
BASE_DIR=~/inpactor3/auto_detection/phase2
PYTHON_SCRIPT="${BASE_DIR}/scripts/prepare_dnabert2_data.py"
PILOT_DATA_DIR="${BASE_DIR}/data"
OUTPUT_BASE_DIR="${BASE_DIR}/datasets"
STRIDE="2048"  # 0 overlap

echo "=========================================="
echo "PROCESSING ALL SPECIES FOR DNABERT-2"
echo "=========================================="
echo "Scanning for species in: ${PILOT_DATA_DIR}"
echo ""

# Counter for processed species
PROCESSED_COUNT=0
FAILED_COUNT=0

# Automatically discover species by iterating over directories
for SPECIES_DIR in "${PILOT_DATA_DIR}"/*; do
    # Check if it's a directory
    if [[ ! -d "${SPECIES_DIR}" ]]; then
        continue
    fi

    # Extract species name from directory path
    SPECIES=$(basename "${SPECIES_DIR}")

    echo "=========================================="
    echo "Processing species: ${SPECIES}"
    echo "=========================================="

    # Define input files for this species
    DATA_DIR="${SPECIES_DIR}/TEgenomeSimulator_${SPECIES}_synth_result"
    FASTA_FILE="${DATA_DIR}/${SPECIES}_synth_genome_sequence_out_final.fasta"
    GFF_FILE="${DATA_DIR}/${SPECIES}_synth_repeat_annotation_out_final.gff"
    OUTPUT_DIR="${OUTPUT_BASE_DIR}/${SPECIES}"

    # Verify input files exist
    if [[ ! -f "${FASTA_FILE}" ]]; then
        echo "ERROR: FASTA file not found: ${FASTA_FILE}"
        ((FAILED_COUNT++))
        continue
    fi

    if [[ ! -f "${GFF_FILE}" ]]; then
        echo "ERROR: GFF file not found: ${GFF_FILE}"
        ((FAILED_COUNT++))
        continue
    fi

    echo "Input FASTA: ${FASTA_FILE}"
    echo "Input GFF:   ${GFF_FILE}"
    echo "Output dir:  ${OUTPUT_DIR}"
    echo ""

    # Execute the Python script with species parameter
    stdbuf -oL -eL python3 -u "${PYTHON_SCRIPT}" \
        "${FASTA_FILE}" \
        "${GFF_FILE}" \
        "${OUTPUT_DIR}" \
        --species "${SPECIES}" \
        --stride "${STRIDE}"

    if [[ $? -eq 0 ]]; then
        echo "✓ Successfully processed ${SPECIES}"
        ((PROCESSED_COUNT++))
    else
        echo "✗ Failed to process ${SPECIES}"
        ((FAILED_COUNT++))
    fi
    echo ""
done

echo "=========================================="
echo "ALL SPECIES PROCESSING COMPLETED"
echo "=========================================="
echo "Successfully processed: ${PROCESSED_COUNT} species"
echo "Failed: ${FAILED_COUNT} species"
echo "=========================================="

# Combine all species splits if at least one species was processed successfully
if [[ ${PROCESSED_COUNT} -gt 0 ]]; then
    echo ""
    echo "=========================================="
    echo "COMBINING SPECIES SPLITS (Split-Then-Combine)"
    echo "=========================================="
    echo "This new pipeline:"
    echo "  1. Each species already has train/val/test splits"
    echo "  2. Combining each split separately (train, val, test)"
    echo "  3. Shuffling to mix species and avoid order bias"
    echo ""

    COMBINE_SCRIPT="${BASE_DIR}/scripts/combine_datasets.py"
    COMBINED_OUTPUT_DIR="${OUTPUT_BASE_DIR}/combined_all_species_splits"

    echo "Input:  ${OUTPUT_BASE_DIR}"
    echo "Output: ${COMBINED_OUTPUT_DIR}"
    echo ""

    stdbuf -oL -eL python3 -u "${COMBINE_SCRIPT}" \
        "${OUTPUT_BASE_DIR}" \
        "${COMBINED_OUTPUT_DIR}" \
        --seed 42

    if [[ $? -eq 0 ]]; then
        echo ""
        echo "✓ Successfully combined all species splits"
        echo "Combined splits location:"
        echo "  Train: ${COMBINED_OUTPUT_DIR}/train/"
        echo "  Val:   ${COMBINED_OUTPUT_DIR}/val/"
        echo "  Test:  ${COMBINED_OUTPUT_DIR}/test/"
        echo ""
        echo "IMPORTANT: Splits have been shuffled to mix species and avoid training bias"
    else
        echo "✗ Failed to combine species splits"
    fi
else
    echo ""
    echo "WARNING: No species were processed successfully. Skipping split combination."
fi

echo ""
echo "=========================================="
echo "JOB COMPLETED"
echo "=========================================="
