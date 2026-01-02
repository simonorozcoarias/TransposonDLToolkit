#!/bin/bash
#SBATCH --job-name=prod_40species
#SBATCH --output=logs/slurm_production_%j.out
#SBATCH --error=logs/slurm_production_%j.err
#SBATCH --mail-type END
#SBATCH --mail-user=REPLACE_WITH_YOUR_EMAIL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:3g.20gb:1
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --partition=REPLACE_WITH_YOUR_PARTITION
#SBATCH --account=REPLACE_WITH_YOUR_ACCOUNT

################################################################################
# PRODUCTION TRAINING - 40 Species Multi-Kingdom Model (Custom Model)
#
# Purpose: Train definitive DNABERT-2 model with custom token classification head
#          using taxonomically balanced dataset
# Expected runtime: 3-5 days
#
# Species distribution:
#   - 15 animals, 10 plants, 10 fungi, 5 other organisms
#
# REQUIRED SETUP:
#   1. Copy slurm/config.env.template to slurm/config.env
#   2. Edit config.env with your system paths
#   3. Update SBATCH directives above with your email/partition/account
#
# Usage:
#   sbatch slurm/production/submit_training_production.sh
################################################################################

# ============================================================
# Load Environment Configuration
# ============================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_ENV="${SCRIPT_DIR}/../config.env"

if [ -f "$CONFIG_ENV" ]; then
    source "$CONFIG_ENV"
else
    echo "ERROR: config.env not found at $CONFIG_ENV"
    echo "Please copy config.env.template to config.env and configure it."
    exit 1
fi

echo "===================================="
echo "PRODUCTION TRAINING - 40 SPECIES"
echo "===================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "===================================="

# ============================================================
# Configuration: Species Distribution
# ============================================================
NUM_ANIMALS=15
NUM_PLANTS=10
NUM_FUNGI=10
NUM_OTHER=5
TOTAL_SPECIES=$((NUM_ANIMALS + NUM_PLANTS + NUM_FUNGI + NUM_OTHER))

echo ""
echo "Target species distribution:"
echo "  Animals: $NUM_ANIMALS"
echo "  Plants:  $NUM_PLANTS"
echo "  Fungi:   $NUM_FUNGI"
echo "  Other:   $NUM_OTHER"
echo "  TOTAL:   $TOTAL_SPECIES species"
echo ""

# Create logs directory
mkdir -p logs

# ============================================================
# Environment Setup
# ============================================================
echo ""
echo "Setting up environment..."

# Activate conda environment (CONDA_ENV from config.env)
conda activate ${CONDA_ENV}

echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "Number of GPUs: $(python -c 'import torch; print(torch.cuda.device_count())')"

# Environment variables already set in config.env, but ensure they're exported
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Environment setup complete."
echo ""

# ============================================================
# Project Setup
# ============================================================

cd ${PHASE2_DIR} || { echo "Error: Cannot access project directory ${PHASE2_DIR}"; exit 1; }
echo "Working directory: $(pwd)"

CONFIG_FILE="config/production.yaml"
CONFIG_BACKUP="${CONFIG_FILE}.backup_$$"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file '$CONFIG_FILE' not found!"
    exit 1
fi

# Create backup of config file
cp "$CONFIG_FILE" "$CONFIG_BACKUP"

# Trap to restore config on exit/error
cleanup_on_exit() {
    if [ -f "$CONFIG_BACKUP" ]; then
        echo ""
        echo "Restoring config from backup..."
        cp "$CONFIG_BACKUP" "$CONFIG_FILE"
        rm -f "$CONFIG_BACKUP"
    fi
}
trap cleanup_on_exit EXIT INT TERM

# ============================================================
# Pre-flight Checks
# ============================================================
echo ""
echo "===================================="
echo "PRE-FLIGHT CHECKS"
echo "===================================="

# Check disk space
df -h ${PHASE2_DIR}

echo ""
echo "===================================="
echo "CATEGORY-BASED SPECIES SELECTION"
echo "===================================="

# Create temporary directory for selection outputs
SELECTION_DIR="${PHASE2_DIR}/temp_selection_$$"
mkdir -p "$SELECTION_DIR"

echo "Running species selection script..."
echo "  Target: $NUM_ANIMALS animals, $NUM_PLANTS plants, $NUM_FUNGI fungi, $NUM_OTHER other"
echo ""

# Run species selection script (now in scripts/data_preparation/)
python -u scripts/data_preparation/select_species_by_category.py \
    --animals $NUM_ANIMALS \
    --plants $NUM_PLANTS \
    --fungi $NUM_FUNGI \
    --other $NUM_OTHER \
    --csv "${PHASE2_DIR}/results/species_gc_data_v2.csv" \
    --datasets-dir "${DATASETS_DIR}" \
    --coverage-csv "${PHASE2_DIR}/results/all_species_coverage.csv" \
    --seed 42 \
    --output "$SELECTION_DIR/selected_species.txt" \
    --json-output "$SELECTION_DIR/species_metadata.json" \
    2>&1 | tee "${PHASE2_DIR}/logs/species_selection_${SLURM_JOB_ID}.log"

SELECTION_EXIT_CODE=${PIPESTATUS[0]}

# Check if selection was successful
if [ $SELECTION_EXIT_CODE -ne 0 ]; then
    echo "❌ ERROR: Species selection failed with exit code $SELECTION_EXIT_CODE"
    echo "Check log: ${PHASE2_DIR}/logs/species_selection_${SLURM_JOB_ID}.log"
    rm -rf "$SELECTION_DIR"
    exit 1
fi

# Load selected species list
if [ ! -f "$SELECTION_DIR/selected_species.txt" ]; then
    echo "❌ ERROR: Species list file not created"
    rm -rf "$SELECTION_DIR"
    exit 1
fi

mapfile -t SPECIES_LIST < "$SELECTION_DIR/selected_species.txt"

echo ""
echo "✅ Selected ${#SPECIES_LIST[@]} species across 4 kingdoms"
echo ""
echo "Selected species:"
printf '  - %s\n' "${SPECIES_LIST[@]}"
echo ""

# Extract imbalance ratio from selection log
IMBALANCE_RATIO=$(grep "^IMBALANCE_RATIO=" "${PHASE2_DIR}/logs/species_selection_${SLURM_JOB_ID}.log" | tail -1 | cut -d'=' -f2)

if [ -n "$IMBALANCE_RATIO" ]; then
    echo "✅ Imbalance ratio calculated: $IMBALANCE_RATIO"
else
    echo "⚠️  Imbalance ratio not calculated, will use default"
fi

echo ""
echo "===================================="
echo "DATASET COMBINATION"
echo "===================================="

# Create combined datasets directory structure
COMBINED_BASE_DIR="${DATASETS_COMBINED_DIR}"
mkdir -p "$COMBINED_BASE_DIR"

# Create unique name with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
COMBINED_DIR="$COMBINED_BASE_DIR/40_species_production_${TIMESTAMP}"
TEMP_SPECIES_DIR="${PHASE2_DIR}/temp_species_$$"

echo "Creating temporary species directory..."
mkdir -p "$TEMP_SPECIES_DIR"

# Create symbolic links for selected species
for species in "${SPECIES_LIST[@]}"; do
    ln -s "${DATASETS_DIR}/$species" "$TEMP_SPECIES_DIR/$species"
done

echo "Combining datasets..."
echo "  Output: $COMBINED_DIR"
python -u scripts/data_preparation/combine_datasets.py \
    "$TEMP_SPECIES_DIR" \
    "$COMBINED_DIR" \
    --seed 42 \
    --coverage-csv "${PHASE2_DIR}/results/all_species_coverage.csv" \
    2>&1 | tee -a "${PHASE2_DIR}/logs/combine_production_${SLURM_JOB_ID}.log"

COMBINE_EXIT_CODE=${PIPESTATUS[0]}

# Check if combination was successful
if [ $COMBINE_EXIT_CODE -ne 0 ]; then
    echo "❌ ERROR: Dataset combination failed with exit code $COMBINE_EXIT_CODE"
    echo "Check log: ${PHASE2_DIR}/logs/combine_production_${SLURM_JOB_ID}.log"
    rm -rf "$TEMP_SPECIES_DIR"
    rm -rf "$COMBINED_DIR"
    rm -rf "$SELECTION_DIR"
    exit 1
fi

# Verify that train/val/test directories were created
if [ ! -d "$COMBINED_DIR/train" ] || [ ! -d "$COMBINED_DIR/val" ] || [ ! -d "$COMBINED_DIR/test" ]; then
    echo "❌ ERROR: Combined dataset splits not found!"
    echo "Expected directories:"
    echo "  - $COMBINED_DIR/train"
    echo "  - $COMBINED_DIR/val"
    echo "  - $COMBINED_DIR/test"
    echo ""
    echo "Actual contents of $COMBINED_DIR:"
    ls -la "$COMBINED_DIR" 2>/dev/null || echo "Directory does not exist"
    echo ""
    echo "Check combine log for details: ${PHASE2_DIR}/logs/combine_production_${SLURM_JOB_ID}.log"
    rm -rf "$TEMP_SPECIES_DIR"
    rm -rf "$COMBINED_DIR"
    rm -rf "$SELECTION_DIR"
    exit 1
fi

echo "✅ Dataset combination successful - all splits created"

# Clean up temp species directory
rm -rf "$TEMP_SPECIES_DIR"

# Update config to point to combined dataset
FULL_DATASET_PATH="$COMBINED_DIR"

# Update config file with RELATIVE path (to avoid absolute path issues)
RELATIVE_COMBINED_PATH=$(realpath --relative-to="${PHASE2_DIR}" "$COMBINED_DIR")

echo ""
echo "===================================="
echo "CONFIG FILE UPDATE"
echo "===================================="
echo "Updating config file to use combined dataset..."
echo "  Relative path: $RELATIVE_COMBINED_PATH"

sed -i "s|dataset_path:.*|dataset_path: \"$RELATIVE_COMBINED_PATH\"|" "$CONFIG_FILE"
sed -i "s|output_dir:.*|output_dir: \"./results/production_40_species_custom\"|" "$CONFIG_FILE"
sed -i "s|logging_dir:.*|logging_dir: \"./results/production_40_species_custom/logs\"|" "$CONFIG_FILE"
sed -i "s|run_name:.*|run_name: \"production_40_species_custom_${TIMESTAMP}\"|" "$CONFIG_FILE"

OUTPUT_DIR="${RESULTS_DIR}/production_40_species_custom"
RUN_NAME="production_40_species_custom_${TIMESTAMP}"

echo "  Output directory: $OUTPUT_DIR"
echo "  Run name: $RUN_NAME"
echo ""

# ============================================================
# Setup Output Directory
# ============================================================
echo "===================================="
echo "OUTPUT DIRECTORY SETUP"
echo "===================================="
echo "Output directory: $OUTPUT_DIR"

# Clean previous production results if exists
if [ -d "$OUTPUT_DIR" ]; then
    echo "⚠️  Removing previous production results..."
    rm -rf "$OUTPUT_DIR"
fi

# Create fresh output directory
mkdir -p "$OUTPUT_DIR"
chmod 755 "$OUTPUT_DIR"

# Test write permissions
TEST_FILE="$OUTPUT_DIR/.write_test_$$"
if touch "$TEST_FILE" 2>/dev/null; then
    rm -f "$TEST_FILE"
    echo "✅ Write permissions verified"
else
    echo "❌ ERROR: Cannot write to output directory!"
    exit 1
fi
echo ""

# ============================================================
# Training with Enhanced Logging
# ============================================================
echo "===================================="
echo "STARTING PRODUCTION TRAINING"
echo "===================================="
echo "Configuration: $CONFIG_FILE"
echo "Expected checkpoints every 500 steps in: $OUTPUT_DIR"
echo "Species: $TOTAL_SPECIES (${NUM_ANIMALS} animals, ${NUM_PLANTS} plants, ${NUM_FUNGI} fungi, ${NUM_OTHER} other)"
echo ""

# Build imbalance ratio argument
if [ -n "$IMBALANCE_RATIO" ]; then
    IMBALANCE_RATIO_ARG="--imbalance-ratio $IMBALANCE_RATIO"
else
    IMBALANCE_RATIO_ARG=""
fi

# Use python -u for unbuffered output
# Redirect both stdout and stderr to separate files while displaying
python -u scripts/training/train_token_classification.py \
    --config $CONFIG_FILE \
    $IMBALANCE_RATIO_ARG \
    2>&1 | tee -a "${PHASE2_DIR}/logs/training_production_${SLURM_JOB_ID}.log"

TRAINING_EXIT_CODE=${PIPESTATUS[0]}

# ============================================================
# Post-Training Verification
# ============================================================
echo ""
echo "===================================="
echo "POST-TRAINING VERIFICATION"
echo "===================================="

if [ -d "$OUTPUT_DIR" ]; then
    echo "Checking saved checkpoints..."

    CHECKPOINTS=$(find "$OUTPUT_DIR" -maxdepth 1 -type d -name "checkpoint-*" 2>/dev/null | sort -V)

    if [ -n "$CHECKPOINTS" ]; then
        NUM_CHECKPOINTS=$(echo "$CHECKPOINTS" | wc -l)
        echo "✅ SUCCESS: Found $NUM_CHECKPOINTS checkpoint(s):"
        echo "$CHECKPOINTS"

        # Show details of latest checkpoint
        LATEST=$(echo "$CHECKPOINTS" | tail -1)
        echo ""
        echo "Latest checkpoint contents:"
        ls -lh "$LATEST" | head -15
    else
        echo "❌ FAILURE: No checkpoints were saved!"
        echo ""
        echo "Output directory contents:"
        ls -lah "$OUTPUT_DIR"
    fi

    # Check for other expected files
    echo ""
    echo "Other output files:"
    ls -lh "$OUTPUT_DIR" | grep -v "^d" | head -10
else
    echo "❌ ERROR: Output directory does not exist!"
fi

# ============================================================
# Summary
# ============================================================
echo ""
echo "===================================="
echo "PRODUCTION TRAINING SUMMARY"
echo "===================================="
echo "Species: $TOTAL_SPECIES (${NUM_ANIMALS} animals, ${NUM_PLANTS} plants, ${NUM_FUNGI} fungi, ${NUM_OTHER} other)"
echo "Exit code: $TRAINING_EXIT_CODE"
echo "Job ID: $SLURM_JOB_ID"
echo "End time: $(date)"

if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✅ Training completed successfully"
else
    echo ""
    echo "❌ Training failed with exit code $TRAINING_EXIT_CODE"
    echo "Check logs:"
    echo "  cat logs/slurm_production_$SLURM_JOB_ID.err"
    echo "  cat logs/training_production_${SLURM_JOB_ID}.log"
fi

echo ""
echo "Training log saved to: logs/training_production_${SLURM_JOB_ID}.log"
echo "Species selection log: logs/species_selection_${SLURM_JOB_ID}.log"
echo "Dataset combination log: logs/combine_production_${SLURM_JOB_ID}.log"

# ============================================================
# Cleanup
# ============================================================
echo ""
echo "===================================="
echo "CLEANUP"
echo "===================================="

# Restore config file to defaults
echo "Restoring config file to default values..."
sed -i 's|dataset_path:.*|dataset_path: "datasets_combined/40_species_production"|' "$CONFIG_FILE"
sed -i 's|output_dir:.*|output_dir: "./results/production_40_species_custom"|' "$CONFIG_FILE"
sed -i 's|logging_dir:.*|logging_dir: "./results/production_40_species_custom/logs"|' "$CONFIG_FILE"
sed -i 's|run_name:.*|run_name: "production_40_species_custom"|' "$CONFIG_FILE"
echo "✅ Config file restored to defaults"

# Clean up selection directory
rm -rf "$SELECTION_DIR"
echo "✅ Temporary selection files cleaned up"

# Preserve combined dataset for evaluation
echo "Combined dataset location: $COMBINED_DIR"
echo "Species metadata: $COMBINED_DIR/../species_metadata.json (if saved)"
echo "✅ Combined dataset preserved for evaluation"

echo ""
echo "===================================="
echo "PRODUCTION RUN COMPLETE"
echo "===================================="

exit $TRAINING_EXIT_CODE
