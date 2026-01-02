#!/bin/bash
#SBATCH --job-name=automodel_debug
#SBATCH --output=logs/slurm_automodel_debug_%j.out
#SBATCH --error=logs/slurm_automodel_debug_%j.err
#SBATCH --mail-type END
#SBATCH --mail-user jgilbaja@uoc.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:3g.20gb:1
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00  # 4 hours should be plenty for 1 species
#SBATCH --partition=gpu
#SBATCH --account=inpactor3

################################################################################
# AutoModel DEBUG Script - Single Species Test
#
# Purpose: Test checkpoint saving with a single species dataset
# Expected runtime: 30-60 minutes
#
# This script includes:
# 1. Explicit Python unbuffered output (-u flag)
# 2. Manual log flushing
# 3. Checkpoint verification after each save
# 4. Detailed debugging output
#
# Usage:
#   sbatch submit_training_debug.sh
################################################################################

echo "==================================="
echo "DEBUG MODE - Single Species Test"
echo "==================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "==================================="

# ============================================================
# Configuration: Number of Species
# ============================================================
# Set to 1 for single species, or N for random multi-species
NUM_SPECIES=32  # Change this to test with multiple species

# Create logs directory
mkdir -p logs

# ============================================================
# Environment Setup
# ============================================================
echo ""
echo "Setting up environment..."

source ~/anaconda3/bin/activate DNABERT2

echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "Number of GPUs: $(python -c 'import torch; print(torch.cuda.device_count())')"

# Critical: Force unbuffered output
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Environment setup complete."
echo ""

# ============================================================
# Project Setup
# ============================================================

PROJECT_DIR="$HOME/inpactor3/auto_detection/phase2"
cd $PROJECT_DIR || { echo "Error: Cannot access project directory"; exit 1; }
echo "Working directory: $(pwd)"

CONFIG_FILE="scripts/config_debug_single_species.yaml"
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
echo "==================================="
echo "PRE-FLIGHT CHECKS"
echo "==================================="

# Check disk space
df -h $PROJECT_DIR

# Output directory will be determined after dataset selection
# (different for single vs multi-species)

echo ""
echo "===================================="
echo "DATASET SELECTION AND PREPARATION"
echo "===================================="

if [ $NUM_SPECIES -eq 1 ]; then
    # Single species mode - use dataset from config
    DATASET_PATH=$(grep "dataset_path:" $CONFIG_FILE | awk '{print $2}' | tr -d '"')
    FULL_DATASET_PATH="$PROJECT_DIR/$DATASET_PATH"

    echo "Mode: Single species"
    echo "Dataset path: $FULL_DATASET_PATH"

    if [ ! -d "$FULL_DATASET_PATH" ]; then
        echo "❌ ERROR: Dataset not found at $FULL_DATASET_PATH"
        exit 1
    fi

    # Extract species name
    SPECIES_NAME=$(basename "$DATASET_PATH")
    SPECIES_LIST=("$SPECIES_NAME")

    # Set output directory for single species
    OUTPUT_DIR="$PROJECT_DIR/results/automodel_debug_single_species"
    RUN_NAME="automodel_debug_single_species"

else
    # Multi-species mode - select random species
    echo "Mode: Multi-species (random selection)"
    echo "Number of species: $NUM_SPECIES"

    # Get list of available species
    AVAILABLE_SPECIES=($(ls "$PROJECT_DIR/datasets/"))
    TOTAL_AVAILABLE=${#AVAILABLE_SPECIES[@]}

    if [ $NUM_SPECIES -gt $TOTAL_AVAILABLE ]; then
        echo "⚠️  WARNING: Requested $NUM_SPECIES species, but only $TOTAL_AVAILABLE available"
        NUM_SPECIES=$TOTAL_AVAILABLE
    fi

    # Select random species using shuf
    SPECIES_LIST=($(printf '%s\n' "${AVAILABLE_SPECIES[@]}" | shuf -n $NUM_SPECIES))

    echo "Selected species:"
    printf '  - %s\n' "${SPECIES_LIST[@]}"

    # Create combined datasets directory structure
    COMBINED_BASE_DIR="$PROJECT_DIR/datasets_combined"
    mkdir -p "$COMBINED_BASE_DIR"

    # Create unique name with timestamp
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    COMBINED_DIR="$COMBINED_BASE_DIR/${NUM_SPECIES}_species_seed42_${TIMESTAMP}"
    TEMP_SPECIES_DIR="$PROJECT_DIR/temp_species_$$"

    echo ""
    echo "Creating temporary species directory..."
    mkdir -p "$TEMP_SPECIES_DIR"

    # Create symbolic links for selected species
    for species in "${SPECIES_LIST[@]}"; do
        ln -s "$PROJECT_DIR/datasets/$species" "$TEMP_SPECIES_DIR/$species"
    done

    echo "Combining datasets..."
    echo "  Output: $COMBINED_DIR"
    python -u scripts/combine_datasets.py \
        "$TEMP_SPECIES_DIR" \
        "$COMBINED_DIR" \
        --seed 42 \
        --coverage-csv "$PROJECT_DIR/results/all_species_coverage.csv" \
        2>&1 | tee -a "$PROJECT_DIR/logs/combine_debug_${SLURM_JOB_ID}.log"

    COMBINE_EXIT_CODE=${PIPESTATUS[0]}

    # Check if combination was successful
    if [ $COMBINE_EXIT_CODE -ne 0 ]; then
        echo "❌ ERROR: Dataset combination failed with exit code $COMBINE_EXIT_CODE"
        echo "Check log: $PROJECT_DIR/logs/combine_debug_${SLURM_JOB_ID}.log"
        rm -rf "$TEMP_SPECIES_DIR"
        rm -rf "$COMBINED_DIR"
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
        echo "Check combine log for details: $PROJECT_DIR/logs/combine_debug_${SLURM_JOB_ID}.log"
        rm -rf "$TEMP_SPECIES_DIR"
        rm -rf "$COMBINED_DIR"
        exit 1
    fi

    echo "✅ Dataset combination successful - all splits created"

    # Extract imbalance ratio from combine output
    IMBALANCE_RATIO=$(grep "^IMBALANCE_RATIO=" "$PROJECT_DIR/logs/combine_debug_${SLURM_JOB_ID}.log" | tail -1 | cut -d'=' -f2)

    # Clean up temp directory
    rm -rf "$TEMP_SPECIES_DIR"

    # Update config to point to combined dataset
    FULL_DATASET_PATH="$COMBINED_DIR"

    # Update config file with RELATIVE path (to avoid absolute path issues)
    # Convert absolute path to relative from PROJECT_DIR
    RELATIVE_COMBINED_PATH=$(realpath --relative-to="$PROJECT_DIR" "$COMBINED_DIR")

    echo "Updating config file to use combined dataset..."
    echo "  Relative path: $RELATIVE_COMBINED_PATH"
    sed -i "s|dataset_path:.*|dataset_path: \"$RELATIVE_COMBINED_PATH\"|" "$CONFIG_FILE"

    # Set output directory for multi-species
    OUTPUT_DIR="$PROJECT_DIR/results/automodel_debug_${NUM_SPECIES}_species"
    RUN_NAME="automodel_debug_${NUM_SPECIES}_species_${TIMESTAMP}"

    # Update output paths in config file
    echo "Updating output directories in config..."
    sed -i "s|output_dir:.*|output_dir: \"./results/automodel_debug_${NUM_SPECIES}_species\"|" "$CONFIG_FILE"
    sed -i "s|logging_dir:.*|logging_dir: \"./results/automodel_debug_${NUM_SPECIES}_species/logs\"|" "$CONFIG_FILE"
    sed -i "s|run_name:.*|run_name: \"${RUN_NAME}\"|" "$CONFIG_FILE"
fi

echo "✅ Dataset prepared: $FULL_DATASET_PATH"
echo ""

# ============================================================
# Setup Output Directory (now that we know single vs multi)
# ============================================================
echo "===================================="
echo "OUTPUT DIRECTORY SETUP"
echo "===================================="
echo "Output directory: $OUTPUT_DIR"
echo "Run name: $RUN_NAME"

# Clean previous test if exists
if [ -d "$OUTPUT_DIR" ]; then
    echo "⚠️  Removing previous debug results..."
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
# Calculate Imbalance Ratio (if not already calculated)
# ============================================================
if [ -z "$IMBALANCE_RATIO" ]; then
    echo "===================================="
    echo "CALCULATING IMBALANCE RATIO"
    echo "===================================="

    COVERAGE_CSV="$PROJECT_DIR/results/all_species_coverage.csv"

    if [ ! -f "$COVERAGE_CSV" ]; then
        echo "⚠️  WARNING: Coverage CSV not found"
        echo "Using global default imbalance ratio (2.96)"
        IMBALANCE_RATIO=""
    else
        # Calculate using Python inline script
        echo "Species: ${SPECIES_LIST[@]}"

        # Convert bash array to Python list safely
        SPECIES_LIST_PYTHON=$(printf "'%s'," "${SPECIES_LIST[@]}")
        SPECIES_LIST_PYTHON="[${SPECIES_LIST_PYTHON%,}]"

        IMBALANCE_RATIO=$(python3 -c "
import csv
import sys

species_list = ${SPECIES_LIST_PYTHON}
csv_path = '${COVERAGE_CSV}'

total_te = 0
total_bg = 0

with open(csv_path, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['species'] in species_list:
            total_te += int(row['total_te_bases'])
            total_bg += int(row['total_background_bases'])

if total_te > 0:
    ratio = total_bg / total_te
    print(f'{ratio:.4f}')
else:
    sys.exit(1)
")

        if [ $? -eq 0 ] && [ -n "$IMBALANCE_RATIO" ]; then
            echo "✅ Imbalance ratio calculated: $IMBALANCE_RATIO"
        else
            echo "⚠️  Failed to calculate ratio, using global default (2.96)"
            IMBALANCE_RATIO=""
        fi
    fi
    echo ""
fi

# ============================================================
# Training with Enhanced Logging
# ============================================================
echo "==================================="
echo "STARTING DEBUG TRAINING"
echo "==================================="
echo "Configuration: $CONFIG_FILE"
echo "Expected checkpoints every 50 steps in: $OUTPUT_DIR"
echo ""

# Build imbalance ratio argument
if [ -n "$IMBALANCE_RATIO" ]; then
    IMBALANCE_RATIO_ARG="--imbalance-ratio $IMBALANCE_RATIO"
else
    IMBALANCE_RATIO_ARG=""
fi

# Use python -u for unbuffered output
# Redirect both stdout and stderr to separate files while displaying
python -u scripts/train_token_classification_automodel.py \
    --config $CONFIG_FILE \
    $IMBALANCE_RATIO_ARG \
    2>&1 | tee -a "$PROJECT_DIR/logs/training_debug_${SLURM_JOB_ID}.log"

TRAINING_EXIT_CODE=${PIPESTATUS[0]}

# ============================================================
# Post-Training Verification
# ============================================================
echo ""
echo "==================================="
echo "POST-TRAINING VERIFICATION"
echo "==================================="

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
echo "==================================="
echo "DEBUG TEST SUMMARY"
echo "==================================="
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
    echo "  cat logs/slurm_debug_$SLURM_JOB_ID.err"
    echo "  cat logs/training_debug_${SLURM_JOB_ID}.log"
fi

echo ""
echo "Full training log saved to: logs/training_debug_${SLURM_JOB_ID}.log"

# ============================================================
# Cleanup
# ============================================================
echo ""
echo "===================================="
echo "CLEANUP"
echo "===================================="

# Restore config file to defaults if it was modified
if [ $NUM_SPECIES -gt 1 ]; then
    echo "Restoring config file to default values..."
    sed -i 's|dataset_path:.*|dataset_path: "datasets/Acinonyx_jubatus"|' "$CONFIG_FILE"
    sed -i 's|output_dir:.*|output_dir: "./results/automodel_debug_single_species"|' "$CONFIG_FILE"
    sed -i 's|logging_dir:.*|logging_dir: "./results/automodel_debug_single_species/logs"|' "$CONFIG_FILE"
    sed -i 's|run_name:.*|run_name: "automodel_debug_single_species"|' "$CONFIG_FILE"
    echo "✅ Config file restored to defaults"
fi

# Clean up combined dataset if multi-species
if [ $NUM_SPECIES -gt 1 ] && [ -d "$COMBINED_DIR" ]; then
    echo "Combined dataset location: $COMBINED_DIR"
    # Optional: Uncomment the line below to automatically clean up combined datasets
    # rm -rf "$COMBINED_DIR"
    echo "✅ Combined dataset preserved for evaluation"
fi

echo ""
echo "===================================="

exit $TRAINING_EXIT_CODE
