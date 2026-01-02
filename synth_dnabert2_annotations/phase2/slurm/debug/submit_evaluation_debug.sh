#!/bin/bash
#SBATCH --job-name=dnabert2_eval_debug
#SBATCH --output=logs/slurm_eval_debug_%j.out
#SBATCH --error=logs/slurm_eval_debug_%j.err
#SBATCH --mail-type END
#SBATCH --mail-user jgilbaja@uoc.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:3g.20gb:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00  # 2 hours should be enough for single species eval
#SBATCH --partition=gpu
#SBATCH --account=inpactor3

################################################################################
# DNABERT-2 DEBUG Evaluation Script
#
# Purpose: Evaluate models trained with submit_training_debug.sh
# Tests the full save/load pipeline with standard HuggingFace methods
#
# This script includes:
# 1. Model loading verification with from_pretrained()
# 2. Evaluation on single or multi-species test sets
# 3. Per-species metrics (optional)
# 4. Detailed debugging output
#
# Parameters:
#   $1 MODEL_PATH    - Path to trained model directory (relative to project root)
#                      Default: results/dnabert2_debug_single_species
#   $2 DATASET_PATH  - Path to dataset directory (relative to project root)
#                      Default: datasets/Acinonyx_jubatus
#   $3 --by_species  - Optional flag to generate per-species metrics
#
# Usage Examples:
#   # Single species evaluation (uses defaults):
#   sbatch submit_evaluation_debug.sh
#
#   # Multi-species evaluation with custom model:
#   sbatch submit_evaluation_debug.sh results/dnabert2_debug_4_species datasets_combined/4_species_seed42_20251208_143022
#
#   # Multi-species with per-species metrics:
#   sbatch submit_evaluation_debug.sh results/dnabert2_debug_4_species datasets_combined/4_species_seed42_20251208_143022 --by_species
#
#   # Custom model with default single-species dataset:
#   sbatch submit_evaluation_debug.sh results/my_custom_model datasets/Acinonyx_jubatus
#
# Notes:
#   - All paths should be relative to $HOME/inpactor3/auto_detection/phase2
#   - The dataset must match the one used during training
#   - For multi-species evaluation, ensure the combined dataset still exists
#   - The config file will be temporarily modified and restored after evaluation
################################################################################

echo "===================================="
echo "DEBUG EVALUATION - Single Species"
echo "===================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Number of GPUs: $SLURM_GPUS"
echo "Number of CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Start time: $(date)"
echo "===================================="

# Create logs directory if it doesn't exist
mkdir -p logs

# ============================================================
# Environment Setup
# ============================================================
echo ""
echo "Setting up environment..."

# Activate conda environment
source ~/anaconda3/bin/activate DNABERT2

echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "Number of GPUs: $(python -c 'import torch; print(torch.cuda.device_count())')"

# Set environment variables
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Environment setup complete."
echo ""

# ============================================================
# Evaluation Configuration
# ============================================================

# Project directory
PROJECT_DIR="$HOME/inpactor3/auto_detection/phase2"

# Navigate to project directory
cd $PROJECT_DIR || { echo "Error: Cannot access project directory"; exit 1; }
echo "Working directory: $(pwd)"

# Parse command line arguments
# Usage: sbatch submit_evaluation_debug.sh [MODEL_PATH] [DATASET_PATH] [--by_species]
MODEL_PATH="${1:-results/dnabert2_debug_single_species}"
DATASET_PATH="${2:-datasets/Acinonyx_jubatus}"
BY_SPECIES_FLAG=""

# Check remaining arguments for --by_species flag
shift 2 2>/dev/null || shift $# 2>/dev/null
for arg in "$@"; do
    if [[ "$arg" == "--by_species" ]]; then
        BY_SPECIES_FLAG="--by_species"
    fi
done

echo "Model path: $MODEL_PATH"
echo "Dataset path: $DATASET_PATH"

# Configuration file (same as debug training)
CONFIG_FILE="scripts/config_debug_single_species.yaml"

# ============================================================
# Pre-flight Checks
# ============================================================
echo ""
echo "===================================="
echo "PRE-FLIGHT CHECKS"
echo "===================================="

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file '$CONFIG_FILE' not found!"
    exit 1
fi
echo "✅ Config file found: $CONFIG_FILE"

# Check if model directory exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ Error: Model directory '$MODEL_PATH' not found!"
    echo ""
    echo "Please ensure training has completed successfully."
    echo "Run: sbatch submit_training_debug.sh"
    exit 1
fi
echo "✅ Model directory found: $MODEL_PATH"

# Check for required model files
echo ""
echo "Checking model files..."
REQUIRED_FILES=("config.json" "pytorch_model.bin")
MISSING_FILES=0

for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$MODEL_PATH/$file" ]; then
        echo "  ✅ $file"
    else
        echo "  ❌ $file (MISSING)"
        MISSING_FILES=$((MISSING_FILES + 1))
    fi
done

if [ $MISSING_FILES -gt 0 ]; then
    echo ""
    echo "❌ Error: Missing $MISSING_FILES required file(s)"
    echo "Model may not have been saved correctly."
    echo ""
    echo "Model directory contents:"
    ls -lh "$MODEL_PATH"
    exit 1
fi

# Show model info
echo ""
echo "Model directory contents:"
ls -lh "$MODEL_PATH" | head -20

# Update config file temporarily to use the correct dataset
CONFIG_BACKUP="${CONFIG_FILE}.backup_eval_$$"
cp "$CONFIG_FILE" "$CONFIG_BACKUP"

# Trap to restore config on exit/error
cleanup_eval() {
    if [ -f "$CONFIG_BACKUP" ]; then
        echo ""
        echo "Restoring config from backup..."
        cp "$CONFIG_BACKUP" "$CONFIG_FILE"
        rm -f "$CONFIG_BACKUP"
    fi
}
trap cleanup_eval EXIT INT TERM

echo ""
echo "Updating config to use dataset: $DATASET_PATH"
sed -i "s|dataset_path:.*|dataset_path: \"$DATASET_PATH\"|" "$CONFIG_FILE"

if [ ! -z "$BY_SPECIES_FLAG" ]; then
    echo "Running evaluation with per-species metrics"
fi

# Batch size for evaluation
BATCH_SIZE=8

# ============================================================
# Run Evaluation
# ============================================================
echo ""
echo "===================================="
echo "STARTING EVALUATION"
echo "===================================="
echo "Model: $MODEL_PATH"
echo "Config: $CONFIG_FILE"
echo "Batch size: $BATCH_SIZE"
echo ""

# Run evaluation script with unbuffered output
python -u scripts/evaluate_model.py \
    --model_path $MODEL_PATH \
    --config $CONFIG_FILE \
    --batch_size $BATCH_SIZE \
    $BY_SPECIES_FLAG \
    2>&1 | tee -a logs/evaluation_debug_${SLURM_JOB_ID}.log

# Capture exit code
EVAL_EXIT_CODE=${PIPESTATUS[0]}

# ============================================================
# Post-Evaluation Verification
# ============================================================
echo ""
echo "===================================="
echo "POST-EVALUATION VERIFICATION"
echo "===================================="

EVAL_DIR="$MODEL_PATH/evaluation"

if [ -d "$EVAL_DIR" ]; then
    echo "✅ Evaluation directory created: $EVAL_DIR"
    echo ""
    echo "Evaluation results:"
    ls -lh "$EVAL_DIR"

    # Check for results files
    if [ -f "$EVAL_DIR/test_results.json" ]; then
        echo ""
        echo "✅ Overall results saved"
        echo ""
        echo "--- TEST RESULTS SUMMARY ---"
        python -c "
import json
with open('$EVAL_DIR/test_results.json', 'r') as f:
    results = json.load(f)
print(f\"Accuracy:  {results['accuracy']:.4f}\")
print(f\"Precision: {results['precision']:.4f}\")
print(f\"Recall:    {results['recall']:.4f}\")
print(f\"F1 Score:  {results['f1']:.4f}\")
print(f\"\nTotal tokens: {results['total_tokens']:,}\")
print(f\"  TE tokens: {results['TE_tokens']:,}\")
print(f\"  Background: {results['background_tokens']:,}\")
" 2>/dev/null || cat "$EVAL_DIR/test_results.json"
    else
        echo "❌ Overall results file not found"
    fi

    if [ -f "$EVAL_DIR/test_results_by_species.json" ] && [ ! -z "$BY_SPECIES_FLAG" ]; then
        echo ""
        echo "✅ Species-specific results saved"
    fi
else
    echo "❌ Evaluation directory was not created"
    echo "Evaluation may have failed"
fi

# ============================================================
# Summary
# ============================================================
echo ""
echo "===================================="
echo "DEBUG EVALUATION SUMMARY"
echo "===================================="
echo "Exit code: $EVAL_EXIT_CODE"
echo "Job ID: $SLURM_JOB_ID"
echo "End time: $(date)"

if [ $EVAL_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✅ Evaluation completed successfully!"
    echo ""
    echo "Results saved to: $EVAL_DIR"
    echo ""
    echo "View detailed results:"
    echo "  cat $EVAL_DIR/test_results.json"
    if [ ! -z "$BY_SPECIES_FLAG" ]; then
        echo "  cat $EVAL_DIR/test_results_by_species.json"
    fi
else
    echo ""
    echo "❌ Evaluation failed with exit code $EVAL_EXIT_CODE"
    echo ""
    echo "Check logs for details:"
    echo "  cat logs/slurm_eval_debug_$SLURM_JOB_ID.err"
    echo "  cat logs/evaluation_debug_${SLURM_JOB_ID}.log"
fi

echo ""
echo "Full evaluation log saved to: logs/evaluation_debug_${SLURM_JOB_ID}.log"
echo "===================================="

exit $EVAL_EXIT_CODE