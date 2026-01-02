#!/bin/bash
#SBATCH --job-name=dnabert2_te_eval
#SBATCH --output=logs/slurm_eval_%j.out
#SBATCH --error=logs/slurm_eval_%j.err
#SBATCH --mail-type END
#SBATCH --mail-user jgilbaja@uoc.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:3g.20gb:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00  # 24 hours should be enough for evaluation
#SBATCH --partition=gpu
#SBATCH --account=inpactor3

################################################################################
# DNABERT-2 Token Classification Model Evaluation - SLURM Job Script
#
# This script evaluates a fine-tuned DNABERT-2 model on the test set.
# It handles environment setup and GPU execution.
#
# Usage:
#   sbatch submit_evaluation.sh
#   sbatch submit_evaluation.sh --by_species  # For per-species evaluation
#
# Monitor job:
#   squeue -u $USER
#   tail -f logs/slurm_eval_<job_id>.out
#
# Cancel job:
#   scancel <job_id>
################################################################################

echo "==================================="
echo "DNABERT-2 Evaluation Job Started"
echo "==================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Number of GPUs: $SLURM_GPUS"
echo "Number of CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Start time: $(date)"
echo "==================================="

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
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false

# CUDA memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Environment setup complete."
echo ""

# ============================================================
# Evaluation Configuration
# ============================================================

# Project directory (adjust to your actual path on the cluster)
PROJECT_DIR="$HOME/inpactor3/auto_detection/phase2"

# Navigate to project directory
cd $PROJECT_DIR || { echo "Error: Cannot access project directory"; exit 1; }
echo "Working directory: $(pwd)"

# Configuration file
CONFIG_FILE="scripts/config.yaml"

# Model path (use the latest checkpoint or best model)
MODEL_PATH="results/dnabert2_te_token_classification"

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model directory '$MODEL_PATH' not found!"
    echo "Please ensure training has completed and model is saved."
    exit 1
fi

# Parse command line arguments
BY_SPECIES=""
if [[ "$1" == "--by_species" ]]; then
    BY_SPECIES="--by_species"
    echo "Running evaluation with per-species metrics"
fi

# Batch size for evaluation (can be larger than training since no gradients)
BATCH_SIZE=16

# ============================================================
# Run Evaluation
# ============================================================
echo ""
echo "==================================="
echo "Starting evaluation..."
echo "==================================="
echo "Model: $MODEL_PATH"
echo "Config: $CONFIG_FILE"
echo "Batch size: $BATCH_SIZE"
echo ""

# Run evaluation script
# Use stdbuf to disable output buffering for real-time logs
stdbuf -oL -eL python -u scripts/evaluate_model.py \
    --model_path $MODEL_PATH \
    --config $CONFIG_FILE \
    --batch_size $BATCH_SIZE \
    $BY_SPECIES

# Capture exit code
EVAL_EXIT_CODE=$?

# ============================================================
# Summary
# ============================================================
echo ""
echo "==================================="
echo "Evaluation completed with exit code: $EVAL_EXIT_CODE"
echo "==================================="

# Print job statistics
echo ""
echo "Job statistics:"
echo "  Job ID: $SLURM_JOB_ID"
echo "  Node: $SLURM_NODELIST"
echo "  Start time: $SLURM_JOB_START_TIME"
echo "  End time: $(date)"

# Print resource usage (if available)
if command -v sacct &> /dev/null; then
    echo ""
    echo "Resource usage:"
    sacct -j $SLURM_JOB_ID --format=JobID,JobName,Elapsed,MaxRSS,MaxVMSize,AveCPU,State
fi

# Check if evaluation was successful
if [ $EVAL_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "==================================="
    echo "SUCCESS: Evaluation completed successfully!"
    echo "==================================="
    echo ""
    echo "Results saved to: $MODEL_PATH/evaluation/"
    echo ""
    echo "View results:"
    echo "  cat $MODEL_PATH/evaluation/test_results.json"
    if [ ! -z "$BY_SPECIES" ]; then
        echo "  cat $MODEL_PATH/evaluation/test_results_by_species.json"
    fi
    echo ""
else
    echo ""
    echo "==================================="
    echo "ERROR: Evaluation failed with exit code $EVAL_EXIT_CODE"
    echo "==================================="
    echo ""
    echo "Check logs for details:"
    echo "  cat logs/slurm_eval_$SLURM_JOB_ID.err"
    echo ""
fi

exit $EVAL_EXIT_CODE