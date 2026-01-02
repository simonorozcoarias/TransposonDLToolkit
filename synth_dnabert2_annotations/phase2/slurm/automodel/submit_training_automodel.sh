#!/bin/bash
#SBATCH --job-name=automodel_te_finetune
#SBATCH --output=logs/slurm_automodel_%j.out
#SBATCH --error=logs/slurm_automodel_%j.err
#SBATCH --mail-type END
#SBATCH --mail-user jgilbaja@uoc.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:3g.20gb:1
#SBATCH --mem=64G
#SBATCH --time=168:00:00  # 7 days - enough for full training with eval every 2h
#SBATCH --partition=gpu
#SBATCH --account=inpactor3

################################################################################
# AutoModel Token Classification Fine-tuning - SLURM Job Script
#
# This script submits a fine-tuning job for AutoModel on a SLURM cluster.
# It handles environment setup, multi-GPU training, and TensorBoard access.
#
# Usage:
#   sbatch submit_training.sh
#   sbatch submit_training.sh --debug  # For testing with small data
#
# Monitor job:
#   squeue -u $USER
#   tail -f logs/slurm_<job_id>.out
#
# Cancel job:
#   scancel <job_id>
################################################################################

echo "=================================="
echo "AutoModel Fine-tuning Job Started"
echo "=================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Number of GPUs: $SLURM_GPUS"
echo "Number of CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Start time: $(date)"
echo "=================================="

# Create logs directory if it doesn't exist
mkdir -p logs

# ============================================================
# Environment Setup
# ============================================================
echo ""
echo "Setting up environment..."

# Load required modules (adjust according to your cluster)
# Uncomment and modify these lines based on your cluster configuration
# module purge
# module load gcc/11.2.0
# module load cuda/11.8
# module load cudnn/8.6
# module load python/3.10

# Activate conda environment
source ~/anaconda3/bin/activate DNABERT2

echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "Number of GPUs: $(python -c 'import torch; print(torch.cuda.device_count())')"

# Set environment variables for optimal performance
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false  # Avoid warnings from HuggingFace tokenizers

# CUDA memory optimization to avoid fragmentation (fixes OOM during eval)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# NCCL settings for multi-GPU training (adjust if needed)
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=3

# HuggingFace cache directory (adjust to your scratch/fast storage)
# export HF_HOME=/scratch/$USER/.cache/huggingface
# export TRANSFORMERS_CACHE=/scratch/$USER/.cache/huggingface/transformers

echo "Environment setup complete."
echo ""

# ============================================================
# Training Configuration
# ============================================================

# Project directory (adjust to your actual path on the cluster)
PROJECT_DIR="$HOME/inpactor3/auto_detection/phase2"

# Navigate to project directory
cd $PROJECT_DIR || { echo "Error: Cannot access project directory"; exit 1; }
echo "Working directory: $(pwd)"

# Configuration file
CONFIG_FILE="scripts/config.yaml"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file '$CONFIG_FILE' not found!"
    exit 1
fi

# Parse command line arguments
DEBUG_MODE=""
if [[ "$1" == "--debug" ]]; then
    DEBUG_MODE="--debug"
    echo "Running in DEBUG mode with limited data"
fi

# ============================================================
# Verify Dataset Exists
# ============================================================
echo ""
echo "Verifying dataset..."
stdbuf -oL -eL python scripts/verify_before_training.py --config $CONFIG_FILE || {
    echo "Error: Dataset verification failed!"
    exit 1
}
echo "Dataset verification passed."
echo ""

# ============================================================
# Start TensorBoard in Background
# ============================================================
echo "Starting TensorBoard server..."

# TensorBoard logs directory
TB_LOGDIR="results/automodel_te_token_classification/logs"
mkdir -p $TB_LOGDIR

# Start TensorBoard on a specific port
TB_PORT=6006
tensorboard --logdir=$TB_LOGDIR --port=$TB_PORT --host=0.0.0.0 &
TB_PID=$!

echo "TensorBoard started (PID: $TB_PID) on port $TB_PORT"
echo ""
echo "To access TensorBoard from your local machine, run this SSH tunnel:"
echo "  ssh -L $TB_PORT:$SLURM_NODELIST:$TB_PORT $USER@<cluster_address>"
echo "Then open in browser: http://localhost:$TB_PORT"
echo ""

# ============================================================
# Multi-GPU Training with torchrun
# ============================================================
echo "=================================="
echo "Starting training..."
echo "=================================="

# Number of GPUs - Use SLURM_GPUS_ON_NODE or SLURM_NTASKS_PER_NODE, fallback to 1
if [ ! -z "$SLURM_GPUS_ON_NODE" ]; then
    NUM_GPUS=$SLURM_GPUS_ON_NODE
elif [ ! -z "$SLURM_GPUS" ]; then
    NUM_GPUS=$SLURM_GPUS
else
    # Fallback: check actual GPUs available
    NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    if [ "$NUM_GPUS" -eq 0 ]; then
        NUM_GPUS=1
    fi
fi

echo "Using $NUM_GPUS GPU(s) for training"

# Launch training with torchrun (recommended for multi-GPU)
# torchrun automatically handles distributed setup
# Use stdbuf to disable output buffering for real-time logs
stdbuf -oL -eL torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=$NUM_GPUS \
    scripts/train_token_classification_automodel.py \
    --config $CONFIG_FILE \
    $DEBUG_MODE

# Capture exit code
TRAINING_EXIT_CODE=$?

# ============================================================
# Cleanup and Summary
# ============================================================
echo ""
echo "=================================="
echo "Training completed with exit code: $TRAINING_EXIT_CODE"
echo "=================================="

# Kill TensorBoard
if [ ! -z "$TB_PID" ]; then
    echo "Stopping TensorBoard (PID: $TB_PID)..."
    kill $TB_PID 2>/dev/null
fi

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

# Check if training was successful
if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "=================================="
    echo "SUCCESS: Training completed successfully!"
    echo "=================================="
    echo ""
    echo "Next steps:"
    echo "  1. View training logs: cat logs/slurm_$SLURM_JOB_ID.out"
    echo "  2. View TensorBoard logs locally: tensorboard --logdir=$TB_LOGDIR"
    echo "  3. Evaluate on test set: python evaluate_model_automodel.py --model_path results/automodel_te_token_classification"
    echo ""
else
    echo ""
    echo "=================================="
    echo "ERROR: Training failed with exit code $TRAINING_EXIT_CODE"
    echo "=================================="
    echo ""
    echo "Check logs for details:"
    echo "  cat logs/slurm_$SLURM_JOB_ID.err"
    echo ""
fi

exit $TRAINING_EXIT_CODE
