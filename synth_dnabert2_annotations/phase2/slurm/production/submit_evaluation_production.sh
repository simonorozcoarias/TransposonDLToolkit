#!/bin/bash
#SBATCH --job-name=prod_eval_40sp
#SBATCH --output=logs/slurm_prod_eval_%j.out
#SBATCH --error=logs/slurm_prod_eval_%j.err
#SBATCH --mail-type END
#SBATCH --mail-user=REPLACE_WITH_YOUR_EMAIL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:3g.20gb:1
#SBATCH --mem=32G
#SBATCH --time=4:00:00  # 4 hours for 40-species evaluation
#SBATCH --partition=REPLACE_WITH_YOUR_PARTITION
#SBATCH --account=REPLACE_WITH_YOUR_ACCOUNT

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

# PRODUCTION MODEL EVALUATION - 40 Species Multi-Kingdom Model

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

#

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

# Purpose: Evaluate the definitive DNABERT-2 model trained with

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

#          submit_training_production.sh

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

#

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

# This script evaluates models using standard HuggingFace methods with:

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

# 1. Model loading verification with from_pretrained()

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

# 2. Evaluation on multi-species test sets

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

# 3. Per-species metrics (optional with --by_species flag)

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

# 4. Detailed performance analysis

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

#

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

# Parameters:

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

#   $1 MODEL_PATH    - Path to trained model directory (relative to project root)

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

#                      Default: results/production_40_species

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

#                      Can also be a specific checkpoint: results/production_40_species/checkpoint-5000

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

#

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

#   $2 DATASET_PATH  - Path to combined dataset directory (relative to project root)

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

#                      Default: auto-detect most recent 40_species_production dataset

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

#                      Example: datasets_combined/40_species_production_20251223_154530

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

#

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

#   $3 --by_species  - Optional flag to generate per-species metrics

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

#

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

# Usage Examples:

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

#   # Evaluate with defaults (finds latest dataset automatically):

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

#   sbatch submit_evaluation_production.sh

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

#

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

#   # Evaluate specific checkpoint with auto-detected dataset:

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

#   sbatch submit_evaluation_production.sh results/production_40_species/checkpoint-5000

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

#

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

#   # Evaluate with specific dataset:

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

#   sbatch submit_evaluation_production.sh results/production_40_species datasets_combined/40_species_production_20251223_154530

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

#

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

#   # Evaluate with per-species metrics:

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

#   sbatch submit_evaluation_production.sh results/production_40_species datasets_combined/40_species_production_20251223_154530 --by_species

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

#

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

#   # Evaluate best checkpoint with per-species breakdown:

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

#   sbatch submit_evaluation_production.sh results/production_40_species datasets_combined/40_species_production_20251223_154530 --by_species

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

#

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

# Notes:

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

#   - All paths should be relative to ${PHASE2_DIR}

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

#   - The dataset must match the one used during training

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

#   - For best results, use the combined dataset that was created during training

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

#   - The config file will be temporarily modified and restored after evaluation

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

#   - If no dataset path is provided, the script will auto-detect the most recent

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

#     40_species_production dataset in datasets_combined/

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
echo "PRODUCTION EVALUATION - 40 Species"
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
source conda activate ${CONDA_ENV}

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
PROJECT_DIR="${PHASE2_DIR}"

# Navigate to project directory
cd ${PHASE2_DIR} || { echo "Error: Cannot access project directory"; exit 1; }
echo "Working directory: $(pwd)"

# Parse command line arguments
# Usage: sbatch submit_evaluation_production.sh [MODEL_PATH] [DATASET_PATH] [--by_species]
MODEL_PATH="${1:-results/production_40_species}"
DATASET_PATH_ARG="${2:-}"
BY_SPECIES_FLAG=""

# Check if second argument is --by_species flag
if [[ "$DATASET_PATH_ARG" == "--by_species" ]]; then
    BY_SPECIES_FLAG="--by_species"
    DATASET_PATH_ARG=""
fi

# Check remaining arguments for --by_species flag
shift 2 2>/dev/null || shift $# 2>/dev/null
for arg in "$@"; do
    if [[ "$arg" == "--by_species" ]]; then
        BY_SPECIES_FLAG="--by_species"
    fi
done

# Auto-detect dataset if not provided
if [ -z "$DATASET_PATH_ARG" ]; then
    echo ""
    echo "No dataset path provided. Auto-detecting most recent 40_species_production dataset..."

    # Find the most recent 40_species_production dataset
    LATEST_DATASET=$(find datasets_combined -maxdepth 1 -type d -name "40_species_production_*" 2>/dev/null | sort -V | tail -1)

    if [ -z "$LATEST_DATASET" ]; then
        echo "❌ Error: No 40_species_production dataset found in datasets_combined/"
        echo ""
        echo "Available datasets:"
        ls -ld datasets_combined/*/ 2>/dev/null || echo "  (none found)"
        echo ""
        echo "Please specify dataset path manually:"
        echo "  sbatch submit_evaluation_production.sh $MODEL_PATH <dataset_path>"
        exit 1
    fi

    DATASET_PATH="$LATEST_DATASET"
    echo "✅ Auto-detected dataset: $DATASET_PATH"
else
    DATASET_PATH="$DATASET_PATH_ARG"
fi

echo ""
echo "Evaluation configuration:"
echo "  Model path: $MODEL_PATH"
echo "  Dataset path: $DATASET_PATH"
if [ ! -z "$BY_SPECIES_FLAG" ]; then
    echo "  Per-species metrics: ENABLED"
else
    echo "  Per-species metrics: DISABLED"
fi

# Configuration file
CONFIG_FILE="scripts/config_production.yaml"

# ============================================================
# Pre-flight Checks
# ============================================================
echo ""
echo "===================================="
echo "PRE-FLIGHT CHECKS"
echo "===================================="

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Error: Configuration file '$CONFIG_FILE' not found!"
    echo "Expected location: ${PHASE2_DIR}/$CONFIG_FILE"
    exit 1
fi
echo "✅ Config file found: $CONFIG_FILE"

# Check if model directory exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ Error: Model directory '$MODEL_PATH' not found!"
    echo ""
    echo "Please ensure training has completed successfully."
    echo "Run: sbatch submit_training_production.sh"
    echo ""
    echo "Or specify a valid checkpoint directory:"
    echo "  sbatch submit_evaluation_production.sh results/production_40_species/checkpoint-5000"
    exit 1
fi
echo "✅ Model directory found: $MODEL_PATH"

# Check if dataset exists
if [ ! -d "$DATASET_PATH" ]; then
    echo "❌ Error: Dataset directory '$DATASET_PATH' not found!"
    echo ""
    echo "Available combined datasets:"
    ls -ld datasets_combined/*/ 2>/dev/null || echo "  (none found)"
    echo ""
    echo "Please specify a valid dataset path:"
    echo "  sbatch submit_evaluation_production.sh $MODEL_PATH <dataset_path>"
    exit 1
fi
echo "✅ Dataset directory found: $DATASET_PATH"

# Verify dataset has required splits
echo ""
echo "Checking dataset structure..."
DATASET_VALID=true

for split in train val test; do
    if [ -d "$DATASET_PATH/$split" ]; then
        NUM_FILES=$(find "$DATASET_PATH/$split" -maxdepth 1 -type f -name "*.jsonl" 2>/dev/null | wc -l)
        echo "  ✅ $split split: $NUM_FILES species files"
    else
        echo "  ❌ $split split: MISSING"
        DATASET_VALID=false
    fi
done

if [ "$DATASET_VALID" = false ]; then
    echo ""
    echo "❌ Error: Dataset structure is incomplete"
    echo "Expected structure:"
    echo "  $DATASET_PATH/train/*.jsonl"
    echo "  $DATASET_PATH/val/*.jsonl"
    echo "  $DATASET_PATH/test/*.jsonl"
    exit 1
fi

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
    echo ""
    echo "If you want to evaluate a specific checkpoint, try:"
    echo "  sbatch submit_evaluation_production.sh results/production_40_species/checkpoint-XXXX"
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
# Use relative path for portability
RELATIVE_DATASET_PATH=$(realpath --relative-to="${PHASE2_DIR}" "$DATASET_PATH")
sed -i "s|dataset_path:.*|dataset_path: \"$RELATIVE_DATASET_PATH\"|" "$CONFIG_FILE"

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
echo "Dataset: $DATASET_PATH"
echo "Config: $CONFIG_FILE"
echo "Batch size: $BATCH_SIZE"
if [ ! -z "$BY_SPECIES_FLAG" ]; then
    echo "Per-species metrics: YES"
fi
echo ""

# Run evaluation script with unbuffered output
python -u scripts/evaluation/evaluate_model_automodel.py \
    --model_path $MODEL_PATH \
    --config $CONFIG_FILE \
    --batch_size $BATCH_SIZE \
    $BY_SPECIES_FLAG \
    2>&1 | tee -a logs/evaluation_production_${SLURM_JOB_ID}.log

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
        echo "╔════════════════════════════════════════════╗"
        echo "║     PRODUCTION MODEL TEST RESULTS          ║"
        echo "╚════════════════════════════════════════════╝"
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
        echo ""
        echo "Top 5 species by F1 score:"
        python -c "
import json
with open('$EVAL_DIR/test_results_by_species.json', 'r') as f:
    results = json.load(f)
# Sort by f1 score
sorted_species = sorted(results.items(), key=lambda x: x[1].get('f1', 0), reverse=True)
for i, (species, metrics) in enumerate(sorted_species[:5], 1):
    f1 = metrics.get('f1', 0)
    precision = metrics.get('precision', 0)
    recall = metrics.get('recall', 0)
    print(f\"{i}. {species:30s} F1={f1:.4f} P={precision:.4f} R={recall:.4f}\")
" 2>/dev/null

        echo ""
        echo "Bottom 5 species by F1 score:"
        python -c "
import json
with open('$EVAL_DIR/test_results_by_species.json', 'r') as f:
    results = json.load(f)
# Sort by f1 score
sorted_species = sorted(results.items(), key=lambda x: x[1].get('f1', 0), reverse=True)
for i, (species, metrics) in enumerate(sorted_species[-5:], len(sorted_species)-4):
    f1 = metrics.get('f1', 0)
    precision = metrics.get('precision', 0)
    recall = metrics.get('recall', 0)
    print(f\"{i}. {species:30s} F1={f1:.4f} P={precision:.4f} R={recall:.4f}\")
" 2>/dev/null
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
echo "PRODUCTION EVALUATION SUMMARY"
echo "===================================="
echo "Model: $MODEL_PATH"
echo "Dataset: $DATASET_PATH"
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
    echo ""
    echo "TensorBoard logs (if available):"
    echo "  tensorboard --logdir $MODEL_PATH/logs"
else
    echo ""
    echo "❌ Evaluation failed with exit code $EVAL_EXIT_CODE"
    echo ""
    echo "Check logs for details:"
    echo "  cat logs/slurm_prod_eval_$SLURM_JOB_ID.err"
    echo "  cat logs/evaluation_production_${SLURM_JOB_ID}.log"
fi

echo ""
echo "Full evaluation log saved to: logs/evaluation_production_${SLURM_JOB_ID}.log"
echo "===================================="

exit $EVAL_EXIT_CODE
