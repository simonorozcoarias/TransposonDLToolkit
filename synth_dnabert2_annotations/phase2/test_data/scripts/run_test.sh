#!/bin/bash
#
# Quick test script for predict_te_regions.py
#
# This script runs a test prediction on the small test dataset.
# Adjust the MODEL_PATH to point to your fine-tuned DNABERT-2 model.

set -e  # Exit on error

# Configuration
MODEL_PATH="results/production_40_species_1/"  # UPDATE THIS PATH
TEST_DIR="data/test_prediction"
OUTPUT_DIR="output/test_predictions_$(date +%Y%m%d_%H%M%S)"

# Data selection: set to true to use REAL Drosophila data, false for synthetic data
USE_REAL_DATA=true

# Select data files based on USE_REAL_DATA flag
if [ "$USE_REAL_DATA" = true ]; then
    TEST_FASTA="$TEST_DIR/test_genome_real_dmel.fasta"
    TEST_GFF3="$TEST_DIR/test_annotations_real_dmel.gff3"
    DATA_TYPE="REAL Drosophila melanogaster"
else
    TEST_FASTA="$TEST_DIR/test_genome.fasta"
    TEST_GFF3="$TEST_DIR/test_annotations.gff3"
    DATA_TYPE="synthetic"
fi

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================"
echo "Testing predict_te_regions.py"
echo "========================================"
echo ""
echo -e "${GREEN}Data type:${NC} $DATA_TYPE"

# Check if test data exists
if [ ! -f "$TEST_FASTA" ] || [ ! -f "$TEST_GFF3" ]; then
    echo -e "${YELLOW}WARNING: Test data not found${NC}"
    if [ "$USE_REAL_DATA" = true ]; then
        echo "Real Drosophila data files not found."
        echo "Please run the extraction script first:"
        echo "  python $TEST_DIR/extract_real_dmel_data.py"
    else
        echo "Synthetic test data files not found."
        echo "Please run the extraction script first:"
        echo "  python $TEST_DIR/extract_test_data.py"
    fi
    exit 1
fi

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo -e "${YELLOW}WARNING: Model path not found: $MODEL_PATH${NC}"
    echo "Please update MODEL_PATH in this script to point to your fine-tuned model."
    echo ""
    echo "If you haven't trained a model yet, you need to run:"
    echo "  python train_dnabert2_te_classifier.py [options]"
    exit 1
fi

echo -e "${GREEN}Model found:${NC} $MODEL_PATH"
echo -e "${GREEN}Test FASTA:${NC} $TEST_FASTA"
echo -e "${GREEN}Test GFF3:${NC} $TEST_GFF3"
echo -e "${GREEN}Output:${NC} $OUTPUT_DIR"
echo ""

# Detect device
if command -v nvidia-smi &> /dev/null; then
    DEVICE="cuda"
    echo -e "${GREEN}GPU detected${NC} - using CUDA"
else
    DEVICE="cpu"
    echo -e "${YELLOW}No GPU detected${NC} - using CPU (slower)"
fi
echo ""

# Run prediction
echo "Running prediction..."
python predict_te_regions.py \
    --fasta "$TEST_FASTA" \
    --model_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --gff3 "$TEST_GFF3" \
    --batch_size 64 \
    --device "$DEVICE" \
    --window_size 2048 \
    --stride 1024 \
    --min_te_length 50 \
    --merge_gap 10 \
    --iou_threshold 0.5 \
    --visualize

echo ""
echo "========================================"
echo -e "${GREEN}Test completed successfully!${NC}"
echo "========================================"
echo ""
echo "Results are in: $OUTPUT_DIR"
echo ""
echo "Key files:"
echo "  - predictions.gff3            : Predicted TE regions"
echo "  - metrics/overall_metrics.json: Evaluation metrics"
echo "  - visualizations/             : Plots and figures"
echo ""
