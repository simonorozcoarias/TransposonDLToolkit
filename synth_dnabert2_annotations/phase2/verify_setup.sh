/#!/bin/bash
################################################################################
# Pre-Training Verification Script
# Verifies that all configurations are correct before submitting the job
################################################################################

echo "=========================================="
echo "DNABERT-2 TRAINING SETUP VERIFICATION"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

ERRORS=0
WARNINGS=0

# Check 1: GPU configuration in submit_training.sh
echo "1. Checking GPU configuration in submit_training.sh..."
if grep -q "#SBATCH --gres=gpu:3g.20gb:1" submit_training.sh; then
    echo -e "   ${GREEN}✓${NC} GPU configured correctly: 3g.20gb:1 (20GB GPU)"
else
    echo -e "   ${RED}✗${NC} GPU configuration incorrect or missing"
    ERRORS=$((ERRORS+1))
fi

# Check 2: Config file exists
echo ""
echo "2. Checking config.yaml exists..."
if [ -f "config.yaml" ]; then
    echo -e "   ${GREEN}✓${NC} config.yaml found"

    # Check num_gpus
    if grep -q "num_gpus: 1" config.yaml; then
        echo -e "   ${GREEN}✓${NC} num_gpus set to 1"
    else
        echo -e "   ${YELLOW}⚠${NC} num_gpus may not be set to 1"
        WARNINGS=$((WARNINGS+1))
    fi

    # Check batch size
    if grep -q "per_device_train_batch_size: 8" config.yaml; then
        echo -e "   ${GREEN}✓${NC} batch size: 8 (optimal for 20GB GPU)"
    else
        echo -e "   ${YELLOW}⚠${NC} batch size not set to 8"
        WARNINGS=$((WARNINGS+1))
    fi

    # Check gradient accumulation
    if grep -q "gradient_accumulation_steps: 4" config.yaml; then
        echo -e "   ${GREEN}✓${NC} gradient accumulation: 4 (global batch = 32)"
    else
        echo -e "   ${YELLOW}⚠${NC} gradient accumulation not set to 4"
        WARNINGS=$((WARNINGS+1))
    fi

    # Check early stopping patience
    if grep -q "patience: 15" config.yaml; then
        echo -e "   ${GREEN}✓${NC} early stopping patience: 15 (allows convergence)"
    else
        echo -e "   ${YELLOW}⚠${NC} early stopping patience not set to 15"
        WARNINGS=$((WARNINGS+1))
    fi
else
    echo -e "   ${RED}✗${NC} config.yaml not found!"
    ERRORS=$((ERRORS+1))
fi

# Check 3: Training script
echo ""
echo "3. Checking train_token_classification.py..."
if [ -f "train_token_classification.py" ]; then
    echo -e "   ${GREEN}✓${NC} train_token_classification.py found"

    # Check for full fine-tuning (no freezing)
    if grep -q "Full fine-tuning" train_token_classification.py; then
        echo -e "   ${GREEN}✓${NC} Full fine-tuning enabled (no parameter freezing)"
    else
        echo -e "   ${RED}✗${NC} Full fine-tuning not confirmed - check for parameter freezing!"
        ERRORS=$((ERRORS+1))
    fi
else
    echo -e "   ${RED}✗${NC} train_token_classification.py not found!"
    ERRORS=$((ERRORS+1))
fi

# Check 4: Dataset directory
echo ""
echo "4. Checking dataset..."
if [ -d "datasets/splits" ]; then
    echo -e "   ${GREEN}✓${NC} Dataset directory exists"

    # Check splits
    for split in train validation test; do
        if [ -d "datasets/splits/$split" ]; then
            echo -e "   ${GREEN}✓${NC} $split split found"
        else
            echo -e "   ${RED}✗${NC} $split split missing!"
            ERRORS=$((ERRORS+1))
        fi
    done
else
    echo -e "   ${RED}✗${NC} Dataset directory not found!"
    ERRORS=$((ERRORS+1))
fi

# Check 5: Logs directory
echo ""
echo "5. Checking logs directory..."
if [ -d "logs" ]; then
    echo -e "   ${GREEN}✓${NC} logs directory exists"
else
    echo -e "   ${YELLOW}⚠${NC} logs directory missing - will be created by script"
    WARNINGS=$((WARNINGS+1))
fi

# Check 6: GPU availability on cluster
echo ""
echo "6. Checking GPU availability (if on cluster)..."
if command -v sinfo &> /dev/null; then
    GPU_FREE=$(sinfo -p gpu -N -h -o "%N %t" | grep "idle\|mix" | grep "gpu-node-03" | wc -l)
    if [ "$GPU_FREE" -gt 0 ]; then
        echo -e "   ${GREEN}✓${NC} gpu-node-03 appears available"
    else
        echo -e "   ${YELLOW}⚠${NC} gpu-node-03 may be busy - check squeue"
        WARNINGS=$((WARNINGS+1))
    fi
else
    echo -e "   ${YELLOW}⚠${NC} Not on SLURM cluster - skipping GPU check"
fi

# Summary
echo ""
echo "=========================================="
echo "VERIFICATION SUMMARY"
echo "=========================================="

if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
    echo -e "${GREEN}✓ ALL CHECKS PASSED!${NC}"
    echo ""
    echo "Your setup is ready for training. To submit:"
    echo "  cd ~/inpactor3/auto_detection/phase2"
    echo "  sbatch submit_training.sh"
    exit 0
elif [ $ERRORS -eq 0 ]; then
    echo -e "${YELLOW}⚠ $WARNINGS warning(s) found${NC}"
    echo "You can proceed but review warnings above."
    exit 0
else
    echo -e "${RED}✗ $ERRORS error(s) found${NC}"
    echo -e "${YELLOW}⚠ $WARNINGS warning(s) found${NC}"
    echo ""
    echo "Please fix errors before submitting the job."
    exit 1
fi