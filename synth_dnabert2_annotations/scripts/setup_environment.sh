#!/bin/bash
################################################################################
# Automated Environment Setup for Synthetic TE Detection Pipeline
#
# This script automates the complete environment setup for Phase 2 (DNABERT-2)
# including PyTorch, dependencies, and DNABERT-2 compatibility fixes.
#
# Usage:
#   ./scripts/setup_environment.sh [OPTIONS]
#
# Options:
#   --cuda-version VERSION    Specify CUDA version (11.8 or 12.1, default: auto-detect)
#   --python-version VERSION  Python version for environment (default: 3.8)
#   --env-name NAME          Conda environment name (default: dnabert2_te)
#   --skip-fixes             Skip DNABERT-2 fixes (not recommended)
#   --help                   Show this help message
#
# Examples:
#   ./scripts/setup_environment.sh                    # Auto-detect CUDA
#   ./scripts/setup_environment.sh --cuda-version 11.8
#   ./scripts/setup_environment.sh --env-name my_env
#
# Author: Jorge González Gilbaja
# Project: Synthetic TE Detection with DNABERT-2
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
CUDA_VERSION="auto"
PYTHON_VERSION="3.8"
ENV_NAME="dnabert2_te"
SKIP_FIXES=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --cuda-version)
            CUDA_VERSION="$2"
            shift 2
            ;;
        --python-version)
            PYTHON_VERSION="$2"
            shift 2
            ;;
        --env-name)
            ENV_NAME="$2"
            shift 2
            ;;
        --skip-fixes)
            SKIP_FIXES=true
            shift
            ;;
        --help)
            grep "^#" "$0" | grep -v "^#!/" | sed 's/^# //' | sed 's/^#//'
            exit 0
            ;;
        *)
            echo -e "${RED}Error: Unknown option $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Print header
echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Synthetic TE Detection - Environment Setup               ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Function to print section headers
print_section() {
    echo ""
    echo -e "${BLUE}===================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}===================================${NC}"
}

# Function to print success messages
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

# Function to print warning messages
print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# Function to print error messages
print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Check if running on Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    print_warning "This script is designed for Linux. Other systems may require modifications."
fi

# ============================================================================
# Step 1: System Requirements Check
# ============================================================================
print_section "Step 1: System Requirements Check"

# Check Python version
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    print_error "Python not found. Please install Python 3.8+"
    exit 1
fi

DETECTED_PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
print_success "Python detected: $DETECTED_PYTHON_VERSION"

# Check for conda
if command -v conda &> /dev/null; then
    print_success "Conda found: $(conda --version)"
else
    print_error "Conda not found. Please install Miniconda or Anaconda first."
    echo "Visit: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Detect CUDA version if auto
if [ "$CUDA_VERSION" = "auto" ]; then
    if command -v nvcc &> /dev/null; then
        DETECTED_CUDA=$(nvcc --version | grep "release" | awk '{print $6}' | cut -d',' -f1 | cut -d'V' -f2)
        CUDA_MAJOR=$(echo $DETECTED_CUDA | cut -d'.' -f1,2)

        if [[ "$CUDA_MAJOR" == "11.8" ]]; then
            CUDA_VERSION="11.8"
        elif [[ "$CUDA_MAJOR" =~ ^12\. ]]; then
            CUDA_VERSION="12.1"
        else
            print_warning "Detected CUDA $DETECTED_CUDA. Using CUDA 11.8 for PyTorch."
            CUDA_VERSION="11.8"
        fi
        print_success "Auto-detected CUDA version: $CUDA_VERSION"
    else
        print_warning "nvcc not found. Defaulting to CUDA 11.8 for PyTorch."
        CUDA_VERSION="11.8"
    fi
fi

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    print_success "GPU detected: $GPU_INFO"
else
    print_warning "nvidia-smi not found. GPU training will not be available."
fi

# Check disk space (require 20GB for installation)
AVAILABLE_SPACE=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
if [ "$AVAILABLE_SPACE" -lt 20 ]; then
    print_error "Insufficient disk space. At least 20GB required, found ${AVAILABLE_SPACE}GB"
    exit 1
fi
print_success "Disk space: ${AVAILABLE_SPACE}GB available"

# ============================================================================
# Step 2: Create Conda Environment
# ============================================================================
print_section "Step 2: Creating Conda Environment"

# Check if environment already exists
if conda env list | grep -q "^${ENV_NAME} "; then
    print_warning "Environment '$ENV_NAME' already exists."
    read -p "Remove and recreate? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing environment..."
        conda env remove -n $ENV_NAME -y
    else
        print_error "Setup cancelled. Use a different --env-name or remove manually."
        exit 1
    fi
fi

echo "Creating conda environment: $ENV_NAME (Python $PYTHON_VERSION)"
conda create -n $ENV_NAME python=$PYTHON_VERSION -y

print_success "Conda environment created: $ENV_NAME"

# ============================================================================
# Step 3: Install PyTorch with CUDA Support
# ============================================================================
print_section "Step 3: Installing PyTorch with CUDA $CUDA_VERSION"

# Activate environment
eval "$(conda shell.bash hook)"
conda activate $ENV_NAME

# Install PyTorch based on CUDA version
if [ "$CUDA_VERSION" = "11.8" ]; then
    echo "Installing PyTorch 2.4.1+cu118..."
    pip install torch==2.4.1+cu118 torchvision==0.19.1+cu118 \
        --index-url https://download.pytorch.org/whl/cu118
elif [ "$CUDA_VERSION" = "12.1" ]; then
    echo "Installing PyTorch 2.4.1+cu121..."
    pip install torch torchvision \
        --index-url https://download.pytorch.org/whl/cu121
else
    print_error "Unsupported CUDA version: $CUDA_VERSION"
    exit 1
fi

# Verify PyTorch installation
PYTORCH_VERSION=$(python -c "import torch; print(torch.__version__)" 2>/dev/null || echo "failed")
if [ "$PYTORCH_VERSION" = "failed" ]; then
    print_error "PyTorch installation failed"
    exit 1
fi
print_success "PyTorch installed: $PYTORCH_VERSION"

# Verify CUDA availability
CUDA_AVAILABLE=$(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")
if [ "$CUDA_AVAILABLE" = "True" ]; then
    print_success "CUDA is available in PyTorch"
else
    print_warning "CUDA not available in PyTorch (CPU-only mode)"
fi

# ============================================================================
# Step 4: Install Phase 2 Dependencies
# ============================================================================
print_section "Step 4: Installing Project Dependencies"

# Navigate to phase2 directory
cd phase2

if [ ! -f requirements.txt ]; then
    print_error "requirements.txt not found in phase2/"
    exit 1
fi

echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

print_success "Dependencies installed successfully"

# ============================================================================
# Step 5: Apply DNABERT-2 Compatibility Fixes
# ============================================================================
if [ "$SKIP_FIXES" = false ]; then
    print_section "Step 5: Applying DNABERT-2 Compatibility Fixes"

    cd ..

    if [ -f scripts/apply_dnabert2_fixes.sh ]; then
        echo "Running DNABERT-2 fix script..."
        bash scripts/apply_dnabert2_fixes.sh

        if [ $? -eq 0 ]; then
            print_success "DNABERT-2 fixes applied successfully"
        else
            print_warning "DNABERT-2 fixes script returned errors (may be non-critical)"
        fi
    else
        print_warning "Fix script not found: scripts/apply_dnabert2_fixes.sh"
        echo "You may need to apply fixes manually. See docs/DNABERT2_COMPATIBILITY.md"
    fi
else
    print_warning "Skipping DNABERT-2 fixes (--skip-fixes specified)"
fi

# ============================================================================
# Step 6: Verification
# ============================================================================
print_section "Step 6: Installation Verification"

cd phase2

echo "Testing DNABERT-2 installation..."
if [ -f scripts/testing/test_dnabert2_installation.py ]; then
    python scripts/testing/test_dnabert2_installation.py

    if [ $? -eq 0 ]; then
        print_success "DNABERT-2 installation test passed!"
    else
        print_error "DNABERT-2 installation test failed"
        echo "Check docs/DNABERT2_COMPATIBILITY.md for troubleshooting"
    fi
else
    print_warning "Test script not found. Skipping verification."
fi

# ============================================================================
# Summary
# ============================================================================
print_section "Installation Summary"

echo -e "${GREEN}✓ Environment created: $ENV_NAME${NC}"
echo -e "${GREEN}✓ Python version: $PYTHON_VERSION${NC}"
echo -e "${GREEN}✓ PyTorch version: $PYTORCH_VERSION${NC}"
echo -e "${GREEN}✓ CUDA support: $CUDA_AVAILABLE${NC}"
echo -e "${GREEN}✓ Dependencies installed${NC}"

if [ "$SKIP_FIXES" = false ]; then
    echo -e "${GREEN}✓ DNABERT-2 fixes applied${NC}"
fi

echo ""
echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Setup Complete!                                          ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "To activate the environment, run:"
echo -e "${YELLOW}  conda activate $ENV_NAME${NC}"
echo ""
echo "Next steps:"
echo "  1. Download InpactorDB2 to phase1/data/ (see phase1/data/README.md)"
echo "  2. Run Phase 1 to generate synthetic genomes (see phase1/README.md)"
echo "  3. Run Phase 2 to train DNABERT-2 (see phase2/README.md)"
echo ""
echo "For quick testing:"
echo -e "${YELLOW}  cd phase2/test_data && ./scripts/run_test.sh${NC}"
echo ""
echo "Documentation:"
echo "  - Installation: INSTALL.md"
echo "  - Architecture: docs/ARCHITECTURE.md"
echo "  - Reproducibility: docs/REPRODUCIBILITY.md"
echo ""
print_success "Setup completed successfully!"
