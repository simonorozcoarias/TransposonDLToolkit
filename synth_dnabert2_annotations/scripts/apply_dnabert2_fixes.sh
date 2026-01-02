#!/bin/bash
################################################################################
# DNABERT-2 FlashAttention Triton Compatibility Fixes
#
# This script automatically applies fixes for DNABERT-2 to work with modern
# Triton versions (2.2.0+). It modifies FlashAttention code to use the current
# Triton API instead of deprecated trans_b/trans_a parameters.
#
# Usage:
#   ./scripts/apply_dnabert2_fixes.sh [OPTIONS]
#
# Options:
#   --skip-download    Skip model download (assume already cached)
#   --verify-only      Only verify if fixes are applied (no modifications)
#   --help             Show this help message
#
# What it fixes:
#   - Line 191: tl.dot(q, k, trans_b=True) → tl.dot(q, tl.trans(k))
#   - Line 434: tl.dot(q, k, trans_b=True) → tl.dot(q, tl.trans(k))
#   - Line 494: tl.dot(p.to(do.dtype), do, trans_a=True) → tl.dot(tl.trans(p.to(do.dtype)), do)
#   - Line 501: tl.dot(do, v, trans_b=True) → tl.dot(do, tl.trans(v))
#   - Line 512: tl.dot(ds, q, trans_a=True) → tl.dot(tl.trans(ds), q)
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
SKIP_DOWNLOAD=false
VERIFY_ONLY=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-download)
            SKIP_DOWNLOAD=true
            shift
            ;;
        --verify-only)
            VERIFY_ONLY=true
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
echo -e "${BLUE}║  DNABERT-2 FlashAttention Triton Compatibility Fixes     ║${NC}"
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

# ============================================================================
# Step 1: Check Python and Dependencies
# ============================================================================
print_section "Step 1: Checking Environment"

# Check Python
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    print_error "Python not found. Please install Python 3.8+"
    exit 1
fi
print_success "Python found: $($PYTHON_CMD --version)"

# Check if transformers is installed
if $PYTHON_CMD -c "import transformers" &> /dev/null; then
    TRANSFORMERS_VERSION=$($PYTHON_CMD -c "import transformers; print(transformers.__version__)" 2>/dev/null)
    print_success "Transformers installed: $TRANSFORMERS_VERSION"
else
    print_error "Transformers not installed. Please install with: pip install transformers"
    exit 1
fi

# ============================================================================
# Step 2: Download DNABERT-2 Model (if needed)
# ============================================================================
if [ "$SKIP_DOWNLOAD" = false ]; then
    print_section "Step 2: Downloading DNABERT-2 Model"

    echo "Downloading DNABERT-2-117M to HuggingFace cache..."
    echo "This will create the necessary cache directories."

    $PYTHON_CMD -c "
from transformers import AutoTokenizer
print('Downloading model and tokenizer...')
try:
    tokenizer = AutoTokenizer.from_pretrained('zhihan1996/DNABERT-2-117M', trust_remote_code=True)
    print('Download complete!')
except Exception as e:
    print(f'Error: {e}')
    exit(1)
" || {
    print_error "Failed to download DNABERT-2 model"
    exit 1
}

    print_success "DNABERT-2 model cached successfully"
else
    print_section "Step 2: Skipping Model Download"
    print_warning "Assuming model is already cached"
fi

# ============================================================================
# Step 3: Locate FlashAttention Files
# ============================================================================
print_section "Step 3: Locating FlashAttention Files"

CACHE_DIR="$HOME/.cache/huggingface"
FILES_FOUND=()

# Search in both cache locations
echo "Searching for flash_attn_triton.py files..."

# Location 1: transformers_modules
MODULES_DIR="${CACHE_DIR}/modules/transformers_modules/zhihan1996"
if [ -d "$MODULES_DIR" ]; then
    while IFS= read -r -d '' file; do
        FILES_FOUND+=("$file")
        echo "  Found: $file"
    done < <(find "$MODULES_DIR" -name "flash_attn_triton.py" -print0 2>/dev/null)
fi

# Location 2: hub/models
HUB_DIR="${CACHE_DIR}/hub/models--zhihan1996--DNABERT-2-117M"
if [ -d "$HUB_DIR" ]; then
    while IFS= read -r -d '' file; do
        FILES_FOUND+=("$file")
        echo "  Found: $file"
    done < <(find "$HUB_DIR" -name "flash_attn_triton.py" -print0 2>/dev/null)
fi

# Check if files were found
if [ ${#FILES_FOUND[@]} -eq 0 ]; then
    print_error "No FlashAttention files found!"
    echo "Expected locations:"
    echo "  - $MODULES_DIR/*/flash_attn_triton.py"
    echo "  - $HUB_DIR/snapshots/*/flash_attn_triton.py"
    echo ""
    echo "Make sure DNABERT-2 model is downloaded. Try without --skip-download"
    exit 1
fi

print_success "Found ${#FILES_FOUND[@]} FlashAttention file(s)"

# ============================================================================
# Step 4: Verify or Apply Fixes
# ============================================================================
print_section "Step 4: $([ "$VERIFY_ONLY" = true ] && echo "Verifying" || echo "Applying") Fixes"

FIXED_COUNT=0
ALREADY_FIXED_COUNT=0
ERROR_COUNT=0

for file_path in "${FILES_FOUND[@]}"; do
    echo ""
    echo "Processing: $file_path"

    # Check if file exists and is readable/writable
    if [ ! -f "$file_path" ]; then
        print_error "File not found: $file_path"
        ((ERROR_COUNT++))
        continue
    fi

    if [ "$VERIFY_ONLY" = false ] && [ ! -w "$file_path" ]; then
        print_error "File not writable: $file_path"
        echo "Try running with sudo or changing file permissions"
        ((ERROR_COUNT++))
        continue
    fi

    # Check if already fixed
    if grep -q "tl\.trans(" "$file_path"; then
        print_success "Already fixed (contains tl.trans calls)"
        ((ALREADY_FIXED_COUNT++))
        continue
    fi

    # Check if needs fixing
    if ! grep -q "trans_b=True\|trans_a=True" "$file_path"; then
        print_warning "No trans_b/trans_a found (unexpected)"
        continue
    fi

    if [ "$VERIFY_ONLY" = true ]; then
        print_warning "Needs fixing (trans_b/trans_a still present)"
        ((ERROR_COUNT++))
        continue
    fi

    # Create backup
    BACKUP_FILE="${file_path}.backup"
    cp "$file_path" "$BACKUP_FILE"
    echo "  Created backup: $BACKUP_FILE"

    # Apply fixes with sed
    echo "  Applying Triton API fixes..."

    # Fix 1 & 2: tl.dot(q, k, trans_b=True) → tl.dot(q, tl.trans(k))
    sed -i 's/tl\.dot(q, k, trans_b=True)/tl.dot(q, tl.trans(k))/g' "$file_path"

    # Fix 3: tl.dot(p.to(do.dtype), do, trans_a=True) → tl.dot(tl.trans(p.to(do.dtype)), do)
    sed -i 's/tl\.dot(p\.to(do\.dtype), do, trans_a=True)/tl.dot(tl.trans(p.to(do.dtype)), do)/g' "$file_path"

    # Fix 4: tl.dot(do, v, trans_b=True) → tl.dot(do, tl.trans(v))
    sed -i 's/tl\.dot(do, v, trans_b=True)/tl.dot(do, tl.trans(v))/g' "$file_path"

    # Fix 5: tl.dot(ds, q, trans_a=True) → tl.dot(tl.trans(ds), q)
    sed -i 's/tl\.dot(ds, q, trans_a=True)/tl.dot(tl.trans(ds), q)/g' "$file_path"

    # Verify fixes were applied
    if grep -q "tl\.trans(" "$file_path" && ! grep -q "trans_b=True\|trans_a=True" "$file_path"; then
        print_success "Successfully applied all fixes"
        ((FIXED_COUNT++))
    else
        print_error "Fix verification failed!"
        echo "  Restoring backup..."
        mv "$BACKUP_FILE" "$file_path"
        ((ERROR_COUNT++))
    fi
done

# ============================================================================
# Summary
# ============================================================================
print_section "Fix Summary"

echo -e "${GREEN}Files processed: ${#FILES_FOUND[@]}${NC}"
if [ "$VERIFY_ONLY" = false ]; then
    echo -e "${GREEN}Files fixed: $FIXED_COUNT${NC}"
    echo -e "${GREEN}Already fixed: $ALREADY_FIXED_COUNT${NC}"
else
    echo -e "${GREEN}Already fixed: $ALREADY_FIXED_COUNT${NC}"
    echo -e "${YELLOW}Need fixing: $ERROR_COUNT${NC}"
fi

if [ $ERROR_COUNT -gt 0 ]; then
    echo -e "${RED}Errors: $ERROR_COUNT${NC}"
fi

echo ""
echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
if [ "$VERIFY_ONLY" = true ]; then
    if [ $ERROR_COUNT -eq 0 ]; then
        echo -e "${BLUE}║  Verification Complete - All Files Fixed!                ║${NC}"
    else
        echo -e "${BLUE}║  Verification Complete - Some Files Need Fixing          ║${NC}"
    fi
else
    if [ $FIXED_COUNT -gt 0 ] || [ $ALREADY_FIXED_COUNT -eq ${#FILES_FOUND[@]} ]; then
        echo -e "${BLUE}║  Fixes Applied Successfully!                             ║${NC}"
    else
        echo -e "${BLUE}║  Fix Application Complete with Errors                    ║${NC}"
    fi
fi
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"

if [ "$VERIFY_ONLY" = false ]; then
    echo ""
    echo "Next steps:"
    echo "  1. Test DNABERT-2 installation:"
    echo "     python phase2/scripts/testing/test_dnabert2_installation.py"
    echo ""
    echo "  2. If issues persist, check:"
    echo "     - PyTorch CUDA compatibility (docs/DNABERT2_COMPATIBILITY.md)"
    echo "     - Triton version (should be 2.2.0+)"
    echo ""
    print_success "DNABERT-2 fixes complete!"
fi

# Exit with appropriate code
if [ $ERROR_COUNT -gt 0 ] && [ $FIXED_COUNT -eq 0 ] && [ $ALREADY_FIXED_COUNT -eq 0 ]; then
    exit 1
else
    exit 0
fi
