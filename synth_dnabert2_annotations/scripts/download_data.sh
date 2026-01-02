#!/bin/bash
################################################################################
# Data Download Assistant for Synthetic TE Detection Pipeline
#
# This script provides interactive assistance for downloading required datasets:
#   - InpactorDB2 (TE database for Phase 1)
#   - FlyBase D. melanogaster genomes (for Phase 2 evaluation)
#
# Usage:
#   ./scripts/download_data.sh [OPTIONS]
#
# Options:
#   --all              Download all datasets (non-interactive)
#   --inpactordb2      Download only InpactorDB2
#   --flybase          Download only FlyBase genomes
#   --skip-verify      Skip file verification
#   --help             Show this help message
#
# Interactive Mode:
#   Without options, the script will ask which datasets to download
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

# Script directory (to find project root)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Data directories
PHASE1_DATA="${PROJECT_ROOT}/phase1/data"
PHASE2_DATA="${PROJECT_ROOT}/phase2/data"

# Default values
DOWNLOAD_ALL=false
DOWNLOAD_INPACTORDB2=false
DOWNLOAD_FLYBASE=false
SKIP_VERIFY=false
INTERACTIVE=true

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --all)
            DOWNLOAD_ALL=true
            INTERACTIVE=false
            shift
            ;;
        --inpactordb2)
            DOWNLOAD_INPACTORDB2=true
            INTERACTIVE=false
            shift
            ;;
        --flybase)
            DOWNLOAD_FLYBASE=true
            INTERACTIVE=false
            shift
            ;;
        --skip-verify)
            SKIP_VERIFY=true
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
echo -e "${BLUE}║  Data Download Assistant                                  ║${NC}"
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

# Function to print info messages
print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# Function to verify file size
verify_file() {
    local file_path=$1
    local min_size_mb=$2

    if [ ! -f "$file_path" ]; then
        return 1
    fi

    local file_size_mb=$(du -m "$file_path" | cut -f1)
    if [ "$file_size_mb" -lt "$min_size_mb" ]; then
        print_warning "File size ($file_size_mb MB) is smaller than expected ($min_size_mb MB)"
        return 1
    fi

    return 0
}

# ============================================================================
# Interactive Mode
# ============================================================================
if [ "$INTERACTIVE" = true ]; then
    print_section "Dataset Selection"

    echo "This script can download the following datasets:"
    echo ""
    echo "1. InpactorDB2 (~5.2 GB) - TE database for Phase 1"
    echo "   Required for: Synthetic genome generation"
    echo ""
    echo "2. FlyBase D. melanogaster genomes (~44 MB) - For Phase 2 evaluation"
    echo "   Required for: Model evaluation on real data"
    echo ""
    echo "Note: These files are already included in the repository:"
    echo "  - dmel-all-chromosome-r6.66.fasta.gz (41 MB)"
    echo "  - dmel-all-transposon-r6.66.fasta.gz (2.5 MB)"
    echo ""

    read -p "Download InpactorDB2? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        DOWNLOAD_INPACTORDB2=true
    fi

    echo ""
    read -p "Re-download FlyBase genomes? (usually not needed) (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        DOWNLOAD_FLYBASE=true
    fi

    if [ "$DOWNLOAD_INPACTORDB2" = false ] && [ "$DOWNLOAD_FLYBASE" = false ]; then
        echo ""
        print_info "No datasets selected for download. Exiting."
        exit 0
    fi
fi

# Set flags for --all option
if [ "$DOWNLOAD_ALL" = true ]; then
    DOWNLOAD_INPACTORDB2=true
    DOWNLOAD_FLYBASE=true
fi

# ============================================================================
# Check Prerequisites
# ============================================================================
print_section "Checking Prerequisites"

# Check wget or curl
DOWNLOAD_CMD=""
if command -v wget &> /dev/null; then
    DOWNLOAD_CMD="wget"
    print_success "wget found"
elif command -v curl &> /dev/null; then
    DOWNLOAD_CMD="curl"
    print_success "curl found"
else
    print_error "Neither wget nor curl found. Please install one of them."
    exit 1
fi

# Check gunzip
if command -v gunzip &> /dev/null; then
    print_success "gunzip found"
else
    print_warning "gunzip not found. Compressed files will not be decompressed."
fi

# Check disk space
AVAILABLE_SPACE=$(df -BG "$PROJECT_ROOT" | tail -1 | awk '{print $4}' | sed 's/G//')
if [ "$AVAILABLE_SPACE" -lt 10 ]; then
    print_error "Insufficient disk space. At least 10GB required, found ${AVAILABLE_SPACE}GB"
    exit 1
fi
print_success "Disk space: ${AVAILABLE_SPACE}GB available"

# ============================================================================
# Download InpactorDB2
# ============================================================================
if [ "$DOWNLOAD_INPACTORDB2" = true ]; then
    print_section "Downloading InpactorDB2"

    # Create directory
    mkdir -p "$PHASE1_DATA"

    print_info "InpactorDB2 is a comprehensive TE database"
    print_info "Source: https://github.com/simonorozcoarias/Inpactor2"
    print_info "Size: ~5.2 GB (compressed)"
    echo ""

    print_warning "IMPORTANT: InpactorDB2 download requires manual steps"
    echo ""
    echo "Please follow these steps:"
    echo ""
    echo "1. Visit the InpactorDB2 repository:"
    echo "   https://github.com/simonorozcoarias/Inpactor2"
    echo ""
    echo "2. Navigate to the database download section"
    echo ""
    echo "3. Download the FASTA file to:"
    echo "   $PHASE1_DATA/"
    echo ""
    echo "4. The file should be named: inpactordb2.fasta (or similar)"
    echo ""
    echo "5. If compressed (.gz), decompress it:"
    echo "   cd $PHASE1_DATA"
    echo "   gunzip inpactordb2.fasta.gz"
    echo ""

    read -p "Press Enter when download is complete (or Ctrl+C to skip)..."

    # Verify download
    if [ "$SKIP_VERIFY" = false ]; then
        echo ""
        echo "Searching for InpactorDB2 file..."

        INPACTORDB_FILE=$(find "$PHASE1_DATA" -maxdepth 1 -name "*inpactor*.fasta" -o -name "*inpactor*.fa" 2>/dev/null | head -1)

        if [ -n "$INPACTORDB_FILE" ]; then
            print_success "Found: $INPACTORDB_FILE"

            # Verify size (should be > 4 GB)
            if verify_file "$INPACTORDB_FILE" 4000; then
                print_success "File size verification passed"
            else
                print_warning "File may be incomplete or corrupted"
            fi
        else
            print_error "InpactorDB2 file not found in $PHASE1_DATA"
            echo "Expected filename pattern: *inpactor*.fasta"
        fi
    fi
fi

# ============================================================================
# Download FlyBase Genomes
# ============================================================================
if [ "$DOWNLOAD_FLYBASE" = true ]; then
    print_section "Downloading FlyBase D. melanogaster Genomes"

    # Create directory
    mkdir -p "$PHASE2_DATA"

    print_info "FlyBase genomes are used for model evaluation on real data"
    print_info "Source: FlyBase (Release r6.66)"
    print_info "Total size: ~44 MB (compressed)"
    echo ""

    # Note: These files are actually already in the repository
    print_warning "Note: These files are already included in the repository"
    print_warning "Re-downloading is usually not necessary"
    echo ""

    # File 1: Chromosomes
    CHROMOSOME_FILE="dmel-all-chromosome-r6.66.fasta.gz"
    CHROMOSOME_URL="ftp://ftp.flybase.net/releases/FB2025_01/dmel_r6.66/fasta/${CHROMOSOME_FILE}"

    echo "Downloading: $CHROMOSOME_FILE"
    echo "URL: $CHROMOSOME_URL"
    echo ""

    if [ -f "${PHASE2_DATA}/${CHROMOSOME_FILE}" ]; then
        print_warning "File already exists: ${PHASE2_DATA}/${CHROMOSOME_FILE}"
        read -p "Overwrite? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_info "Skipping $CHROMOSOME_FILE"
        else
            rm "${PHASE2_DATA}/${CHROMOSOME_FILE}"
        fi
    fi

    if [ ! -f "${PHASE2_DATA}/${CHROMOSOME_FILE}" ]; then
        cd "$PHASE2_DATA"

        if [ "$DOWNLOAD_CMD" = "wget" ]; then
            wget -c "$CHROMOSOME_URL" || {
                print_error "Download failed. Check URL or network connection."
                print_info "Manual download: $CHROMOSOME_URL"
            }
        else
            curl -C - -O "$CHROMOSOME_URL" || {
                print_error "Download failed. Check URL or network connection."
                print_info "Manual download: $CHROMOSOME_URL"
            }
        fi

        # Verify
        if [ "$SKIP_VERIFY" = false ] && [ -f "$CHROMOSOME_FILE" ]; then
            if verify_file "$CHROMOSOME_FILE" 30; then
                print_success "Chromosome file downloaded successfully"
            fi
        fi
    fi

    echo ""

    # File 2: Transposons
    TRANSPOSON_FILE="dmel-all-transposon-r6.66.fasta.gz"
    TRANSPOSON_URL="ftp://ftp.flybase.net/releases/FB2025_01/dmel_r6.66/fasta/${TRANSPOSON_FILE}"

    echo "Downloading: $TRANSPOSON_FILE"
    echo "URL: $TRANSPOSON_URL"
    echo ""

    if [ -f "${PHASE2_DATA}/${TRANSPOSON_FILE}" ]; then
        print_warning "File already exists: ${PHASE2_DATA}/${TRANSPOSON_FILE}"
        read -p "Overwrite? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_info "Skipping $TRANSPOSON_FILE"
        else
            rm "${PHASE2_DATA}/${TRANSPOSON_FILE}"
        fi
    fi

    if [ ! -f "${PHASE2_DATA}/${TRANSPOSON_FILE}" ]; then
        cd "$PHASE2_DATA"

        if [ "$DOWNLOAD_CMD" = "wget" ]; then
            wget -c "$TRANSPOSON_URL" || {
                print_error "Download failed. Check URL or network connection."
                print_info "Manual download: $TRANSPOSON_URL"
            }
        else
            curl -C - -O "$TRANSPOSON_URL" || {
                print_error "Download failed. Check URL or network connection."
                print_info "Manual download: $TRANSPOSON_URL"
            }
        fi

        # Verify
        if [ "$SKIP_VERIFY" = false ] && [ -f "$TRANSPOSON_FILE" ]; then
            if verify_file "$TRANSPOSON_FILE" 2; then
                print_success "Transposon file downloaded successfully"
            fi
        fi
    fi

    cd "$PROJECT_ROOT"
fi

# ============================================================================
# Summary
# ============================================================================
print_section "Download Summary"

echo "Data directories:"
echo "  Phase 1 data: $PHASE1_DATA"
echo "  Phase 2 data: $PHASE2_DATA"
echo ""

if [ "$DOWNLOAD_INPACTORDB2" = true ]; then
    echo "InpactorDB2:"
    if [ -n "$INPACTORDB_FILE" ]; then
        echo "  ✓ File: $(basename "$INPACTORDB_FILE")"
        echo "  ✓ Location: $(dirname "$INPACTORDB_FILE")"
    else
        echo "  ⚠ Manual download required (see instructions above)"
    fi
    echo ""
fi

if [ "$DOWNLOAD_FLYBASE" = true ]; then
    echo "FlyBase genomes:"
    if [ -f "${PHASE2_DATA}/${CHROMOSOME_FILE}" ]; then
        echo "  ✓ $CHROMOSOME_FILE"
    else
        echo "  ⚠ $CHROMOSOME_FILE (not downloaded or already in repository)"
    fi
    if [ -f "${PHASE2_DATA}/${TRANSPOSON_FILE}" ]; then
        echo "  ✓ $TRANSPOSON_FILE"
    else
        echo "  ⚠ $TRANSPOSON_FILE (not downloaded or already in repository)"
    fi
    echo ""
fi

echo ""
echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Download Assistant Complete                             ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

echo "Next steps:"
echo ""

if [ "$DOWNLOAD_INPACTORDB2" = true ]; then
    echo "Phase 1: Synthetic Genome Generation"
    echo "  1. Verify InpactorDB2 location in phase1/data/"
    echo "  2. Follow Phase 1 workflow: phase1/README.md"
    echo "  3. Run species extraction and indexing scripts"
    echo ""
fi

if [ "$DOWNLOAD_FLYBASE" = true ]; then
    echo "Phase 2: Model Training and Evaluation"
    echo "  1. Verify FlyBase genomes are in phase2/data/"
    echo "  2. These are used for real data evaluation"
    echo "  3. See: phase2/README.md, Phase 2, Step 6"
    echo ""
fi

echo "For more information:"
echo "  - Data sources: docs/DATA_SOURCES.md"
echo "  - Installation: INSTALL.md"
echo "  - Reproducibility: docs/REPRODUCIBILITY.md"
echo ""

print_success "Download process complete!"
