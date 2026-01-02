# Automation Scripts

This directory contains utility scripts that automate common setup and maintenance tasks for the Synthetic TE Detection pipeline.

## Available Scripts

### 1. setup_environment.sh

**Purpose**: Automated environment setup for DNABERT-2 pipeline

**What it does**:
- Detects system configuration (Python, Conda, CUDA, GPU)
- Creates conda environment with correct Python version
- Installs PyTorch with appropriate CUDA support (auto-detected)
- Installs all Phase 2 dependencies
- Applies DNABERT-2 compatibility fixes
- Verifies installation with test suite

**Usage**:
```bash
# Basic usage (auto-detect everything)
./scripts/setup_environment.sh

# Specify CUDA version
./scripts/setup_environment.sh --cuda-version 11.8
./scripts/setup_environment.sh --cuda-version 12.1

# Custom environment name
./scripts/setup_environment.sh --env-name my_custom_env

# Custom Python version
./scripts/setup_environment.sh --python-version 3.9

# Skip DNABERT-2 fixes (not recommended)
./scripts/setup_environment.sh --skip-fixes

# Show help
./scripts/setup_environment.sh --help
```

**Requirements**:
- Conda or Miniconda installed
- CUDA Toolkit (for GPU support)
- At least 20GB free disk space

**Duration**: ~10-15 minutes (depending on download speeds)

**Output**:
- Conda environment: `dnabert2_te` (default)
- All dependencies installed
- DNABERT-2 fixes applied
- Verification test results

---

### 2. apply_dnabert2_fixes.sh

**Purpose**: Apply compatibility fixes for DNABERT-2 with modern PyTorch/Triton

**What it does**:
- Downloads DNABERT-2 model to HuggingFace cache (if not present)
- Locates FlashAttention files in both cache locations
- Applies Triton API fixes (trans_b/trans_a → tl.trans())
- Creates backups before modifying files
- Verifies fixes were applied correctly

**Usage**:
```bash
# Basic usage (download + fix)
./scripts/apply_dnabert2_fixes.sh

# Skip model download (assume already cached)
./scripts/apply_dnabert2_fixes.sh --skip-download

# Verify only (check if fixes are needed)
./scripts/apply_dnabert2_fixes.sh --verify-only

# Show help
./scripts/apply_dnabert2_fixes.sh --help
```

**When to use**:
- After fresh installation
- After clearing HuggingFace cache
- If you encounter `TypeError: dot() got an unexpected keyword argument 'trans_b'`
- When moving to a new system

**What it fixes**:
1. Line 191: `tl.dot(q, k, trans_b=True)` → `tl.dot(q, tl.trans(k))`
2. Line 434: `tl.dot(q, k, trans_b=True)` → `tl.dot(q, tl.trans(k))`
3. Line 494: `tl.dot(p.to(do.dtype), do, trans_a=True)` → `tl.dot(tl.trans(p.to(do.dtype)), do)`
4. Line 501: `tl.dot(do, v, trans_b=True)` → `tl.dot(do, tl.trans(v))`
5. Line 512: `tl.dot(ds, q, trans_a=True)` → `tl.dot(tl.trans(ds), q)`

**Duration**: ~2-3 minutes

**Note**: This script is automatically called by `setup_environment.sh`

---

### 3. download_data.sh

**Purpose**: Interactive assistant for downloading required datasets

**What it does**:
- Guides download of InpactorDB2 (~5.2 GB)
- Downloads FlyBase D. melanogaster genomes (~44 MB)
- Verifies downloaded files
- Checks disk space before downloading

**Usage**:
```bash
# Interactive mode (recommended)
./scripts/download_data.sh

# Download all datasets (non-interactive)
./scripts/download_data.sh --all

# Download only InpactorDB2
./scripts/download_data.sh --inpactordb2

# Download only FlyBase genomes
./scripts/download_data.sh --flybase

# Skip file verification
./scripts/download_data.sh --all --skip-verify

# Show help
./scripts/download_data.sh --help
```

**Datasets**:

1. **InpactorDB2** (Phase 1 - Required)
   - Size: ~5.2 GB
   - Source: https://github.com/simonorozcoarias/Inpactor2
   - Purpose: TE database for synthetic genome generation
   - Location: `phase1/data/inpactordb2.fasta`
   - Note: Requires manual download from GitHub repository

2. **FlyBase Genomes** (Phase 2 - Optional)
   - Size: ~44 MB (compressed)
   - Files:
     - `dmel-all-chromosome-r6.66.fasta.gz` (41 MB)
     - `dmel-all-transposon-r6.66.fasta.gz` (2.5 MB)
   - Source: FlyBase FTP (Release r6.66)
   - Purpose: Real data evaluation
   - Location: `phase2/data/`
   - Note: Already included in repository

**Duration**: Depends on download speeds (InpactorDB2: ~5-10 minutes with good connection)

---

## Quick Start Workflow

### New Installation (from scratch)

```bash
# Step 1: Setup environment (includes DNABERT-2 fixes)
./scripts/setup_environment.sh

# Step 2: Download data
./scripts/download_data.sh --all

# Step 3: Activate environment
conda activate dnabert2_te

# Step 4: Verify installation
python phase2/scripts/testing/test_dnabert2_installation.py
```

### Existing Installation (apply fixes only)

```bash
# If you encounter Triton errors
./scripts/apply_dnabert2_fixes.sh

# Verify fixes
./scripts/apply_dnabert2_fixes.sh --verify-only
```

### Re-download Data

```bash
# Interactive mode (asks which datasets)
./scripts/download_data.sh

# Or specific dataset
./scripts/download_data.sh --inpactordb2
```

---

## Troubleshooting

### setup_environment.sh

**Issue**: "CUDA not available in PyTorch"
- **Cause**: Wrong PyTorch version for your CUDA
- **Solution**: Specify CUDA version manually
  ```bash
  ./scripts/setup_environment.sh --cuda-version 11.8
  ```

**Issue**: "Conda not found"
- **Cause**: Conda not installed or not in PATH
- **Solution**: Install Miniconda from https://docs.conda.io/en/latest/miniconda.html

**Issue**: "Insufficient disk space"
- **Cause**: Less than 20GB available
- **Solution**: Free up disk space or use different location

### apply_dnabert2_fixes.sh

**Issue**: "No FlashAttention files found"
- **Cause**: DNABERT-2 model not downloaded
- **Solution**: Run without `--skip-download`
  ```bash
  ./scripts/apply_dnabert2_fixes.sh
  ```

**Issue**: "File not writable"
- **Cause**: Permission issues with HuggingFace cache
- **Solution**: Check file permissions or run with appropriate permissions

**Issue**: "Fix verification failed"
- **Cause**: sed replacements didn't work as expected
- **Solution**: Check backup files and try manual fix (see docs/DNABERT2_COMPATIBILITY.md)

### download_data.sh

**Issue**: "Download failed" for FlyBase
- **Cause**: FTP server down or network issues
- **Solution**: Try again later or download manually from FlyBase website

**Issue**: InpactorDB2 file not found
- **Cause**: Manual download not completed
- **Solution**: Follow on-screen instructions to download from GitHub

**Issue**: "Neither wget nor curl found"
- **Cause**: Download tools not installed
- **Solution**: Install wget or curl
  ```bash
  # Ubuntu/Debian
  sudo apt-get install wget

  # RHEL/CentOS
  sudo yum install wget
  ```

---

## Advanced Usage

### Custom Conda Environment

```bash
# Create environment with specific name and Python version
./scripts/setup_environment.sh \
    --env-name dnabert2_production \
    --python-version 3.9 \
    --cuda-version 12.1
```

### Verify Installation Without Changes

```bash
# Check if DNABERT-2 fixes are needed
./scripts/apply_dnabert2_fixes.sh --verify-only
```

### Batch Mode (CI/CD)

```bash
# Non-interactive setup for automation
./scripts/setup_environment.sh --cuda-version 11.8
./scripts/download_data.sh --all --skip-verify
```

---

## Integration with SLURM

These scripts can be used on HPC systems before submitting SLURM jobs:

```bash
# On login node
./scripts/setup_environment.sh --cuda-version 11.8
./scripts/download_data.sh --inpactordb2

# Then submit jobs
cd phase2/slurm
cp config.env.template config.env
# Edit config.env with your paths
sbatch production/submit_training_production.sh
```

---

## Script Maintenance

### Updating Scripts

If you need to modify these scripts:

1. **Test changes locally** before committing
2. **Update help text** (`--help` option)
3. **Update this README** with new options
4. **Verify error handling** works correctly

### Adding New Scripts

When adding new automation scripts:

1. Place in `scripts/` directory
2. Make executable: `chmod +x scripts/new_script.sh`
3. Add header with description and usage
4. Document in this README
5. Add colored output for user feedback
6. Include `--help` option

---

## Related Documentation

- **Installation Guide**: `../INSTALL.md`
- **DNABERT-2 Compatibility**: `../docs/DNABERT2_COMPATIBILITY.md`
- **Data Sources**: `../docs/DATA_SOURCES.md`
- **Reproducibility**: `../docs/REPRODUCIBILITY.md`
- **SLURM Guide**: `../phase2/slurm/README.md`

---

## Support

For issues with these scripts:

1. Check troubleshooting section above
2. Review related documentation
3. Check script output for specific error messages
4. See `INSTALL.md` for detailed installation steps

For DNABERT-2 specific issues:
- `docs/DNABERT2_COMPATIBILITY.md`
- Original repository: https://github.com/MAGICS-LAB/DNABERT_2

---

**Last Updated**: January 2026
