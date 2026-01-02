# Installation Guide

This guide provides detailed instructions for setting up the synthetic genome generation and DNABERT-2 training environment.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation Options](#installation-options)
   - [Option A: Conda (Recommended)](#option-a-conda-recommended)
   - [Option B: Virtual Environment](#option-b-virtual-environment)
3. [DNABERT-2 Compatibility Fixes](#dnabert-2-compatibility-fixes)
4. [Verification](#verification)
5. [Troubleshooting](#troubleshooting)
6. [HPC-Specific Setup](#hpc-specific-setup)

## System Requirements

### Hardware

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **CPU** | 8 cores | 16+ cores |
| **RAM** | 64 GB | 128 GB |
| **GPU** | 16 GB VRAM (RTX 3090) | 40+ GB VRAM (A100) |
| **Storage** | 100 GB free | 250 GB free |

**Notes:**
- Phase 1 (genome generation): GPU optional, CPU sufficient
- Phase 2 (training): GPU **required**

### Software

- **Operating System**: Linux (Ubuntu 20.04+ tested), other Unix-like systems
- **Python**: 3.8, 3.9, or 3.10 (3.8 tested in thesis)
- **CUDA**: 11.8 or 12.1+
- **Git**: For cloning repository

**Check CUDA version:**
```bash
nvcc --version
nvidia-smi  # Check GPU and driver
```

**Check Python version:**
```bash
python --version  # Should be 3.8+
```

## Installation Options

### Option A: Conda (Recommended)

#### 1. Install Miniconda

If you don't have conda installed:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# Follow the prompts, restart shell after installation
```

#### 2. Phase 1 Environment (Optional)

Phase 1 only requires basic Python packages:

```bash
cd phase1
conda create -n te_phase1 python=3.8 -y
conda activate te_phase1
pip install -r requirements.txt
```

#### 3. Phase 2 Environment (Required for Training)

This environment includes DNABERT-2, PyTorch, and HuggingFace libraries:

```bash
cd phase2
conda create -n dnabert2_te python=3.8 -y
conda activate dnabert2_te
```

**CRITICAL: Install PyTorch with CUDA support FIRST**

The PyTorch version must match your CUDA version:

**For CUDA 11.8:**
```bash
pip install torch==2.4.1+cu118 torchvision==0.19.1+cu118 \
  --index-url https://download.pytorch.org/whl/cu118
```

**For CUDA 12.1:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**Verify CUDA is available:**
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"
```

Expected output: `CUDA available: True`

**Install remaining dependencies:**
```bash
pip install -r requirements.txt
```

### Option B: Virtual Environment

If you prefer not to use conda:

#### Phase 1

```bash
cd phase1
python3 -m venv venv_phase1
source venv_phase1/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

#### Phase 2

```bash
cd phase2
python3 -m venv venv_phase2
source venv_phase2/bin/activate
pip install --upgrade pip

# Install PyTorch with CUDA (choose version based on your CUDA)
pip install torch==2.4.1+cu118 --index-url https://download.pytorch.org/whl/cu118

# Verify CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Install remaining dependencies
pip install -r requirements.txt
```

## DNABERT-2 Compatibility Fixes

**CRITICAL**: DNABERT-2 requires manual fixes for modern Triton/PyTorch compatibility.

### Automated Fix (Recommended)

Run the automated fix script:

```bash
cd scripts
./apply_dnabert2_fixes.sh
```

This script will:
1. Download DNABERT-2 model to HuggingFace cache
2. Locate FlashAttention files
3. Apply sed replacements for Triton API
4. Verify fixes were applied

### Manual Fix (If Script Fails)

If the automated script fails, see [docs/DNABERT2_COMPATIBILITY.md](docs/DNABERT2_COMPATIBILITY.md) for detailed manual instructions.

**Files to modify** (after model download):
```
~/.cache/huggingface/hub/models--zhihan1996--DNABERT-2-117M/snapshots/*/flash_attn_triton.py
~/.cache/huggingface/modules/transformers_modules/zhihan1996/DNABERT-2-117M/*/flash_attn_triton.py
```

**Changes required:**
- Line 191: `tl.dot(q, k, trans_b=True)` → `tl.dot(q, tl.trans(k))`
- Line 434: `tl.dot(q, k, trans_b=True)` → `tl.dot(q, tl.trans(k))`
- Line 494: `tl.dot(p.to(do.dtype), do, trans_a=True)` → `tl.dot(tl.trans(p.to(do.dtype)), do)`
- Line 501: `tl.dot(do, v, trans_b=True)` → `tl.dot(do, tl.trans(v))`
- Line 512: `tl.dot(ds, q, trans_a=True)` → `tl.dot(tl.trans(ds), q)`

## Verification

### Phase 1 Verification

```bash
cd phase1
conda activate te_phase1  # or source venv_phase1/bin/activate

# Test script help
python scripts/02_build_species_index.py --help
```

Expected: Script displays help message without errors.

### Phase 2 Verification

```bash
cd phase2
conda activate dnabert2_te  # or source venv_phase2/bin/activate

# Run installation test
python scripts/testing/test_dnabert2_installation.py
```

**Expected output:**
```
Testing DNABERT-2 model loading strategies...
✅ Successfully loaded model using strategy: dnabert2_explicit_config
✅ Model validation passed
✅ Model inference test passed!
```

### Quick Functional Test

Test the complete prediction pipeline:

```bash
cd phase2/test_data
./scripts/run_test.sh
```

Expected: Predictions generated in `output/` directory (~5 minutes).

## Troubleshooting

### Issue: "CUDA not available"

**Cause**: PyTorch not compiled with CUDA or wrong CUDA version.

**Solution**:
```bash
# Check CUDA version
nvcc --version  # Should match PyTorch CUDA version

# Uninstall and reinstall PyTorch with correct CUDA
pip uninstall torch torchvision
pip install torch==2.4.1+cu118 --index-url https://download.pytorch.org/whl/cu118

# Verify
python -c "import torch; print(torch.cuda.is_available())"
```

### Issue: "trans_b=True" or "trans_a=True" error

**Cause**: DNABERT-2 FlashAttention fixes not applied.

**Solution**:
```bash
./scripts/apply_dnabert2_fixes.sh
```

Or apply manually (see docs/DNABERT2_COMPATIBILITY.md).

### Issue: "config_class inconsistency"

**Cause**: Transformers version incompatibility.

**Solution**:
```bash
# Verify transformers version
pip show transformers  # Should be >= 4.35.0

# Reinstall if needed
pip install --upgrade transformers
```

### Issue: Out of Memory (OOM) during training

**Solutions**:
1. Reduce batch size in config:
```yaml
per_device_train_batch_size: 16  # Instead of 24
```

2. Enable gradient checkpointing:
```yaml
gradient_checkpointing: true
```

3. Reduce max sequence length (if applicable)

### Issue: SLURM job fails immediately

**Cause**: Environment variables or paths not set.

**Solution**:
1. Copy template: `cp phase2/slurm/config.env.template phase2/slurm/config.env`
2. Edit paths in `config.env`
3. Verify SLURM account/partition names

### Issue: ImportError for specific packages

**Solution**:
```bash
# Ensure all dependencies installed
pip install -r requirements.txt

# If specific package missing
pip install <package_name>
```

## HPC-Specific Setup

### SLURM Configuration

1. **Configure environment**:
```bash
cd phase2/slurm
cp config.env.template config.env
nano config.env  # Edit paths for your system
```

2. **Required environment variables**:
```bash
export PROJECT_ROOT="/path/to/synth_dnabert2_annotations"
export SLURM_ACCOUNT="your_account"
export SLURM_PARTITION="gpu"
export CONDA_ENV="dnabert2_te"
```

3. **Load required modules** (if your HPC uses modules):
```bash
module load cuda/11.8
module load anaconda3
# Or module load python/3.8
```

4. **Test SLURM submission**:
```bash
sbatch slurm/production/submit_training_production.sh
# Check job: squeue -u $USER
```

### Non-SLURM HPC Systems

If your HPC uses PBS, SGE, or LSF instead of SLURM, see [docs/SLURM_REFERENCE.md](docs/SLURM_REFERENCE.md) for adaptation instructions.

## Environment Variables

For optimal performance, set these in your `~/.bashrc` or job scripts:

```bash
# Python unbuffered output (critical for SLURM logs)
export PYTHONUNBUFFERED=1

# OpenMP threads
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK  # Or set to number of CPU cores

# Disable tokenizers parallelism (avoid warnings)
export TOKENIZERS_PARALLELISM=false

# PyTorch CUDA memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## Next Steps

After successful installation:

1. **Download data**: See `phase1/data/README.md` and `phase2/data/README.md`
2. **Generate synthetic genomes**: Follow `phase1/README.md`
3. **Train model**: Follow `phase2/README.md`
4. **Run predictions**: See quick test above

For complete reproduction workflow, see [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Support

If you encounter issues not covered here:
- Check [docs/DNABERT2_COMPATIBILITY.md](docs/DNABERT2_COMPATIBILITY.md) for DNABERT-2-specific issues
- Open an issue on GitHub with error logs and system info
- Contact: jgilbaja@uoc.edu
