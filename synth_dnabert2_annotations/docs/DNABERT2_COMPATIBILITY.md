# DNABERT-2 Compatibility Guide

This document details compatibility issues and solutions for running DNABERT-2 with modern PyTorch, Triton, and Transformers versions.

## Overview

DNABERT-2 was released with code compatible with older versions of Triton and PyTorch. Running it with modern dependencies requires several fixes. This guide documents all required modifications.

## Issues Resolved

1. ✅ `trans_b=True` and `trans_a=True` errors in FlashAttention/Triton
2. ✅ `config_class` inconsistency in AutoModel loading
3. ✅ PyTorch/CUDA version incompatibility
4. ✅ Missing dependencies and version conflicts

---

## Fix 1: FlashAttention Triton API Modernization

### Problem

DNABERT-2 uses obsolete FlashAttention code with `trans_b=True` and `trans_a=True` parameters in `tl.dot()`, which were removed in all modern Triton versions (2.2.0+).

**Typical errors:**
```
TypeError: dot() got an unexpected keyword argument 'trans_b'
TypeError: dot() got an unexpected keyword argument 'trans_a'
```

### Solution

Modify FlashAttention code to use modern Triton API.

**Files to modify** (BOTH HuggingFace cache locations):
```
~/.cache/huggingface/modules/transformers_modules/zhihan1996/DNABERT-2-117M/*/flash_attn_triton.py
~/.cache/huggingface/hub/models--zhihan1996--DNABERT-2-117M/snapshots/*/flash_attn_triton.py
```

**Required changes:**

| Line | Original Code | Modified Code |
|------|--------------|---------------|
| 191 | `qk += tl.dot(q, k, trans_b=True)` | `qk += tl.dot(q, tl.trans(k))` |
| 434 | `qk = tl.dot(q, k, trans_b=True)` | `qk = tl.dot(q, tl.trans(k))` |
| 494 | `dv += tl.dot(p.to(do.dtype), do, trans_a=True)` | `dv += tl.dot(tl.trans(p.to(do.dtype)), do)` |
| 501 | `dp = tl.dot(do, v, trans_b=True)` | `dp = tl.dot(do, tl.trans(v))` |
| 512 | `dk += tl.dot(ds, q, trans_a=True)` | `dk += tl.dot(tl.trans(ds), q)` |

### Automated Fix Script

This repository includes an automated fix script. See `scripts/apply_dnabert2_fixes.sh`.

**Manual application:**
```bash
# 1. Download DNABERT-2 model (triggers cache creation)
python -c "from transformers import AutoModel; AutoModel.from_pretrained('zhihan1996/DNABERT-2-117M', trust_remote_code=True)"

# 2. Locate FlashAttention files
file1="$HOME/.cache/huggingface/hub/models--zhihan1996--DNABERT-2-117M/snapshots/*/flash_attn_triton.py"
file2="$HOME/.cache/huggingface/modules/transformers_modules/zhihan1996/DNABERT-2-117M/*/flash_attn_triton.py"

# 3. Apply fixes to both locations
for file_path in $file1 $file2; do
    if [ -f "$file_path" ]; then
        echo "Fixing: $file_path"
        sed -i 's/tl\.dot(q, k, trans_b=True)/tl.dot(q, tl.trans(k))/g' "$file_path"
        sed -i 's/tl\.dot(do, v, trans_b=True)/tl.dot(do, tl.trans(v))/g' "$file_path"
        sed -i 's/tl\.dot(p\.to(do\.dtype), do, trans_a=True)/tl.dot(tl.trans(p.to(do.dtype)), do)/g' "$file_path"
        sed -i 's/tl\.dot(ds, q, trans_a=True)/tl.dot(tl.trans(ds), q)/g' "$file_path"
    fi
done

# 4. Verify fixes
find ~/.cache/huggingface -name "*flash_attn_triton.py" -exec grep -l "tl.trans" {} \;
```

**Expected output**: Both files should be found and modified.

---

## Fix 2: BertConfig Explicit Loading Strategy

### Problem

`AutoModel.from_pretrained()` fails with `config_class` inconsistency error due to conflicts between DNABERT-2 and transformers library configurations.

**Typical error:**
```
RuntimeError: config_class attribute that is not consistent
```

### Solution

Use explicit BertConfig loading strategy when initializing DNABERT-2.

**Recommended approach:**
```python
from transformers import AutoModel, AutoTokenizer
from transformers.models.bert.configuration_bert import BertConfig

# Load config explicitly
config = BertConfig.from_pretrained("zhihan1996/DNABERT-2-117M")
config._attn_implementation = "eager"

# Load model with explicit config
model = AutoModel.from_pretrained(
    "zhihan1996/DNABERT-2-117M",
    trust_remote_code=True,
    config=config,
    attn_implementation="eager",
    torch_dtype=torch.float32
)

tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M")
```

**Note**: This strategy is already implemented in the training scripts.

---

## Fix 3: PyTorch/CUDA Compatibility

### Problem

Mismatch between local CUDA Toolkit version and PyTorch CUDA version.

**Common scenarios:**
- System CUDA: 11.8
- PyTorch default: 12.1+ (incompatible)

**Error:**
```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

### Solution

Install PyTorch compiled for your specific CUDA version.

**For CUDA 11.8:**
```bash
pip install torch==2.4.1+cu118 torchvision==0.19.1+cu118 \
  --index-url https://download.pytorch.org/whl/cu118
```

**For CUDA 12.1:**
```bash
pip install torch torchvision \
  --index-url https://download.pytorch.org/whl/cu121
```

**Verify compatibility:**
```python
import torch
print(f"PyTorch: {torch.__version__}")        # Should show +cu118 or +cu121
print(f"CUDA available: {torch.cuda.is_available()}")  # Should be True
print(f"CUDA version: {torch.version.cuda}")  # Should match your system
```

**Check system CUDA version:**
```bash
nvcc --version  # Shows CUDA Toolkit version
nvidia-smi      # Shows driver version and supported CUDA
```

---

## Verified Working Configuration

### Version Matrix

| Component | Version | Notes |
|-----------|---------|-------|
| **Python** | 3.8, 3.9, 3.10 | 3.8 tested in thesis |
| **PyTorch** | 2.4.1+cu118 | Compiled for CUDA 11.8 |
| **Triton** | 3.0.0 | With manual fixes applied |
| **Transformers** | 4.44.2+ | Supports explicit config strategy |
| **CUDA Toolkit** | 11.8.89 | Local installation |
| **NVIDIA Driver** | 535+  | Supports CUDA 11.8+ |

### Additional Dependencies

```
einops>=0.6.0
datasets>=2.14.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
pandas>=1.5.0
seaborn>=0.12.0
```

See `phase2/requirements.txt` for complete list.

---

## Complete Installation Procedure

### Step 1: Verify System

```bash
# Check CUDA Toolkit
nvcc --version

# Check GPU and driver
nvidia-smi

# Verify Python version
python --version  # Should be 3.8+
```

### Step 2: Create Environment

**Option A: Conda (Recommended)**
```bash
conda create -n dnabert2_te python=3.8
conda activate dnabert2_te
```

**Option B: venv**
```bash
python -m venv venv_dnabert2
source venv_dnabert2/bin/activate  # Linux/Mac
# or: venv_dnabert2\Scripts\activate  # Windows
```

### Step 3: Install PyTorch with CUDA

**CRITICAL**: Install PyTorch BEFORE other dependencies.

```bash
# For CUDA 11.8
pip install torch==2.4.1+cu118 torchvision==0.19.1+cu118 \
  --index-url https://download.pytorch.org/whl/cu118

# Verify
python -c "import torch; print(torch.cuda.is_available())"  # Should print True
```

### Step 4: Install Dependencies

```bash
cd phase2
pip install -r requirements.txt
```

### Step 5: Apply DNABERT-2 Fixes

**CRITICAL**: Must be done after first model download.

```bash
# Run automated fix script (from repository root)
./scripts/apply_dnabert2_fixes.sh
```

Or manually (see Fix 1 above).

### Step 6: Verify Installation

```bash
python scripts/testing/test_dnabert2_installation.py
```

**Expected output:**
```
Testing DNABERT-2 model loading strategies...
✅ Successfully loaded model using strategy: dnabert2_explicit_config
✅ Model validation passed
✅ Model inference test passed!
```

---

## Troubleshooting

### Issue: "trans_b=True" error persists

**Cause**: FlashAttention fixes not applied to BOTH cache locations

**Solution**:
1. Locate ALL FlashAttention files:
   ```bash
   find ~/.cache/huggingface -name "*flash_attn_triton.py"
   ```
2. Verify BOTH locations exist
3. Reapply fixes to both files
4. Verify with `grep`:
   ```bash
   grep "tl.trans" ~/.cache/huggingface/*/flash_attn_triton.py
   ```

### Issue: "config_class inconsistency"

**Cause**: Outdated transformers version or improper loading

**Solution**:
```bash
# Update transformers
pip install --upgrade transformers

# Verify version
pip show transformers  # Should be >= 4.35.0

# Use explicit config loading (see Fix 2)
```

### Issue: "CUDA not available"

**Cause**: PyTorch not compiled for your CUDA version

**Solution**:
```bash
# Check CUDA version
nvcc --version

# Uninstall current PyTorch
pip uninstall torch torchvision

# Reinstall with correct CUDA version
pip install torch==2.4.1+cu118 --index-url https://download.pytorch.org/whl/cu118

# Verify
python -c "import torch; print(torch.cuda.is_available())"
```

### Issue: Segmentation fault

**Cause**: NumPy/PyTorch version incompatibility

**Solution**:
```bash
pip install 'numpy>=2.0.0,<3.0.0' --force-reinstall
```

### Issue: Out of Memory (OOM)

**Solutions**:
1. Reduce batch size in config:
   ```yaml
   per_device_train_batch_size: 16  # Instead of 24
   ```

2. Enable gradient checkpointing:
   ```yaml
   gradient_checkpointing: true
   ```

3. Reduce eval batch size:
   ```yaml
   per_device_eval_batch_size: 128
   ```

---

## Important Notes

### Fix Persistence

⚠️ **FlashAttention fixes are TEMPORARY**

The modifications to `flash_attn_triton.py` will be lost if you:
- Clear HuggingFace cache
- Re-download the model
- Install on a new system

**Solution**: Re-run `scripts/apply_dnabert2_fixes.sh` after any of the above.

### Automated Fix Script

This repository includes `scripts/apply_dnabert2_fixes.sh` which:
1. Downloads DNABERT-2 model
2. Locates FlashAttention files
3. Applies all required fixes
4. Verifies modifications

**Usage:**
```bash
./scripts/apply_dnabert2_fixes.sh
```

### Performance Considerations

**GPU Memory Requirements**:
- Minimum: 16 GB VRAM (RTX 3090, A100)
- Training batch size 24: ~11-12 GB
- Inference: ~4-6 GB

**CPU Fallback**:
If no GPU available, set:
```python
device = "cpu"
# Training will be 50-100x slower
```

---

## Alternative Solutions

### Using Flash Attention 2

If you have Flash Attention 2 installed:

```python
model = AutoModel.from_pretrained(
    "zhihan1996/DNABERT-2-117M",
    trust_remote_code=True,
    attn_implementation="flash_attention_2"  # Requires flash-attn package
)
```

**Requirements:**
```bash
pip install flash-attn --no-build-isolation
```

**Note**: May require additional CUDA dependencies.

### Using Eager Attention (Safe Fallback)

Force eager attention (no FlashAttention):

```python
model = AutoModel.from_pretrained(
    "zhihan1996/DNABERT-2-117M",
    trust_remote_code=True,
    attn_implementation="eager"
)
```

**Trade-off**: Slower but more compatible.

---

## Version Compatibility Matrix

| PyTorch | Transformers | Triton | Status | Notes |
|---------|-------------|--------|--------|-------|
| 2.4.1+cu118 | 4.44.2 | 3.0.0 | ✅ Tested | Recommended |
| 2.3.0+cu118 | 4.40.0 | 2.3.0 | ✅ Works | Older but stable |
| 2.5.0+cu121 | 4.45.0 | 3.1.0 | ⚠️ Untested | Should work with fixes |
| 2.0.0 | 4.30.0 | 2.1.0 | ❌ Fails | trans_b errors |

---

## References

- **DNABERT-2 Paper**: Zhou et al., ICLR 2024
- **DNABERT-2 GitHub**: https://github.com/MAGICS-LAB/DNABERT_2
- **HuggingFace Model**: https://huggingface.co/zhihan1996/DNABERT-2-117M
- **PyTorch CUDA Versions**: https://pytorch.org/get-started/previous-versions/
- **Triton Documentation**: https://triton-lang.org/

---

## Support

For issues specific to this repository:
- Check `INSTALL.md` for installation steps
- Run `scripts/testing/test_dnabert2_installation.py`
- Open issue with error logs and system info

For DNABERT-2 specific issues:
- Check original repository: https://github.com/MAGICS-LAB/DNABERT_2
- HuggingFace model page community tab

---

**Last Updated**: January 2026
**Status**: ✅ Fully functional with modern dependencies
