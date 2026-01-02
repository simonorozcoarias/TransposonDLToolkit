# SLURM Job Scripts

This directory contains SLURM batch scripts for running training, evaluation, and data preparation on HPC clusters.

## Directory Structure

```
slurm/
├── config.env.template         # Environment configuration template
├── config.env                  # Your local configuration (gitignored)
├── production/                 # Production training/evaluation
│   ├── submit_training_production.sh
│   └── submit_evaluation_production.sh
├── automodel/                  # AutoModel experiments
│   ├── submit_training_automodel.sh
│   ├── submit_evaluation_automodel.sh
│   ├── submit_training_debug_automodel.sh
│   └── submit_evaluation_debug_automodel.sh
├── debug/                      # Debug scripts
│   ├── submit_training_debug.sh
│   ├── submit_evaluation_debug.sh
│   ├── submit_training.sh
│   └── submit_evaluation.sh
├── job_prepare_dnabert2_data.sh
├── job_combine_only.sh
└── job_test_dnabert2.sh
```

## Setup Instructions

### 1. Configure Environment

Before running any SLURM scripts, you must configure your environment:

```bash
# Copy template to create your config
cp config.env.template config.env

# Edit with your paths
nano config.env
```

**Required modifications in config.env**:
- `PROJECT_ROOT`: Path to repository root
- `INPACTORDB_PATH`: Path to InpactorDB2 database
- `SLURM_ACCOUNT`: Your SLURM account name
- `SLURM_PARTITION`: Your GPU partition name
- `SLURM_MAIL_USER`: Your email for notifications
- `CONDA_ENV`: Your conda environment name

### 2. Update SBATCH Directives

Each script contains SBATCH directives at the top. Update these placeholders:
- `REPLACE_WITH_YOUR_EMAIL` → your email
- `REPLACE_WITH_YOUR_PARTITION` → your partition name
- `REPLACE_WITH_YOUR_ACCOUNT` → your account name

**Example**:
```bash
#SBATCH --mail-user=user@domain.com
#SBATCH --partition=gpu
#SBATCH --account=myaccount
```

### 3. Test Configuration

Verify your setup before running long jobs:

```bash
# Test DNABERT-2 installation
sbatch job_test_dnabert2.sh

# Check logs
tail -f logs/slurm_*.out
```

## Script Usage

### Production Training (40 Species)

Train production model with taxonomically balanced dataset:

```bash
sbatch production/submit_training_production.sh
```

**What it does**:
1. Selects 40 species (15 animals, 10 plants, 10 fungi, 5 other)
2. Combines datasets with balanced sampling
3. Trains DNABERT-2 custom model
4. Saves checkpoints every 500 steps

**Runtime**: ~3-5 days
**Resources**: 1 GPU (20GB), 64GB RAM, 8 CPUs

### Production Evaluation

Evaluate trained model on test set:

```bash
sbatch production/submit_evaluation_production.sh \
  --model-path results/production_run/checkpoint-30000 \
  --dataset-path datasets_combined/40_species/test
```

**Runtime**: ~1-4 hours
**Resources**: 1 GPU, 64GB RAM

### AutoModel Training

Train with HuggingFace AutoModel (more stable):

```bash
sbatch automodel/submit_training_automodel.sh
```

**Differences from custom model**:
- Uses `BertForTokenClassification` directly
- Simpler configuration
- More stable convergence
- **Recommended for most users**

### Data Preparation

Prepare tokenized datasets from synthetic genomes:

```bash
# Prepare all species in batch
sbatch job_prepare_dnabert2_data.sh

# Or prepare single species
python scripts/data_preparation/prepare_dnabert2_data.py \
  --fasta ../phase1/synthetic_genomes/species/genome.fasta \
  --gff3 ../phase1/synthetic_genomes/species/annotations.gff3 \
  --output_dir datasets/species
```

### Debug Runs

Quick tests with smaller datasets:

```bash
# Debug training (faster, fewer epochs)
sbatch debug/submit_training_debug.sh

# Debug evaluation
sbatch debug/submit_evaluation_debug.sh
```

## Monitoring Jobs

### Check Job Status

```bash
# View your jobs
squeue -u $USER

# View specific job
squeue -j JOBID

# Cancel job
scancel JOBID
```

### View Logs

Logs are written to `logs/` directory:

```bash
# SLURM output/error
tail -f logs/slurm_production_JOBID.out
tail -f logs/slurm_production_JOBID.err

# Training logs
tail -f logs/training_production_JOBID.log

# Species selection logs
cat logs/species_selection_JOBID.log
```

### TensorBoard

Monitor training in real-time:

```bash
# From login node (port forward to local machine)
tensorboard --logdir results/production_run/logs

# Or use SLURM interactive session
srun --pty --gres=gpu:1 bash
tensorboard --logdir results/production_run/logs --port 6006
```

## Resource Requirements

### Recommended Allocations

| Task | GPUs | Memory | CPUs | Time |
|------|------|--------|------|------|
| Production training | 1 × 20GB | 64GB | 8 | 3 days |
| Evaluation | 1 × 20GB | 64GB | 8 | 4 hours |
| Data preparation (per species) | 0 | 32GB | 4 | 2 hours |
| Debug training | 1 × 20GB | 32GB | 4 | 4 hours |

### Adjusting Resources

If you encounter OOM errors, reduce batch size in config:

```yaml
# config/production.yaml
training:
  per_device_train_batch_size: 16  # Instead of 24
  gradient_checkpointing: true     # Enable if still OOM
```

## Common Issues

### Issue: "config.env not found"

**Solution**: Create config.env from template:
```bash
cp slurm/config.env.template slurm/config.env
nano slurm/config.env
```

### Issue: "Cannot activate conda environment"

**Solution**: Check CONDA_ENV name in config.env:
```bash
conda env list  # List available environments
# Update CONDA_ENV in config.env
```

### Issue: "Dataset path not found"

**Solution**: Verify paths in config.env:
```bash
# Check variables
source slurm/config.env
echo $DATASETS_DIR
ls -l $DATASETS_DIR
```

### Issue: Job fails immediately

**Solutions**:
1. Check SBATCH directives (account, partition)
2. Verify paths in config.env exist
3. Check SLURM error log: `cat logs/slurm_*.err`

### Issue: CUDA out of memory

**Solutions**:
1. Reduce `per_device_train_batch_size` in config
2. Enable `gradient_checkpointing: true`
3. Reduce `per_device_eval_batch_size`

## Adapting to Non-SLURM Systems

### PBS/Torque

Replace SBATCH directives with PBS equivalents:

```bash
#PBS -N job_name
#PBS -l walltime=72:00:00
#PBS -l nodes=1:ppn=8:gpus=1
#PBS -l mem=64gb
#PBS -q gpu
```

### SGE

```bash
#$ -N job_name
#$ -l h_rt=72:00:00
#$ -pe smp 8
#$ -l gpu=1
#$ -l m_mem_free=64G
```

### Local Execution

Run scripts directly without SLURM:

```bash
# Source config
source slurm/config.env

# Activate environment
conda activate ${CONDA_ENV}

# Run training directly
python scripts/training/train_token_classification_automodel.py \
  --config config/production.yaml
```

## Script Modification Guide

All scripts follow this pattern:

```bash
#!/bin/bash
#SBATCH directives...

# 1. Load config.env
source slurm/config.env

# 2. Activate environment
conda activate ${CONDA_ENV}

# 3. Change to phase2 directory
cd ${PHASE2_DIR}

# 4. Run script with environment variables
python scripts/training/train_*.py \
  --config config/production.yaml
```

To add new scripts:
1. Copy template from existing script
2. Update SBATCH directives
3. Modify python command and arguments
4. Test with debug configuration first

## Best Practices

1. **Always test first**: Use debug scripts before production runs
2. **Monitor early**: Check logs after first few minutes
3. **Save checkpoints**: Enable frequent checkpointing for long runs
4. **Use config.env**: Never hardcode paths in scripts
5. **Version control**: Commit config changes before experiments
6. **Document experiments**: Add comments to config files with experiment notes

## References

- SLURM documentation: https://slurm.schedmd.com/
- HPC best practices: See `docs/SLURM_REFERENCE.md`
- Training configuration: `config/README.md`

## Support

For SLURM-specific issues:
- Check your HPC cluster documentation
- Contact your system administrator

For script issues:
- Open issue in repository
- Check `docs/` for troubleshooting guides
