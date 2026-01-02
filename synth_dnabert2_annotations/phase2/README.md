# Phase 2: DNABERT-2 Training and Evaluation

This phase fine-tunes DNABERT-2 for token-level classification of transposable elements (TEs) in genomic sequences. The pipeline converts synthetic genomes to tokenized datasets, trains models with class imbalance handling, and evaluates performance on synthetic and real genomes.

## Overview

**Input**: Synthetic genomes (FASTA) + TE annotations (GFF3) from Phase 1
**Output**: Fine-tuned DNABERT-2 model + evaluation metrics
**Base model**: DNABERT-2-117M (zhihan1996/DNABERT-2-117M)
**Task**: Binary token classification (TE vs background)

## Pipeline Workflow

### Step 1: Data Preparation

Convert FASTA + GFF3 to tokenized HuggingFace datasets:

**Batch processing (HPC/SLURM):**
```bash
sbatch slurm/job_prepare_dnabert2_data.sh
```

**Executes**: `scripts/data_preparation/prepare_dnabert2_data.py` for each species

**Single species (local):**
```bash
python scripts/data_preparation/prepare_dnabert2_data.py \
  --fasta ../phase1/synthetic_genomes/species/genome.fasta \
  --gff3 ../phase1/synthetic_genomes/species/annotations.gff3 \
  --output_dir datasets/species \
  --window_size 2048 \
  --stride 2048
```

**Parameters**:
- `--window_size`: Window size in bp (default: 2048)
- `--stride`: Stride in bp (default: 2048, non-overlapping)
- `--output_dir`: Output directory for tokenized dataset

**Process**:
1. Sliding windows over chromosomes (2048 bp, non-overlapping)
2. Tokenization with DNABERT-2 BPE tokenizer (vocabulary: 4,096)
3. Binary labels per token (0=background, 1=TE)
4. Train/val/test split (80/10/10) per species

**Output**:
```
datasets/species/
├── train/
├── val/
├── test/
└── dataset_info.json
```

### Step 2: Combine Datasets

Combine multiple species into a single training dataset:

```bash
# Select balanced species by category
python scripts/data_preparation/select_species_by_category.py \
  --coverage_csv ../phase1/all_species_coverage.csv \
  --num_animals 15 \
  --num_plants 10 \
  --num_fungi 10 \
  --num_other 5 \
  --output species_selection.json

# Combine selected species
python scripts/data_preparation/combine_datasets.py \
  --input_dirs datasets/species1 datasets/species2 ... \
  --output_dir datasets_combined/40_species \
  --coverage_csv ../phase1/all_species_coverage.csv
```

**Executes**:
- `scripts/data_preparation/select_species_by_category.py`: Category-based selection
- `scripts/data_preparation/combine_datasets.py`: Merge datasets

**Output**: Combined dataset with global train/val/test splits and shuffling.

### Step 3: Compute Class Weights (Optional)

Calculate class weights for imbalanced data:

```bash
python scripts/data_preparation/compute_class_weights_optimized.py \
  --dataset_path datasets_combined/40_species/train \
  --output class_weights.json
```

**Note**: Weights can also be calculated dynamically during training (see config).

### Step 4: Train Model

Train DNABERT-2 for TE detection:

**Production training (HPC/SLURM):**
```bash
sbatch slurm/production/submit_training_production.sh
```

**Executes**: `scripts/training/train_token_classification.py` (custom model)

**Alternative with AutoModel:**
```bash
sbatch slurm/automodel/submit_training_automodel.sh
```

**Executes**: `scripts/training/train_token_classification_automodel.py`

**Local execution:**
```bash
python scripts/training/train_token_classification_automodel.py \
  --config config/production.yaml
```

**Training configuration** (config/production.yaml):
- Learning rate: 3e-4 (frozen base) or 2e-5 (full fine-tuning)
- Epochs: 10 (frozen) or 5 (full fine-tuning)
- Batch size: 24 (per device)
- Gradient accumulation: 3 (effective batch: 72)
- Class weights: Dynamic (sklearn balanced) or static (imbalance ratio)
- Early stopping: 10 evaluations patience

**Two model approaches**:

1. **Custom model** (`train_token_classification.py`):
   - Supports Focal Loss
   - Discriminative learning rates
   - More experimental features
   - May have stability issues

2. **AutoModel** (`train_token_classification_automodel.py`):
   - Standard HuggingFace BertForTokenClassification
   - More stable convergence
   - Simpler configuration
   - **Recommended for production**

**Output**:
```
results/production_run/
├── checkpoint-30000/        # Best model
├── logs/
│   └── tensorboard/
├── training_args.bin
└── trainer_state.json
```

**Training time**: ~15 hours for 40 species on A100 GPU

### Step 5: Evaluate Model

Evaluate trained model on test set:

**HPC/SLURM:**
```bash
sbatch slurm/production/submit_evaluation_production.sh \
  --model_path results/best_model \
  --dataset_path datasets_combined/40_species/test
```

**Executes**: `scripts/evaluation/evaluate_model.py` or `evaluate_model_automodel.py`

**Local execution:**
```bash
python scripts/evaluation/evaluate_model_automodel.py \
  --model_path results/production_run/checkpoint-30000 \
  --dataset_path datasets_combined/40_species/test \
  --output_dir evaluation_results
```

**Output**:
- Overall metrics (accuracy, precision, recall, F1)
- Per-species metrics (optional with `--by_species`)
- Confusion matrix
- Results saved in JSON

### Step 6: Predict TE Regions

Run inference on new genomes and evaluate:

```bash
# Prediction with evaluation (ground truth available)
python scripts/evaluation/predict_te_regions.py \
  --fasta genome.fasta \
  --gff3 ground_truth.gff3 \
  --model_path results/best_model \
  --output_dir predictions \
  --window_size 2048 \
  --stride 1024 \
  --min_te_length 50 \
  --merge_gap 10 \
  --iou_threshold 0.5 \
  --batch_size 32 \
  --device cuda \
  --visualize

# Prediction only (no ground truth)
python scripts/evaluation/predict_te_regions.py \
  --fasta genome.fasta \
  --model_path results/best_model \
  --output_dir predictions \
  --batch_size 32 \
  --device cuda
```

**Parameters**:
- `--window_size`: Window for inference (default: 2048 bp)
- `--stride`: Stride for sliding windows (default: 1024 bp, 50% overlap)
- `--min_te_length`: Minimum TE length to report (default: 50 bp)
- `--merge_gap`: Maximum gap to merge adjacent TEs (default: 10 bp)
- `--iou_threshold`: IoU threshold for evaluation (default: 0.5)
- `--visualize`: Generate plots (requires `--gff3`)

**Output**:
```
predictions/
├── predictions.gff3                      # All predicted TEs
├── predictions.bed                       # BED format
├── predictions_by_chromosome/            # Per-chromosome predictions
├── summary_statistics.json               # Summary stats
├── metrics/                              # Evaluation metrics (if --gff3)
│   ├── overall_metrics.json
│   ├── per_chromosome_metrics.json
│   └── iou_distribution.csv
└── visualizations/                       # Plots (if --visualize)
    ├── iou_distribution.png
    ├── confusion_matrix.png
    ├── length_distribution_comparison.png
    └── genome_tracks/
```

## Script Reference

### Data Preparation

| Script | Purpose |
|--------|---------|
| `prepare_dnabert2_data.py` | Convert FASTA+GFF3 to tokenized datasets |
| `combine_datasets.py` | Merge datasets from multiple species |
| `select_species_by_category.py` | Select balanced species by taxonomy |
| `compute_class_weights_optimized.py` | Calculate class weights |

### Training

| Script | Model Type | Features |
|--------|------------|----------|
| `train_token_classification.py` | Custom | Focal Loss, discriminative LR |
| `train_token_classification_automodel.py` | AutoModel | Standard HuggingFace, stable |
| `verify_before_training.py` | Utility | Validate dataset before training |

### Evaluation

| Script | Purpose |
|--------|---------|
| `evaluate_model.py` | Evaluate custom model on test set |
| `evaluate_model_automodel.py` | Evaluate AutoModel on test set |
| `predict_te_regions.py` | Full prediction pipeline with evaluation |

### Testing

| Script | Purpose |
|--------|---------|
| `test_dnabert2_installation.py` | Verify DNABERT-2 setup |

### Utilities (utils/)

| Module | Purpose |
|--------|---------|
| `data_collator.py` | Custom data collator for pre-tokenized sequences |
| `postprocessing.py` | Merge and filter predicted TE regions |
| `region_metrics.py` | Compute region-level metrics (IoU, precision, recall) |
| `visualizations.py` | Generate evaluation plots |
| `format_converters.py` | Convert between GFF3/BED formats |

## Configuration

### Main Config File: config/production.yaml

Key sections:

```yaml
model:
  name_or_path: "zhihan1996/DNABERT-2-117M"
  num_labels: 2

data:
  dataset_path: "datasets_combined/40_species"
  max_length: 512
  is_pretokenized: true

training:
  freeze_base_model: true          # Feature extraction (faster)
  use_dynamic_class_weights: true  # Sklearn balanced weights
  learning_rate: 3.0e-4            # High LR for frozen base
  num_train_epochs: 10
  per_device_train_batch_size: 24
  gradient_accumulation_steps: 3
  eval_steps: 500

early_stopping:
  enabled: true
  patience: 10
```

See [config/production.yaml](config/production.yaml) for full configuration with experimental history.

### Templates

- `config/production.template.yaml`: Template with placeholders for paths
- `slurm/config.env.template`: Environment variables for SLURM

## Test Data

Quick validation with small test datasets:

```bash
cd test_data
./scripts/run_test.sh
```

**Datasets**:
- `synthetic/`: 500kb synthetic D. melanogaster
- `real/`: 500kb real D. melanogaster (FlyBase r6.66)

See [test_data/README.md](test_data/README.md) for details.

## Expected Results

### Training Metrics (40 species)

| Metric | Synthetic Test Set |
|--------|-------------------|
| **Accuracy** | ~0.94 |
| **Precision** | ~0.88 |
| **Recall** | ~0.92 |
| **F1** | ~0.90 |

**Training time**: ~15 hours, 41 minutes (35,000 steps, early stopping)

### Evaluation Metrics

See [docs/RESULTS.md](../docs/RESULTS.md) for complete quantitative results.

## Troubleshooting

### Issue: DNABERT-2 loading fails

**Cause**: FlashAttention compatibility issues.

**Solution**: Apply fixes (see [INSTALL.md](../INSTALL.md)):
```bash
../scripts/apply_dnabert2_fixes.sh
```

### Issue: OOM during training

**Solutions**:
1. Reduce batch size: `per_device_train_batch_size: 16`
2. Enable gradient checkpointing: `gradient_checkpointing: true`
3. Reduce eval batch size: `per_device_eval_batch_size: 128`

### Issue: Model not converging

**Check**:
1. Class weights enabled: `use_dynamic_class_weights: true`
2. Learning rate appropriate for freeze setting
3. Data shuffled correctly (no species grouping)

### Issue: SLURM job fails

**Solution**: Configure environment variables:
```bash
cp slurm/config.env.template slurm/config.env
nano slurm/config.env  # Edit paths
```

## Next Steps

1. **Evaluate on real genomes**: Test generalization
2. **Hyperparameter tuning**: Adjust config for your data
3. **Deploy model**: Use `predict_te_regions.py` for production

## References

- **DNABERT-2**: Zhou et al., ICLR 2024 ([GitHub](https://github.com/MAGICS-LAB/DNABERT_2))
- **HuggingFace Transformers**: [Documentation](https://huggingface.co/docs/transformers)
- **Compatibility fixes**: [../docs/DNABERT2_COMPATIBILITY.md](../docs/DNABERT2_COMPATIBILITY.md)
