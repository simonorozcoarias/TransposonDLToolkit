# Test Dataset for TE Prediction

This directory contains test datasets for validating the `predict_te_regions.py` script with both synthetic and real genomic data.

## Contents

```
test_data/
├── synthetic/                          # Synthetic test data
│   ├── test_genome.fasta              # 500kb synthetic genome
│   └── test_annotations.gff3          # Ground truth (50 TEs)
├── real/                              # Real D. melanogaster data
│   ├── test_genome_real_dmel.fasta    # 500kb from chr2L
│   └── test_annotations_real_dmel.gff3 # FlyBase annotations
├── scripts/
│   ├── run_test.sh                    # Automated test script
│   ├── extract_test_data.py           # Extract synthetic data
│   └── extract_real_dmel_data.py      # Extract real data
└── expected_outputs/
    └── README.md                      # Expected results
```

## Dataset Description

### Synthetic Test Data

Located in `synthetic/`:
- **test_genome.fasta**: First 500kb of a synthetic *Drosophila melanogaster* genome
- **test_annotations.gff3**: Ground truth TE annotations (50 TE regions)
- **Source**: TEgenomeSimulator synthetic genome output
- **Genome region**: chr_synthetic_Drosophila_melanogaster_01:1-500,000
- **TE classes**: CLASSI (LTR retrotransposons) and CLASSII (DNA transposons)

### Real Test Data

Located in `real/`:
- **test_genome_real_dmel.fasta**: First 500kb from *D. melanogaster* chr2L
- **test_annotations_real_dmel.gff3**: FlyBase TE annotations
- **Source**: FlyBase Release r6.66
- **Genome region**: chr2L:1-500,000
- **Purpose**: Validation on real genomic data

## Quick Start

### Option 1: Automated Test (Recommended)

The `run_test.sh` script provides a fully automated way to test model predictions:

```bash
cd phase2/test_data

# Configure the script (edit run_test.sh)
# - Set MODEL_PATH to your trained model
# - Set USE_REAL_DATA=true for real data, false for synthetic

# Run the test
bash scripts/run_test.sh
```

**What it does**:
- Automatically detects GPU/CPU
- Selects synthetic or real data
- Runs predictions with optimized parameters
- Generates all outputs and visualizations
- Creates timestamped output directory

**Configuration** (edit `scripts/run_test.sh`):
```bash
MODEL_PATH="results/production_40_species_1/"  # Path to trained model
USE_REAL_DATA=true                              # true=real, false=synthetic
```

**Example output**:
```
========================================
Testing predict_te_regions.py
========================================

Data type: REAL Drosophila melanogaster
Model found: results/production_40_species_1/
Test FASTA: real/test_genome_real_dmel.fasta
Test GFF3: real/test_annotations_real_dmel.gff3
Output: output/test_predictions_20260102_104523
GPU detected - using CUDA

Running prediction...
[Progress bars and metrics]

========================================
Test completed successfully!
========================================

Results are in: output/test_predictions_20260102_104523
```

### Option 2: Manual Prediction

For more control, run `predict_te_regions.py` directly (see sections below).

---

## Automated Testing with run_test.sh

The `run_test.sh` script is the recommended way to quickly validate your trained model.

### Prerequisites

1. **Trained model**: Complete Phase 2 training (or download pre-trained model)
2. **Test data**: Included in repository
3. **Environment**: Conda environment activated (`conda activate dnabert2_te`)

### Configuration

Edit `scripts/run_test.sh` to configure:

```bash
# Line 11: Path to your trained model checkpoint
MODEL_PATH="results/production_40_species_1/"  # UPDATE THIS

# Line 16: Data type selection
USE_REAL_DATA=true  # true = real D. mel data, false = synthetic data
```

**Model path options**:
- `results/production_40_species_1/` - Production training output
- `results/automodel_40_species/` - AutoModel training output
- `/path/to/custom/checkpoint/` - Custom model location

### Running the Test

```bash
# From phase2 directory
cd phase2/test_data

# Edit configuration
nano scripts/run_test.sh  # or vim, emacs, etc.

# Run test
bash scripts/run_test.sh
```

### Parameters Used by run_test.sh

The script runs `predict_te_regions.py` with these optimized parameters:

```bash
--fasta       <test_genome>         # Auto-selected based on USE_REAL_DATA
--model_path  $MODEL_PATH           # From configuration
--output_dir  output/test_predictions_TIMESTAMP
--gff3        <test_annotations>    # Ground truth for evaluation
--batch_size  64                    # Optimized for A100 GPU
--device      cuda|cpu              # Auto-detected
--window_size 2048                  # Token window (training size)
--stride      1024                  # 50% overlap for inference
--min_te_length 50                  # Minimum TE size (bp)
--merge_gap   10                    # Gap to merge adjacent regions
--iou_threshold 0.5                 # IoU for region matching
--visualize                         # Generate plots
```

### Output Structure

```
output/test_predictions_20260102_104523/
├── predictions.gff3                      # All predictions
├── predictions.bed                       # BED format
├── predictions_by_chromosome/
│   └── chr2L.gff3                       # Per-chromosome
├── summary_statistics.json               # Prediction stats
├── metrics/
│   ├── overall_metrics.json             # Precision, Recall, F1
│   ├── per_chromosome_metrics.json
│   └── iou_distribution.csv
└── visualizations/
    ├── iou_distribution.png
    ├── confusion_matrix.png
    ├── length_distribution_comparison.png
    ├── evaluation_summary.png
    └── genome_tracks/                    # Visual genome browser
        └── chr2L_0-100kb.png
```

### Interpreting Results

After running, check `metrics/overall_metrics.json`:

```json
{
  "precision": 0.85,
  "recall": 0.82,
  "f1_score": 0.83,
  "num_predictions": 45,
  "num_true_annotations": 50,
  "true_positives": 41,
  "false_positives": 4,
  "false_negatives": 9,
  "mean_iou": 0.75
}
```

**Good results** (trained on synthetic data):
- F1 ≥ 0.80 on synthetic test data
- F1 ≥ 0.65 on real *D. melanogaster* data (domain shift expected)

### Troubleshooting

**Error: Model path not found**
```bash
# Check your model path
ls -l results/production_40_species_1/
# Update MODEL_PATH in run_test.sh
```

**Error: Test data not found**
```bash
# For real data, extract first:
cd phase2
python test_data/scripts/extract_real_dmel_data.py
```

**Error: CUDA out of memory**
```bash
# Edit run_test.sh, reduce batch_size:
# Change line 88: --batch_size 32  (or 16)
```

**Slow on CPU**
- Normal: CPU inference is 50-100x slower than GPU
- For 500kb: ~5-10 minutes on CPU vs ~5-10 seconds on GPU

### Comparing Synthetic vs Real Performance

Run both data types to see domain shift:

```bash
# Test on synthetic data
# Edit run_test.sh: USE_REAL_DATA=false
bash scripts/run_test.sh

# Test on real data
# Edit run_test.sh: USE_REAL_DATA=true
bash scripts/run_test.sh

# Compare metrics in the two output directories
```

Expected performance gap:
- Synthetic F1: 0.85-0.90 (trained on similar data)
- Real F1: 0.60-0.75 (domain shift from synthetic to real)

---

## Manual Prediction Usage

For more control over parameters, run `predict_te_regions.py` directly.

### 1. Prediction Only (without evaluation)

Run predictions on the test genome without comparing to ground truth:

```bash
python predict_te_regions.py \
  --fasta data/test_prediction/test_genome.fasta \
  --model_path models/dnabert2_finetuned \
  --output_dir output/test_predictions \
  --batch_size 32 \
  --device cuda
```

### 2. Prediction with Evaluation

Run predictions and evaluate against ground truth annotations:

```bash
python predict_te_regions.py \
  --fasta data/test_prediction/test_genome.fasta \
  --model_path models/dnabert2_finetuned \
  --output_dir output/test_predictions \
  --gff3 data/test_prediction/test_annotations.gff3 \
  --batch_size 32 \
  --device cuda \
  --visualize
```

This will:
- Generate predictions in GFF3 and BED formats
- Compare predictions to ground truth
- Compute precision, recall, F1, and IoU metrics
- Generate visualization plots (with `--visualize` flag)

### 3. Quick Test (CPU mode)

For quick testing without GPU:

```bash
python predict_te_regions.py \
  --fasta data/test_prediction/test_genome.fasta \
  --model_path models/dnabert2_finetuned \
  --output_dir output/test_predictions \
  --gff3 data/test_prediction/test_annotations.gff3 \
  --batch_size 8 \
  --device cpu
```

## Expected Output

After running the prediction script, the output directory will contain:

```
output/test_predictions/
├── predictions.gff3                    # All predicted TE regions (GFF3)
├── predictions.bed                     # All predicted TE regions (BED)
├── predictions_by_chromosome/          # Per-chromosome predictions
│   ├── chr_synthetic_Drosophila_melanogaster_01.gff3
│   └── chr_synthetic_Drosophila_melanogaster_01.bed
├── summary_statistics.json             # Summary statistics
├── metrics/                            # Evaluation metrics (if --gff3 provided)
│   ├── overall_metrics.json
│   ├── per_chromosome_metrics.json
│   └── iou_distribution.csv
└── visualizations/                     # Plots (if --visualize flag used)
    ├── iou_distribution.png
    ├── confusion_matrix.png
    ├── length_distribution_comparison.png
    ├── evaluation_summary.png
    ├── genome_tracks/
    │   └── chr_synthetic_Drosophila_melanogaster_01_0-100kb.png
    └── nucleotide_comparison/
        └── chr_synthetic_Drosophila_melanogaster_01_comparison.png
```

## Parameter Tuning

You can adjust the following parameters for different behaviors:

### Window parameters
- `--window_size`: Window size in bp (default: 2048)
- `--stride`: Stride in bp (default: 1024 for 50% overlap)

### Region detection parameters
- `--min_te_length`: Minimum TE length in bp (default: 50)
- `--merge_gap`: Maximum gap to merge adjacent regions (default: 10)

### Evaluation parameters
- `--iou_threshold`: IoU threshold for region matching (default: 0.5)

### Example with custom parameters

```bash
python predict_te_regions.py \
  --fasta data/test_prediction/test_genome.fasta \
  --model_path models/dnabert2_finetuned \
  --output_dir output/test_predictions \
  --gff3 data/test_prediction/test_annotations.gff3 \
  --window_size 2048 \
  --stride 512 \
  --min_te_length 100 \
  --merge_gap 20 \
  --iou_threshold 0.3 \
  --batch_size 32 \
  --device cuda \
  --visualize
```

## Regenerating the Test Dataset

If you need to regenerate the test dataset or create a different size:

```bash
# Edit extract_test_data.py to change REGION_SIZE
python data/test_prediction/extract_test_data.py
```

## Notes

- This is a small test dataset for quick validation
- For full genome analysis, use the complete genome files in `data/pilot_outputs/`
- The synthetic genome was generated using TEgenomeSimulator
- Ground truth annotations include both complete and fragmented TE insertions