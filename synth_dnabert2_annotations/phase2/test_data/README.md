# Test Dataset for TE Prediction

This directory contains a small test dataset for testing the `predict_te_regions.py` script.

## Dataset Description

The test dataset consists of:
- **test_genome.fasta**: First 500kb of a synthetic *Drosophila melanogaster* genome
- **test_annotations.gff3**: Ground truth TE annotations for the test region (50 TE regions)

The data was extracted from the TEgenomeSimulator synthetic genome pilot output.

## Dataset Statistics

- **Genome region**: chr_synthetic_Drosophila_melanogaster_01:1-500,000
- **Size**: 500 kb
- **Number of TE annotations**: 50
- **TE classes**: CLASSI (LTR retrotransposons) and CLASSII (DNA transposons)

## Usage

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