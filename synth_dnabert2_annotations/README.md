# Synthetic Genome Generation and DNABERT-2 for Transposable Element Detection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Implementation of DNABERT-2 for transposable element (TE) detection in genomic sequences using synthetic training data.

## Project Overview

This repository contains the complete implementation of a Master's thesis investigating deep learning approaches for automated TE detection. The project consists of two main phases:

- **Phase 1**: Synthetic genome generation with TEgenomeSimulator (1,012 species)
- **Phase 2**: DNABERT-2 fine-tuning for token-level classification and evaluation

## Repository Structure

```
synth_dnabert2_annotations/
├── phase1/               # Synthetic genome generation
│   ├── scripts/          # Data preparation and generation scripts
│   ├── slurm/            # HPC job submission scripts
│   ├── data/             # Placeholder for InpactorDB2
│   └── examples/         # Sample outputs
├── phase2/               # DNABERT-2 training and evaluation
│   ├── scripts/          # Training, evaluation, and prediction scripts
│   │   ├── data_preparation/
│   │   ├── training/
│   │   ├── evaluation/
│   │   └── testing/
│   ├── slurm/            # HPC job scripts
│   ├── config/           # Training configurations
│   ├── utils/            # Utility modules
│   ├── test_data/        # Test datasets (synthetic and real)
│   └── data/             # D. melanogaster reference genomes
├── docs/                 # Technical documentation
├── scripts/              # Setup and automation scripts
└── notebooks/            # Analysis notebooks
```

## Quick Start

### Prerequisites

**Hardware:**
- GPU with 16GB+ VRAM (for training)
- 100GB free disk space
- 64GB RAM recommended

**Software:**
- Python 3.8+
- CUDA 11.8 or 12.1+
- Git

### Installation

1. Clone the repository:
```bash
git clone https://github.com/[username]/synth_dnabert2_annotations.git
cd synth_dnabert2_annotations
```

2. Install dependencies (see [INSTALL.md](INSTALL.md) for detailed instructions):
```bash
# Phase 2 environment (includes DNABERT-2)
conda create -n dnabert2_te python=3.8
conda activate dnabert2_te

# Install PyTorch with CUDA
pip install torch==2.4.1+cu118 --index-url https://download.pytorch.org/whl/cu118

# Install remaining dependencies
pip install -r phase2/requirements.txt

# Apply DNABERT-2 compatibility fixes
./scripts/apply_dnabert2_fixes.sh
```

3. Verify installation:
```bash
python phase2/scripts/testing/test_dnabert2_installation.py
```

## Usage

### Phase 1: Synthetic Genome Generation

Complete workflow for generating synthetic genomes with embedded TEs using HPC/SLURM:

```bash
cd phase1

# 1. Extract species information from InpactorDB2
python scripts/01_extract_species_from_inpactordb2.py \
  --fasta data/inpactordb2.fasta \
  --output data/species_list.csv

# 2. Build species index from InpactorDB2
sbatch slurm/job_build_species_index.sh
# Executes: scripts/02_build_species_index.py

# 3. Get GC content for each species from NCBI
sbatch slurm/job_get_species_gc.sh
# Executes: scripts/03_get_species_gc_optimized.py

# 4. Generate synthetic genomes for all species (batch - Recommended)
# Prepare species list
bash slurm/prepare_species_array.sh
# Submit job array (avoids partition time limits)
sbatch slurm/job_array_generate_genomes.sh
# Executes: scripts/04_generate_genome_onthefly.py (managed by SLURM array)

# Alternative batch method:
bash slurm/batch_generate_onthefly.sh
# Note: May hit partition time limits. Use job arrays instead.

# Or for a single species:
sbatch slurm/job_generate_genome_onthefly.sh "Drosophila melanogaster"

# 5. Analyze TE coverage in generated genomes
python scripts/05_analyze_te_coverage.py \
  --genomes_dir synthetic_genomes/ \
  --output analysis_results.csv
```

**Alternative (without SLURM):**

If running locally without HPC access:

```bash
# 1. Extract species information
python scripts/01_extract_species_from_inpactordb2.py \
  --fasta data/inpactordb2.fasta \
  --output data/species_list.csv

# 2. Build species index
python scripts/02_build_species_index.py \
  --fasta data/inpactordb2.fasta \
  --output data/species_index.json

# 3. Get GC content
python scripts/03_get_species_gc_optimized.py \
  --species_index data/species_index.json \
  --output data/species_gc_data.csv

# 4. Generate synthetic genome (single species)
python scripts/04_generate_genome_onthefly.py \
  --species "Drosophila melanogaster" \
  --inpactordb data/inpactordb2.fasta \
  --species_index data/species_index.json \
  --output_dir synthetic_genomes/drosophila \
  --max_copies 100
```

See [phase1/README.md](phase1/README.md) for detailed workflow.

### Phase 2: DNABERT-2 Training and Evaluation

Complete workflow using HPC/SLURM:

```bash
cd phase2

# 1. Prepare tokenized datasets from synthetic genomes (all species)
sbatch slurm/job_prepare_dnabert2_data.sh
# Executes: scripts/data_preparation/prepare_dnabert2_data.py for each species

# 2. Combine datasets from multiple species
# Executes: scripts/data_preparation/combine_datasets.py
#           scripts/data_preparation/select_species_by_category.py

# 3. Train model (production - 40 species)
sbatch slurm/production/submit_training_production.sh
# Executes: scripts/training/train_token_classification.py
# Alternative with AutoModel:
sbatch slurm/automodel/submit_training_automodel.sh
# Executes: scripts/training/train_token_classification_automodel.py

# 4. Evaluate trained model
sbatch slurm/production/submit_evaluation_production.sh \
  --model_path results/best_model \
  --dataset_path datasets_combined/40_species/test
# Executes: scripts/evaluation/evaluate_model.py

# 5. Predict TEs on new genomes
# Run via prediction script (no SLURM needed for inference)
python scripts/evaluation/predict_te_regions.py \
  --fasta genome.fasta \
  --model_path models/dnabert2_finetuned \
  --output_dir predictions \
  --batch_size 32 \
  --device cuda
```

**Alternative (without SLURM):**

```bash
# 1. Prepare single species dataset
python scripts/data_preparation/prepare_dnabert2_data.py \
  --fasta ../phase1/synthetic_genomes/species/genome.fasta \
  --gff3 ../phase1/synthetic_genomes/species/annotations.gff3 \
  --output_dir datasets/species \
  --window_size 2048 \
  --stride 2048

# 2. Train model
python scripts/training/train_token_classification_automodel.py \
  --config config/production.yaml

# 3. Evaluate
python scripts/evaluation/evaluate_model_automodel.py \
  --model_path results/best_model \
  --dataset_path datasets/species/test
```

See [phase2/README.md](phase2/README.md) for complete workflow.

### Model Validation

Validate your trained model with the automated test script:

```bash
cd phase2/test_data

# Configure the test (edit scripts/run_test.sh):
# - MODEL_PATH: Path to your trained model checkpoint
# - USE_REAL_DATA: true for real D. mel data, false for synthetic

# Run validation
bash scripts/run_test.sh
```

**What it validates**:
- Model predictions on 500kb test genome (synthetic or real)
- Evaluation metrics: Precision, Recall, F1, IoU
- Region-level performance with ground truth comparison
- Visualizations of predictions vs annotations

**Expected performance**:
- **Fine-tuned model (40 species)**: F1 ~0.88-0.90 on synthetic, ~0.65-0.80 on real
- **Fine-tuned model (10 species)**: F1 ~0.75-0.85 on synthetic, ~0.55-0.70 on real
- **Base DNABERT-2 (no training)**: F1 ~0.35-0.55 on synthetic, ~0.20-0.40 on real

**Output**:
```
output/test_predictions_TIMESTAMP/
├── predictions.gff3              # Predicted TE regions
├── metrics/overall_metrics.json  # Precision, Recall, F1, IoU
└── visualizations/               # Evaluation plots and genome tracks
```

Use this to verify training effectiveness and assess domain shift from synthetic to real data.

See [phase2/test_data/README.md](phase2/test_data/README.md) for detailed validation options.

## Key Features

- **Multi-kingdom dataset**: 1,012 species (animals, plants, fungi, others)
- **Scalable pipeline**: Automated generation and tokenization
- **DNABERT-2 compatibility**: Fixes for modern PyTorch/Triton
- **HPC ready**: SLURM job templates with configurable paths
- **Comprehensive evaluation**: Token-level and region-level metrics
- **Test data included**: 500kb synthetic and real genomes

## Technical Details

### Synthetic Genome Generation (Phase 1)

- **Tool**: TEgenomeSimulator v1.0.0
- **Dataset size**: 67.67 GB (1,012 species)
- **TE insertions**: 17.24 million annotated elements
- **Coverage**: 25.23% TEs, 74.77% background (average)

### Model Training (Phase 2)

- **Base model**: DNABERT-2-117M (zhihan1996/DNABERT-2-117M)
- **Task**: Binary token classification (TE vs background)
- **Tokenization**: Byte Pair Encoding (BPE), 4,096 vocabulary
- **Architecture**: Transformer encoder + linear classification head
- **Training**: 40 species subset, ~15 hours on A100 GPU

### Evaluation Metrics

Comprehensive metrics are computed at two levels:
- **Token-level**: Precision, Recall, F1, Accuracy
- **Region-level**: IoU, Precision, Recall, F1 (with IoU threshold)

See [docs/RESULTS.md](docs/RESULTS.md) for quantitative results.

## Data Requirements

### Phase 1 Requirements

- **InpactorDB2**: TE sequence database (~[SIZE])
  - Download instructions: [phase1/data/README.md](phase1/data/README.md)

### Phase 2 Requirements

- **D. melanogaster genome** (included): `dmel-all-chromosome-r6.66.fasta.gz`
- **TE annotations** (included): `dmel-all-transposon-r6.66.fasta.gz`

Alternatively, use the small test dataset in `phase2/test_data/` for quick validation.

## Documentation

- **[INSTALL.md](INSTALL.md)**: Detailed installation instructions
- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**: Technical architecture and workflow
- **[docs/DNABERT2_COMPATIBILITY.md](docs/DNABERT2_COMPATIBILITY.md)**: Compatibility fixes
- **[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)**: Reproduction guide
- **[docs/RESULTS.md](docs/RESULTS.md)**: Quantitative results
- **[docs/DATA_SOURCES.md](docs/DATA_SOURCES.md)**: External data sources
- **[docs/SLURM_REFERENCE.md](docs/SLURM_REFERENCE.md)**: HPC adaptation guide

## System Requirements

| Component | Requirement |
|-----------|-------------|
| **Python** | 3.8, 3.9, or 3.10 |
| **CUDA** | 11.8 or 12.1+ |
| **GPU** | 16GB+ VRAM (RTX 3090, A100, or better) |
| **RAM** | 64GB minimum |
| **Storage** | 100GB free space |
| **OS** | Linux (Ubuntu 20.04+ tested) |

## Citation

If you use this code in your research, please cite:

```bibtex
@mastersthesis{gonzalez2026te_detection,
  author = {González Gilbaja, Jorge},
  title = {Detección automática de elementos transponibles mediante técnicas de Deep Learning},
  school = {Universitat Oberta de Catalunya},
  year = {2026},
  type = {Master's Thesis},
  month = {January}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **DNABERT-2**: [Zhou et al., ICLR 2024](https://github.com/MAGICS-LAB/DNABERT_2)
- **TEgenomeSimulator**: Rodriguez & Makałowski, 2024
- **InpactorDB2**: Orozco-Arias et al.
- **Supervisor**: Dr. Simón Orozco Arias

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Support

- **Issues**: [GitHub Issues](https://github.com/[username]/synth_dnabert2_annotations/issues)
- **Documentation**: [docs/](docs/)
- **Contact**: jgilbaja@uoc.edu

---

**Master's Thesis** | Universitat Oberta de Catalunya | 2026
