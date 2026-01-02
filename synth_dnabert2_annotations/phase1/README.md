# Phase 1: Synthetic Genome Generation

This phase generates synthetic genomes with embedded transposable elements (TEs) using TEgenomeSimulator. The pipeline processes TE sequences from InpactorDB2 and creates chromosomes with controlled TE insertions for training deep learning models.

## Overview

**Input**: InpactorDB2 TE sequence database, formerly known as PanTEon Database
**Output**: Synthetic genomes (FASTA) + TE annotations (GFF3)
**Tool**: TEgenomeSimulator v1.0.0
**Dataset generated**: 1,012 species, 67.67 GB, 17.24M TE insertions

## Pipeline Workflow

### Step 1: Extract Species Information

Extract metadata about available species in PanTEon Database:

```bash
python scripts/01_extract_species_from_inpactordb2.py \
  --fasta data/inpactordb2.fasta \
  --output data/species_list.csv
```

**Output**: CSV file with species names and sequence counts.

### Step 2: Build Species Index

Create a JSON index mapping species to their TE sequences:

**HPC/SLURM:**
```bash
sbatch slurm/job_build_species_index.sh
```

**Local execution:**
```bash
python scripts/02_build_species_index.py \
  --fasta data/inpactordb2.fasta \
  --output data/species_index.json
```

**Executes**: `scripts/02_build_species_index.py`

**Output**:
- `species_index.json`: Species → TE sequences mapping
- `species_index_summary.csv`: Statistics per species

### Step 3: Retrieve GC Content

Query NCBI Datasets API to get genome-level GC content for each species:

**HPC/SLURM:**
```bash
sbatch slurm/job_get_species_gc.sh
```

**Local execution:**
```bash
python scripts/03_get_species_gc_optimized.py \
  --species_index data/species_index.json \
  --output data/species_gc_data.csv
```

**Executes**: `scripts/03_get_species_gc_optimized.py`

**Output**: `species_gc_data.csv` with GC percentages per species.

**Success rate**: ~93% (1,711 of 1,844 species)

### Step 4: Generate Synthetic Genomes

Generate chromosomes with embedded TEs using TEgenomeSimulator:

**Batch generation with job arrays (HPC/SLURM - Recommended):**
```bash
# 1. Prepare species list for all species
bash slurm/prepare_species_array.sh
# Creates species list for all available species

# 2. Submit job array for all species (avoids partition time limits)
sbatch slurm/job_array_generate_genomes.sh
```
**Executes**: `scripts/04_generate_genome_onthefly.py` (managed by SLURM array)

**Advantages**: Avoids partition time limits, native concurrency control, individual task management

**Single species (HPC/SLURM):**
```bash
sbatch slurm/job_generate_genome_onthefly.sh "Drosophila melanogaster"
```

**Alternative batch method:**
```bash
bash slurm/batch_generate_onthefly.sh
```
**Note**: May hit partition time limits for large batches. Use job arrays instead.

**Local execution (single species):**
```bash
python scripts/04_generate_genome_onthefly.py \
  --species "Drosophila melanogaster" \
  --inpactordb data/inpactordb2.fasta \
  --species_index data/species_index.json \
  --output_dir synthetic_genomes/drosophila \
  --max_copies 100
```

**Executes**: `scripts/04_generate_genome_onthefly.py`

**Parameters**:
- `--species`: Species scientific name (e.g., "Drosophila melanogaster")
- `--inpactordb`: Path to PanTEon Database FASTA file
- `--species_index`: JSON index from Step 2
- `--output_dir`: Output directory for generated genome
- `--max_copies`: Maximum copies per TE family (default: 500)

**Output per species**:
```
synthetic_genomes/species_name/
├── chr_synthetic_Species_name_01.fasta       # Chromosome sequence
├── annotations_Species_name_01.gff3          # TE annotations (ground truth)
└── generation_metadata.json                  # Generation parameters
```

**TEgenomeSimulator parameters** (calculated automatically):
- Target size: ~100 MB per chromosome
- TE coverage: ~50% (varies by species)
- Mutation rate: 85-95% identity to consensus
- Fragmentation: Realistic TE truncation

**Runtime**: 2-45 minutes per species (depends on TE count)

### Step 5: Analyze TE Coverage

Compute statistics about TE distribution in generated genomes:

```bash
python scripts/05_analyze_te_coverage.py \
  --genomes_dir synthetic_genomes/ \
  --output analysis_results.csv
```

**Output**: CSV with coverage statistics per species:
- Total genome size
- TE bases vs background bases
- TE percentage
- Number of insertions
- Length distributions

## Script Reference

### Core Scripts

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `01_extract_species_from_inpactordb2.py` | Extract species list | PanTEon Database FASTA | species_list.csv |
| `02_build_species_index.py` | Index TE sequences by species | PanTEon Database FASTA | species_index.json |
| `03_get_species_gc_optimized.py` | Query NCBI for GC content | species_index.json | species_gc_data.csv |
| `04_generate_genome_onthefly.py` | Generate synthetic chromosome | Species index + GC data | FASTA + GFF3 |
| `05_analyze_te_coverage.py` | Compute coverage statistics | Generated genomes | analysis_results.csv |
| `analisis_inpactordb2.py` | Additional PanTEon Database analysis | PanTEon Database FASTA | Various CSVs |

### SLURM Job Scripts

| Script | Executes | Purpose |
|--------|----------|---------|
| `slurm/job_build_species_index.sh` | `02_build_species_index.py` | Build index on HPC |
| `slurm/job_get_species_gc.sh` | `03_get_species_gc_optimized.py` | Query NCBI on HPC |
| `slurm/job_generate_genome_onthefly.sh` | `04_generate_genome_onthefly.py` | Generate single genome |
| `slurm/batch_generate_onthefly.sh` | `04_generate_genome_onthefly.py` (loop) | Batch generate all species |
| `slurm/job_array_generate_genomes.sh` | Array job for parallel generation | Generate multiple species in parallel |
| `slurm/prepare_species_array.sh` | Prepare job array input | Create species list for array job |

## Data Requirements

### Input Data

Download PanTEon Database TE database (see [data/README.md](data/README.md)):
- **File**: `inpactordb2.fasta` or `r.1.5_all.fasta`
- **Size**: ~[check actual size] GB
- **Format**: Multi-FASTA with species annotations in headers
- **Species**: 1,844+ species across multiple kingdoms

### Output Data

**Storage requirements**:
- Single species: ~50-150 MB (varies)
- Full dataset (1,012 species): ~67.67 GB
- Recommended: 100 GB free space for generation + analysis

## Configuration

### TEgenomeSimulator Parameters

The pipeline calculates optimal parameters per species:

```python
# Scaling factor (m) calculation
m = 50,000,000 / (N_consensos × L_media)

# Variability margin
m_max = min(m × 1.10, 500)  # 10% upper bound, max 500
m_min = m × 0.90            # 10% lower bound

# Mutation parameters
identity_range = [85, 95]   # % identity to consensus
standard_deviation = [5, 15] # Mutation SD
```

### Customization

Modify generation parameters in `scripts/04_generate_genome_onthefly.py`:
- `MIN_COPIES`, `MAX_COPIES`: Copy number range
- `MIN_IDENTITY`, `MAX_IDENTITY`: Mutation range
- `MIN_SD`, `MAX_SD`: Mutation standard deviation

## Expected Results

### Dataset Statistics

From 1,012 species generation:

| Metric | Value |
|--------|-------|
| **Total genome size** | 67.67 GB |
| **TE bases** | 17.07 GB (25.23%) |
| **Background bases** | 50.60 GB (74.77%) |
| **Total TE insertions** | 17,236,489 |
| **TE:Background ratio** | 1:2.96 |

**Per-species statistics:**
- Mean TE coverage: 24.50%
- Median TE coverage: 29.22%
- Range: 1.48% - 30.15%

### Output File Formats

**FASTA (genome sequence):**
```
>chr_synthetic_Drosophila_melanogaster_01
ATCGATCGATCG...
```

**GFF3 (annotations):**
```
##gff-version 3
chr_synthetic_Drosophila_melanogaster_01  TEgenomeSimulator  transposable_element  1000  5000  .  +  .  ID=TE_001;family=Copia;class=LTR
```

## Troubleshooting

### Issue: Species not found in PanTEon Database

**Cause**: Species name spelling or not present in database.

**Solution**: Check available species in `species_index_summary.csv`.

### Issue: NCBI API rate limit

**Cause**: Too many requests to NCBI Datasets API.

**Solution**: Script includes automatic retry with exponential backoff. Wait and re-run.

### Issue: TEgenomeSimulator slow for high-copy species

**Cause**: Species with many TE families × high copy number.

**Solution**: Parameter `m_max = 500` already limits this. Reduce if needed.

### Issue: Generated genome smaller than expected

**Cause**: Limited TE sequences for species or copy number constraints.

**Expected behavior**: Size varies by species (1.48% - 30.15% TE coverage is normal).

## Next Steps

After generating synthetic genomes, proceed to **Phase 2** for DNABERT-2 training:

1. Tokenize genomes: `phase2/scripts/data_preparation/prepare_dnabert2_data.py`
2. Combine datasets: `phase2/scripts/data_preparation/combine_datasets.py`
3. Train model: See [phase2/README.md](../phase2/README.md)

## References

- **TEgenomeSimulator**: Rodriguez & Makałowski, 2024 ([bioRxiv](https://doi.org/10.1101/2024.03.15.585130))
- **PanTEon Database**: Orozco-Arias et al.
