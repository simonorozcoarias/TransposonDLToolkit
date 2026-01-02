# Phase 2 Data Requirements

This directory contains reference genomes and annotations for *Drosophila melanogaster* used for model evaluation on real genomic data.

## Included Files

The following reference files are **included in this repository**:

1. **`dmel-all-chromosome-r6.66.fasta.gz`** (~41 MB)
   - Complete *D. melanogaster* genome assembly
   - Release: FlyBase r6.66
   - Format: FASTA (gzipped)
   - Content: All chromosomes (2L, 2R, 3L, 3R, 4, X, Y, mitochondrion)

2. **`dmel-all-transposon-r6.66.fasta.gz`** (~2.5 MB)
   - TE annotations for *D. melanogaster*
   - Release: FlyBase r6.66
   - Format: FASTA (gzipped)
   - Content: Canonical transposable element sequences

## File Purpose

These files are used for:

1. **Model Evaluation on Real Data**
   - Testing model generalization to real genomes
   - Comparing predictions with curated TE annotations
   - Validating synthetic training effectiveness

2. **Test Dataset Generation**
   - Creating small evaluation datasets in `test_data/`
   - Extracting regions for quick validation

## Usage in Pipeline

### Decompressing Files

Scripts can work with gzipped files directly, but you may decompress if needed:

```bash
# Decompress genome
gunzip -k dmel-all-chromosome-r6.66.fasta.gz
# Creates: dmel-all-chromosome-r6.66.fasta

# Decompress TE annotations
gunzip -k dmel-all-transposon-r6.66.fasta.gz
# Creates: dmel-all-transposon-r6.66.fasta
```

**Note**: The `-k` flag keeps the original `.gz` files.

### Used By Scripts

1. **`test_data/scripts/extract_real_dmel_data.py`**
   - Extracts test regions from real genome
   - Creates evaluation datasets

2. **`scripts/evaluation/predict_te_regions.py`**
   - Runs inference on real genomic sequences
   - Compares with FlyBase annotations

## File Information

### Genome Assembly (dmel-all-chromosome-r6.66.fasta.gz)

- **Source**: FlyBase
- **Version**: Release 6.66
- **Organism**: *Drosophila melanogaster*
- **Assembly**: dm6
- **Size**: ~143 MB (uncompressed)
- **Chromosomes**: 2L, 2R, 3L, 3R, 4, X, Y, mitochondrion

**FASTA format**:
```
>2L type=golden_path_region; loc=2L:1..23513712; ID=2L; dbxref=GB:AE014134,GB:AE014134,REFSEQ:NT_033779; MD5=b6a98b7c676bdaa11ec9521ed15aff2b; length=23513712; release=r6.66; species=Dmel;
CGACAATGCACGACAGAGGAAGC...
```

### TE Annotations (dmel-all-transposon-r6.66.fasta.gz)

- **Source**: FlyBase
- **Version**: Release 6.66
- **Content**: Canonical TE sequences
- **Size**: ~7 MB (uncompressed)
- **Elements**: ~100+ TE families

**FASTA format**:
```
>FBti0019430 type=transposable_element; loc=unkn:unkn..unkn; name=1360; dbxref=FlyBase_Annotation_IDs:FBti0019430; md5=e59178e0f7af6a0f1b1f6e5c4c8e6b4f; length=7439; parent=FBte0000002; derived_computed_cyto=; species=Dmel;
ATCGATCGATCG...
```

## Download Instructions (If Missing)

If these files are not present or you need a different version:

### Direct Download from FlyBase

```bash
cd phase2/data

# Download genome (release 6.66)
wget https://ftp.flybase.net/releases/FB2024_06/dmel_r6.66/fasta/dmel-all-chromosome-r6.66.fasta.gz

# Download TE annotations (release 6.66)
wget https://ftp.flybase.net/releases/FB2024_06/dmel_r6.66/fasta/dmel-all-transposon-r6.66.fasta.gz
```

### Alternative: Latest Release

For the most recent FlyBase release, visit:
- **FTP**: https://ftp.flybase.net/releases/current/
- **Website**: https://flybase.org/

Navigate to: `fasta/` directory and download:
- `dmel-all-chromosome-r6.XX.fasta.gz`
- `dmel-all-transposon-r6.XX.fasta.gz`

## Verification

Verify downloaded files:

```bash
# Check file sizes
ls -lh dmel-all-*.gz

# Genome should be ~41 MB (compressed)
# TEs should be ~2.5 MB (compressed)

# Count sequences in genome
zcat dmel-all-chromosome-r6.66.fasta.gz | grep -c "^>"
# Expected: 8 (main chromosomes)

# Count TE sequences
zcat dmel-all-transposon-r6.66.fasta.gz | grep -c "^>"
# Expected: ~100+ TE families

# View first entries
zcat dmel-all-chromosome-r6.66.fasta.gz | head -n 5
```

## Additional Data Sources

### For Training (Not Included)

Training data comes from **Phase 1** synthetic genomes:
- Generated using TEgenomeSimulator
- Located in: `../phase1/synthetic_genomes/`
- See: `../phase1/README.md` for generation instructions

### Test Data (Included)

Small test datasets are provided in `../test_data/`:
- **Synthetic**: 500kb synthetic *D. melanogaster* genome
- **Real**: 500kb real *D. melanogaster* genome (extracted from these files)

See: `../test_data/README.md` for details.

## File Versions

| File | Version | Release Date | Size (compressed) |
|------|---------|--------------|-------------------|
| dmel-all-chromosome-r6.66.fasta.gz | r6.66 | 2024 | ~41 MB |
| dmel-all-transposon-r6.66.fasta.gz | r6.66 | 2024 | ~2.5 MB |

**Note**: Version r6.66 was used in the thesis. Newer versions may be available.

## License and Attribution

- **Source**: FlyBase (https://flybase.org/)
- **License**: FlyBase data usage policy
- **Citation**: Please cite FlyBase if using these data in publications

**FlyBase Citation**:
```
Larkin A, et al. (2021). FlyBase: updates to the Drosophila melanogaster
knowledge base. Nucleic Acids Research, 49(D1), D899-D907.
```

## References

- **FlyBase**: https://flybase.org/
- **FTP Repository**: https://ftp.flybase.net/
- **Genome Release Notes**: https://flybase.org/releases/

## Support

For issues with FlyBase data:
- Visit: https://flybase.org/
- Contact: flybase@morgan.harvard.edu

For issues with this pipeline:
- Open an issue in this repository
- Contact: jgilbaja@uoc.edu
