# Phase 1 Data Requirements

This directory should contain the InpactorDB2 transposable element sequence database required for synthetic genome generation.

## Required File

**InpactorDB2 Database formerly known as PanTEon Database**
- **Filename**: `inpactordb2.fasta` (or `r.1.5_all.fasta`)
- **Format**: Multi-FASTA with species annotations in headers
- **Size**: ~1.5 GB (compressed)
- **Species**: 1,844+ species across multiple kingdoms
- **Content**: TE consensus sequences annotated with classification

## Download Instructions

### Option 1: Direct Download (Recommended)

PanTEon Database is available from the official repository:

```bash
# Navigate to this directory
cd phase1/data

# Download PanTEon Database (adjust URL to actual repository)
wget https://zenodo.org/records/18039747/files/PanTEon_Database_v1.5.1.fasta?download=1

# Decompress
gunzip r.1.5_all.fasta.gz

# Rename for consistency (optional)
mv r.1.5_all.fasta inpactordb2.fasta
```

**Note**: Replace `[https://zenodo.org/records/18039747]` with the actual download URL from the PanTEon Database repository.

### Option 2: Contact Authors

If direct download is unavailable, contact the PanTEon Database authors:
- **Reference**: Orozco-Arias et al. (PanTEon Database)
- **Request**: Access to TE sequence database (r.1.5 or latest version)

### Alternative File Locations

If you have PanTEon Database stored elsewhere, you can either:

1. **Symlink to this directory**:
```bash
ln -s /path/to/your/inpactordb2.fasta phase1/data/inpactordb2.fasta
```

2. **Update script parameters** to point to your file location when running generation scripts

## File Format

PanTEon Database FASTA headers contain species and classification information:

```
>Species_name|TE_family|Classification|Additional_metadata
ATCGATCGATCG...
```

**Example**:
```
>Drosophila melanogaster|Copia|LTR/Copia|...
ATGCGATCGTAGCTAGC...
```

## Verification

After downloading, verify the file:

```bash
# Check file size (should be ~5-10 GB uncompressed)
ls -lh inpactordb2.fasta

# Count sequences
grep -c "^>" inpactordb2.fasta
# Expected: ~500,000+ sequences

# View first few headers
head -n 20 inpactordb2.fasta
```

## Usage in Pipeline

Once downloaded, this file is used by:

1. **Step 1**: `scripts/01_extract_species_from_inpactordb2.py`
   - Extracts list of available species

2. **Step 2**: `scripts/02_build_species_index.py`
   - Creates species → TE sequences mapping

3. **Step 4**: `scripts/04_generate_genome_onthefly.py`
   - Retrieves TE sequences for genome generation

## File Not Included in Repository

This file is **not included** in the repository due to:
- Large size (~5-10 GB uncompressed)
- External database maintained by PanTEon Database authors
- Licensing considerations

Users must download it separately before running the Phase 1 pipeline.

## References

- **PanTEon Database**: Orozco-Arias et al.
- **Repository**: https://zenodo.org/records/18039747
- **Documentation**: https://zenodo.org/records/18039747

## Support

If you have issues obtaining PanTEon Database:
- Check the PanTEon Database official repository
- Contact the authors directly
- Open an issue in this repository for alternative solutions
