#!/usr/bin/env python3
"""
Extract a test dataset from REAL Drosophila melanogaster data for testing predictions.
Extracts first 500kb of the genome and corresponding TE annotations from FlyBase.
"""

import gzip
import re
from Bio import SeqIO
from pathlib import Path

# Paths
BASE_DIR = Path("~/")
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "data/test_prediction"

# Input files (compressed)
INPUT_FASTA_GZ = DATA_DIR / "dmel-all-chromosome-r6.66.fasta.gz"
INPUT_TRANSPOSON_GZ = DATA_DIR / "dmel-all-transposon-r6.66.fasta.gz"

# Output files
OUTPUT_FASTA = OUTPUT_DIR / "test_genome_real_dmel.fasta"
OUTPUT_GFF = OUTPUT_DIR / "test_annotations_real_dmel.gff3"

# Parameters
REGION_SIZE = 500000  # 500kb

def extract_fasta_region():
    """Extract first 500kb from the real Drosophila genome."""
    print(f"Reading genome FASTA: {INPUT_FASTA_GZ}")

    # Read first sequence from compressed file
    with gzip.open(INPUT_FASTA_GZ, 'rt') as f:
        seq_record = next(SeqIO.parse(f, "fasta"))

    print(f"  Original chromosome: {seq_record.id}, length: {len(seq_record):,} bp")

    # Extract region
    region_record = seq_record[:REGION_SIZE]
    region_record.id = seq_record.id
    region_record.description = f"Test region: 1-{REGION_SIZE}"

    # Write output
    print(f"  Extracting region: 1-{REGION_SIZE:,} bp")
    with open(OUTPUT_FASTA, 'w') as f:
        SeqIO.write(region_record, f, "fasta")

    print(f"✓ Wrote test FASTA: {OUTPUT_FASTA}")
    return seq_record.id

def parse_transposon_header(header):
    """
    Parse transposon FASTA header to extract coordinates and metadata.

    Example header:
    >FBti0019256 type=transposable_element; loc=2L:22300300..22304444; name=invader2{}555; ...

    Returns dict with: id, chrom, start, end, name, type
    """
    # Extract ID (first word)
    parts = header.split()
    te_id = parts[0].lstrip('>')

    # Extract fields using regex
    loc_match = re.search(r'loc=([^:]+):(\d+)\.\.(\d+)', header)
    name_match = re.search(r'name=([^;]+)', header)
    type_match = re.search(r'type=([^;]+)', header)

    if not loc_match:
        return None

    return {
        'id': te_id,
        'chrom': loc_match.group(1),
        'start': int(loc_match.group(2)),
        'end': int(loc_match.group(3)),
        'name': name_match.group(1).strip() if name_match else 'unknown',
        'type': type_match.group(1).strip() if type_match else 'transposable_element'
    }

def extract_te_annotations(chr_id):
    """Extract TE annotations from transposon file that fall within the test region."""
    print(f"\nReading transposons: {INPUT_TRANSPOSON_GZ}")

    te_annotations = []
    total_tes = 0
    matched_tes = 0

    # Parse transposon headers
    with gzip.open(INPUT_TRANSPOSON_GZ, 'rt') as f:
        for line in f:
            if not line.startswith('>'):
                continue

            total_tes += 1

            # Parse header
            te_info = parse_transposon_header(line)
            if not te_info:
                continue

            # Check if TE is in our chromosome and region
            if te_info['chrom'] == chr_id and te_info['start'] <= REGION_SIZE:
                # Clip end coordinate if it extends beyond region
                end = min(te_info['end'], REGION_SIZE)

                te_annotations.append({
                    'chrom': chr_id,
                    'start': te_info['start'],
                    'end': end,
                    'id': te_info['id'],
                    'name': te_info['name'],
                    'type': te_info['type']
                })
                matched_tes += 1

    print(f"  Total TEs in file: {total_tes}")
    print(f"  TEs in test region ({chr_id}:1-{REGION_SIZE:,}): {matched_tes}")

    return te_annotations

def create_gff3(chr_id, te_annotations):
    """Create GFF3 file from TE annotations."""
    print(f"\nCreating GFF3: {OUTPUT_GFF}")

    # Write GFF3 file
    with open(OUTPUT_GFF, 'w') as f:
        # Write header
        f.write("##gff-version 3\n")
        f.write(f"##sequence-region {chr_id} 1 {REGION_SIZE}\n")

        # Write annotations
        for te in sorted(te_annotations, key=lambda x: x['start']):
            # GFF3 format: seqid, source, type, start, end, score, strand, phase, attributes
            gff_line = "\t".join([
                chr_id,                          # seqid
                "FlyBase",                       # source
                "transposable_element",          # type
                str(te['start']),                # start
                str(te['end']),                  # end
                ".",                             # score
                ".",                             # strand (unknown)
                ".",                             # phase
                f"ID={te['id']};Name={te['name']}"  # attributes
            ])
            f.write(gff_line + "\n")

    print(f"✓ Wrote test GFF3: {OUTPUT_GFF}")

    # Statistics
    total_bp = sum(te['end'] - te['start'] + 1 for te in te_annotations)
    coverage = (total_bp / REGION_SIZE) * 100

    print(f"\nAnnotation statistics:")
    print(f"  Total TEs: {len(te_annotations)}")
    print(f"  Total bp covered: {total_bp:,} ({coverage:.2f}%)")
    if te_annotations:
        lengths = [te['end'] - te['start'] + 1 for te in te_annotations]
        print(f"  TE length range: {min(lengths)}-{max(lengths):,} bp")
        print(f"  Average TE length: {sum(lengths)//len(lengths):,} bp")

def main():
    print("="*80)
    print("EXTRACTING REAL DROSOPHILA MELANOGASTER TEST DATASET")
    print("="*80)
    print(f"Region size: {REGION_SIZE:,} bp ({REGION_SIZE/1000:.0f} kb)")
    print(f"Output directory: {OUTPUT_DIR}\n")

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Extract FASTA region
    chr_id = extract_fasta_region()

    # Extract TE annotations from transposon file
    te_annotations = extract_te_annotations(chr_id)

    # Create GFF3 file
    create_gff3(chr_id, te_annotations)

    print("\n" + "="*80)
    print("REAL TEST DATASET READY")
    print("="*80)
    print(f"Files created:")
    print(f"  - {OUTPUT_FASTA}")
    print(f"  - {OUTPUT_GFF}")
    print(f"\nYou can now test predict_te_regions.py with real Drosophila data.")

if __name__ == "__main__":
    main()
