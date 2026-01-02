#!/usr/bin/env python3
"""
Extract a test dataset from the full synthetic genome for testing predictions.
Extracts first 500kb of the genome and corresponding GFF annotations.
"""

from Bio import SeqIO
from pathlib import Path

# Paths
BASE_DIR = Path("~/")
INPUT_DIR = BASE_DIR / "data/pilot_outputs/Drosophila_melanogaster/TEgenomeSimulator_Drosophila_melanogaster_synth_result"
OUTPUT_DIR = BASE_DIR / "data/test_prediction"

# Input files
INPUT_FASTA = INPUT_DIR / "Drosophila_melanogaster_synth_genome_sequence_out_final.fasta"
INPUT_GFF = INPUT_DIR / "Drosophila_melanogaster_synth_repeat_annotation_out_final.gff"

# Output files
OUTPUT_FASTA = OUTPUT_DIR / "test_genome.fasta"
OUTPUT_GFF = OUTPUT_DIR / "test_annotations.gff3"

# Parameters
REGION_SIZE = 500000  # 500kb

def extract_fasta_region():
    """Extract first 500kb from the FASTA file."""
    print(f"Reading FASTA: {INPUT_FASTA}")

    # Read first sequence
    seq_record = next(SeqIO.parse(INPUT_FASTA, "fasta"))
    print(f"  Original sequence: {seq_record.id}, length: {len(seq_record):,} bp")

    # Extract region
    region_record = seq_record[:REGION_SIZE]
    region_record.id = seq_record.id
    region_record.description = f"{seq_record.description} | Test region: 1-{REGION_SIZE}"

    # Write output
    print(f"  Extracting region: 1-{REGION_SIZE:,} bp")
    with open(OUTPUT_FASTA, 'w') as f:
        SeqIO.write(region_record, f, "fasta")

    print(f"✓ Wrote test FASTA: {OUTPUT_FASTA}")
    return seq_record.id

def extract_gff_annotations(chr_id):
    """Extract GFF annotations that fall within the test region."""
    print(f"\nReading GFF: {INPUT_GFF}")

    annotations_in_region = []
    total_annotations = 0

    with open(INPUT_GFF, 'r') as f:
        for line in f:
            # Keep header/comment lines
            if line.startswith('#'):
                annotations_in_region.append(line)
                continue

            total_annotations += 1

            # Parse annotation
            parts = line.strip().split('\t')
            if len(parts) < 5:
                continue

            chrom = parts[0]
            start = int(parts[3])
            end = int(parts[4])

            # Check if annotation is in our region
            if chrom == chr_id and start >= 1 and start <= REGION_SIZE:
                annotations_in_region.append(line)

    print(f"  Total annotations: {total_annotations}")
    print(f"  Annotations in test region: {len(annotations_in_region) - 1}")  # -1 for header

    # Write output
    with open(OUTPUT_GFF, 'w') as f:
        f.writelines(annotations_in_region)

    print(f"✓ Wrote test GFF3: {OUTPUT_GFF}")

def main():
    print("="*80)
    print("EXTRACTING TEST DATASET")
    print("="*80)
    print(f"Region size: {REGION_SIZE:,} bp ({REGION_SIZE/1000:.0f} kb)")
    print(f"Output directory: {OUTPUT_DIR}\n")

    # Extract FASTA region
    chr_id = extract_fasta_region()

    # Extract corresponding GFF annotations
    extract_gff_annotations(chr_id)

    print("\n" + "="*80)
    print("TEST DATASET READY")
    print("="*80)
    print(f"Files created:")
    print(f"  - {OUTPUT_FASTA}")
    print(f"  - {OUTPUT_GFF}")
    print(f"\nYou can now test predict_te_regions.py with these files.")

if __name__ == "__main__":
    main()
