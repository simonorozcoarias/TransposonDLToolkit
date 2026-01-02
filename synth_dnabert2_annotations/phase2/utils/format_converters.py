#!/usr/bin/env python3
"""
Format conversion utilities for genomic annotations.

Handles reading and writing GFF3 and BED formats for TE predictions.
"""

import gzip
from pathlib import Path
from typing import Dict, List
from collections import defaultdict
from datetime import datetime
import numpy as np

from utils.postprocessing import TERegion, coordinate_converter


def write_gff3(
    regions: List[TERegion],
    output_path: Path,
    include_header: bool = True,
    source_name: str = "DNABERT2-TE-Detector",
    version: str = "1.0"
):
    """
    Write TE regions to GFF3 format.

    GFF3 Format (tab-separated, 9 columns):
    1. seqid: Sequence ID (chromosome)
    2. source: Software name
    3. type: Feature type
    4. start: Start position (1-based, inclusive)
    5. end: End position (1-based, inclusive)
    6. score: Confidence score (scaled 0-1000)
    7. strand: Strand (+, -, or .)
    8. phase: Reading frame (. for not applicable)
    9. attributes: Semicolon-separated key=value pairs

    Args:
        regions: List of TERegion objects
        output_path: Output file path
        include_header: Whether to include GFF3 header
        source_name: Source name for column 2
        version: Version string for header
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        # Write header
        if include_header:
            f.write("##gff-version 3\n")
            f.write(f"##source {source_name} v{version}\n")
            f.write(f"##date {datetime.now().strftime('%Y-%m-%d')}\n")
            f.write("##Note: Predicted transposable elements from DNABERT-2 model\n")

        # Write regions
        for idx, region in enumerate(regions, start=1):
            # Convert coordinates: 0-based (Python) to 1-based (GFF3)
            gff_start = coordinate_converter(region.start, 'zero', 'one')
            # End is already correct (Python exclusive = GFF3 inclusive)
            gff_end = region.end

            # Scale confidence to 0-1000
            score = int(region.score * 1000)

            # Calculate length
            length = region.end - region.start

            # Build attributes
            attributes = (
                f"ID=TE_{idx:06d};"
                f"Length={length};"
                f"Confidence={region.score:.4f}"
            )

            # Write line
            line = "\t".join([
                region.seqid,
                region.source,
                region.feature_type,
                str(gff_start),
                str(gff_end),
                str(score),
                region.strand,
                ".",  # phase
                attributes
            ])
            f.write(line + "\n")

    print(f"✓ Wrote {len(regions)} regions to {output_path}")


def write_bed(
    regions: List[TERegion],
    output_path: Path,
    name_prefix: str = "TE",
    use_colors: bool = True
):
    """
    Write TE regions to BED format.

    BED Format (tab-separated):
    Basic (6 columns):
    1. chrom: Chromosome name
    2. chromStart: Start position (0-based, inclusive)
    3. chromEnd: End position (0-based, exclusive)
    4. name: Feature name
    5. score: Score (0-1000)
    6. strand: Strand (+, -, or .)

    Extended with colors (9 columns):
    7. thickStart: Same as chromStart
    8. thickEnd: Same as chromEnd
    9. itemRgb: RGB color based on confidence

    Color scheme:
    - High confidence (>0.8): dark green (0,100,0)
    - Medium confidence (0.5-0.8): light green (0,200,0)
    - Low confidence (<0.5): yellow (200,200,0)

    Args:
        regions: List of TERegion objects
        output_path: Output file path
        name_prefix: Prefix for feature names
        use_colors: Whether to use 9-column format with colors
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        # Optional: write track header for UCSC browser
        if use_colors:
            f.write(f'track name="TE_predictions" description="DNABERT-2 TE predictions" '
                   f'itemRgb="On"\n')

        for idx, region in enumerate(regions, start=1):
            # BED uses 0-based coordinates (same as Python)
            chrom_start = region.start
            chrom_end = region.end

            # Scale confidence to 0-1000
            score = int(region.score * 1000)

            # Feature name
            name = f"{name_prefix}_{idx:06d}"

            if use_colors:
                # Determine color based on confidence
                if region.score > 0.8:
                    rgb = "0,100,0"  # Dark green
                elif region.score > 0.5:
                    rgb = "0,200,0"  # Light green
                else:
                    rgb = "200,200,0"  # Yellow

                # 9-column format
                line = "\t".join([
                    region.seqid,
                    str(chrom_start),
                    str(chrom_end),
                    name,
                    str(score),
                    region.strand,
                    str(chrom_start),  # thickStart
                    str(chrom_end),    # thickEnd
                    rgb
                ])
            else:
                # 6-column format
                line = "\t".join([
                    region.seqid,
                    str(chrom_start),
                    str(chrom_end),
                    name,
                    str(score),
                    region.strand
                ])

            f.write(line + "\n")

    print(f"✓ Wrote {len(regions)} regions to {output_path}")


def parse_gff3_regions(gff3_path: Path) -> Dict[str, List[TERegion]]:
    """
    Parse ground truth GFF3 file into TERegion objects.

    Reuses the parsing logic but converts to TERegion format
    for consistent interface with predictions.

    Args:
        gff3_path: Path to GFF3 file

    Returns:
        Dictionary mapping sequence_id → List[TERegion]
    """
    gff3_path = Path(gff3_path)
    regions_by_seq = defaultdict(list)

    # Determine if file is gzipped
    open_func = gzip.open if str(gff3_path).endswith('.gz') else open
    mode = 'rt' if str(gff3_path).endswith('.gz') else 'r'

    with open_func(gff3_path, mode) as f:
        for line in f:
            # Skip comments and headers
            if line.startswith('#'):
                continue

            fields = line.strip().split('\t')
            if len(fields) < 9:
                continue

            seqid, source, feature_type, start, end, score, strand, phase, attributes = fields

            # Convert coordinates: 1-based (GFF3) to 0-based (Python)
            py_start = coordinate_converter(int(start), 'one', 'zero')
            py_end = int(end)  # End is inclusive in GFF3, exclusive in Python

            # Parse score
            if score == '.':
                region_score = 0.5  # Default confidence if not specified
            else:
                # Assume score is 0-1000, convert to 0-1
                region_score = float(score) / 1000.0
                # Clamp to 0-1
                region_score = max(0.0, min(1.0, region_score))

            # Create TERegion
            region = TERegion(
                seqid=seqid,
                start=py_start,
                end=py_end,
                score=region_score,
                strand=strand if strand in ['+', '-', '.'] else '.',
                source=source,
                feature_type=feature_type
            )

            regions_by_seq[seqid].append(region)

    print(f"✓ Parsed {sum(len(r) for r in regions_by_seq.values())} regions "
          f"from {len(regions_by_seq)} sequences")

    return dict(regions_by_seq)


def build_nucleotide_array_from_regions(
    regions: List[TERegion],
    chromosome_length: int
) -> np.ndarray:
    """
    Convert TE regions to binary nucleotide array.

    Useful for calculating global nucleotide-level IoU.

    Args:
        regions: List of TERegion objects (must be from same chromosome)
        chromosome_length: Length of chromosome in bp

    Returns:
        Binary numpy array of shape (chromosome_length,)
        Values: 0 = background, 1 = TE
    """
    array = np.zeros(chromosome_length, dtype=np.int8)

    for region in regions:
        # Mark all positions within region as TE
        start = max(0, region.start)
        end = min(chromosome_length, region.end)
        array[start:end] = 1

    return array


def write_regions_by_chromosome(
    regions: List[TERegion],
    output_dir: Path,
    format: str = 'gff3'
):
    """
    Write regions split by chromosome.

    Creates one file per chromosome.

    Args:
        regions: List of TERegion objects
        output_dir: Output directory
        format: 'gff3' or 'bed'
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group regions by chromosome
    regions_by_chr = defaultdict(list)
    for region in regions:
        regions_by_chr[region.seqid].append(region)

    # Write each chromosome
    for seqid, chr_regions in regions_by_chr.items():
        if format == 'gff3':
            output_path = output_dir / f"{seqid}.gff3"
            write_gff3(chr_regions, output_path)
        elif format == 'bed':
            output_path = output_dir / f"{seqid}.bed"
            write_bed(chr_regions, output_path)
        else:
            raise ValueError(f"Unknown format: {format}")


def sort_regions(regions: List[TERegion]) -> List[TERegion]:
    """
    Sort regions by chromosome and start position.

    Args:
        regions: List of TERegion objects

    Returns:
        Sorted list
    """
    return sorted(regions, key=lambda r: (r.seqid, r.start))


def group_regions_by_chromosome(regions: List[TERegion]) -> Dict[str, List[TERegion]]:
    """
    Group regions by chromosome.

    Args:
        regions: List of TERegion objects

    Returns:
        Dictionary mapping seqid → List[TERegion]
    """
    grouped = defaultdict(list)
    for region in regions:
        grouped[region.seqid].append(region)
    return dict(grouped)


def load_chromosome_lengths_from_fasta(fasta_path: Path) -> Dict[str, int]:
    """
    Load chromosome lengths from FASTA file.

    Args:
        fasta_path: Path to FASTA file

    Returns:
        Dictionary mapping sequence_id → length
    """
    from Bio import SeqIO

    lengths = {}
    for record in SeqIO.parse(fasta_path, "fasta"):
        lengths[record.id] = len(record.seq)

    return lengths


def calculate_genome_coverage(
    regions: List[TERegion],
    chromosome_lengths: Dict[str, int]
) -> Dict[str, float]:
    """
    Calculate genome coverage statistics.

    Args:
        regions: List of TERegion objects
        chromosome_lengths: Dictionary mapping seqid → length

    Returns:
        Dictionary with coverage statistics
    """
    # Total bases covered by TEs
    total_te_bp = sum(r.end - r.start for r in regions)

    # Total genome length
    total_genome_bp = sum(chromosome_lengths.values())

    # Per-chromosome coverage
    regions_by_chr = group_regions_by_chromosome(regions)
    per_chr_coverage = {}

    for seqid, chr_length in chromosome_lengths.items():
        chr_regions = regions_by_chr.get(seqid, [])
        chr_te_bp = sum(r.end - r.start for r in chr_regions)
        per_chr_coverage[seqid] = {
            'n_regions': len(chr_regions),
            'te_bp': chr_te_bp,
            'total_bp': chr_length,
            'coverage_pct': (chr_te_bp / chr_length * 100) if chr_length > 0 else 0
        }

    return {
        'total_te_bp': total_te_bp,
        'total_genome_bp': total_genome_bp,
        'genome_coverage_pct': (total_te_bp / total_genome_bp * 100) if total_genome_bp > 0 else 0,
        'n_regions': len(regions),
        'per_chromosome': per_chr_coverage
    }
