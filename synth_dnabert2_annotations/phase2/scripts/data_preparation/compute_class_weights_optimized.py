#!/usr/bin/env python3
"""
Compute Class Weights from Global Dataset Statistics (Optimized)

This script calculates class weights WITHOUT loading the full dataset into memory.
Instead, it uses pre-computed genomic statistics from all_species_coverage.csv.

For a 170GB dataset, this approach is:
  - INSTANT (vs 2-4 hours for full dataset loading)
  - MEMORY EFFICIENT (uses only CSV, not 170GB RAM)
  - ACCURATE (based on all 1,012 species)

Usage:
    # Option 1: Use pre-calculated weights from CSV statistics
    python compute_class_weights_optimized.py --stats_csv results/all_species_coverage.csv

    # Option 2: Verify with small sample from dataset
    python compute_class_weights_optimized.py --dataset_path datasets/splits/train --sample_ratio 0.02

    # Option 3: Both (calculate from CSV and verify with sample)
    python compute_class_weights_optimized.py --stats_csv results/all_species_coverage.csv \\
        --dataset_path datasets/splits/train --sample_ratio 0.02
"""

import argparse
import csv
import json
import logging
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def compute_weights_from_csv(csv_path: str):
    """
    Compute class weights from genomic statistics CSV (FAST, NO DATASET LOADING).

    This method works at NUCLEOTIDE level and assumes:
      - Imbalance ratio at nucleotide level ≈ Imbalance ratio at token level

    This assumption is reasonable because:
      - DNABERT-2 BPE tokenization is trained on general genomic data
      - No biological reason for TEs to tokenize differently than background
      - Both are DNA sequences with similar nucleotide composition

    Args:
        csv_path: Path to all_species_coverage.csv

    Returns:
        dict: Statistics including counts, ratios, and weights
    """
    logger.info(f"Loading genomic statistics from: {csv_path}")

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        data = list(reader)

    n_species = len(data)
    logger.info(f"  Species count: {n_species:,}")

    # Sum genomic bases across all species
    total_genome_length = sum(int(row['total_genome_length']) for row in data)
    total_te_bases = sum(int(row['total_te_bases']) for row in data)
    total_background_bases = sum(int(row['total_background_bases']) for row in data)

    logger.info(f"\nGenomic coverage (nucleotide level):")
    logger.info(f"  Total genome length:    {total_genome_length:,} bp")
    logger.info(f"  TE bases:               {total_te_bases:,} bp ({total_te_bases/total_genome_length*100:.2f}%)")
    logger.info(f"  Background bases:       {total_background_bases:,} bp ({total_background_bases/total_genome_length*100:.2f}%)")

    # Calculate imbalance ratio at nucleotide level
    imbalance_ratio = total_background_bases / total_te_bases

    logger.info(f"\nImbalance ratio (nucleotide level): {imbalance_ratio:.2f}:1")
    logger.info(f"Assumption: This ratio approximates the token-level ratio")
    logger.info(f"Reasoning: BPE tokenization should not favor TEs or background")

    # Calculate class weights using balanced formula
    # weight[class] = n_samples / (n_classes × n_samples_per_class)
    # Using nucleotide counts as proxy for token counts
    n_classes = 2
    weight_background = total_genome_length / (n_classes * total_background_bases)
    weight_te = total_genome_length / (n_classes * total_te_bases)

    # Normalize weights (min = 1.0)
    min_weight = min(weight_background, weight_te)
    weight_background_norm = weight_background / min_weight
    weight_te_norm = weight_te / min_weight

    stats = {
        "source": "genomic_statistics_csv",
        "method": "nucleotide_level_proxy",
        "assumption": "imbalance_ratio_nucleotide ≈ imbalance_ratio_token",
        "n_species": n_species,
        "nucleotide_level": {
            "total_genome_length": int(total_genome_length),
            "te_bases": int(total_te_bases),
            "background_bases": int(total_background_bases),
            "te_percent": float(total_te_bases / total_genome_length * 100),
            "background_percent": float(total_background_bases / total_genome_length * 100),
        },
        "imbalance_ratio": float(imbalance_ratio),
        "class_weights": {
            "background": float(weight_background),
            "TE": float(weight_te)
        },
        "class_weights_normalized": {
            "background": float(weight_background_norm),
            "TE": float(weight_te_norm)
        }
    }

    return stats


def compute_weights_from_sample(dataset_path: str, sample_ratio: float = 0.02, ignore_index: int = -100):
    """
    Compute class weights from a random sample of the dataset (VERIFICATION).

    This is much faster than loading the full dataset but still validates
    that the CSV-based calculation is accurate.

    Args:
        dataset_path: Path to dataset directory
        sample_ratio: Fraction of dataset to sample (default: 0.02 = 2%)
        ignore_index: Label value to ignore (default: -100)

    Returns:
        dict: Statistics from sampled data
    """
    from datasets import load_from_disk

    logger.info(f"\nLoading dataset sample for verification...")
    logger.info(f"  Dataset path: {dataset_path}")
    logger.info(f"  Sample ratio: {sample_ratio*100:.1f}%")

    # Load full dataset metadata (fast, doesn't load data)
    dataset = load_from_disk(dataset_path)
    total_size = len(dataset)

    # Calculate sample size (minimum 1000 samples)
    sample_size = max(1000, int(total_size * sample_ratio))
    sample_size = min(sample_size, total_size)  # Don't exceed dataset size

    logger.info(f"  Total samples: {total_size:,}")
    logger.info(f"  Sample size: {sample_size:,} ({sample_size/total_size*100:.2f}%)")

    # Sample dataset randomly
    sampled_dataset = dataset.shuffle(seed=42).select(range(sample_size))

    # Count tokens per class
    class_counts = {0: 0, 1: 0}  # 0=background, 1=TE
    total_tokens = 0

    logger.info(f"  Counting tokens in sample...")
    for sample in tqdm(sampled_dataset, desc="Processing samples"):
        labels = np.array(sample['labels'])

        # Filter out padding/ignored tokens
        valid_labels = labels[labels != ignore_index]

        # Count each class
        for label in valid_labels:
            class_counts[int(label)] += 1
            total_tokens += 1

    # Calculate statistics
    background_count = class_counts[0]
    te_count = class_counts[1]

    background_ratio = background_count / total_tokens
    te_ratio = te_count / total_tokens

    # Calculate class weights
    n_classes = 2
    background_weight = total_tokens / (n_classes * background_count)
    te_weight = total_tokens / (n_classes * te_count)

    # Normalize weights
    min_weight = min(background_weight, te_weight)
    background_weight_normalized = background_weight / min_weight
    te_weight_normalized = te_weight / min_weight

    stats = {
        "source": "dataset_sample",
        "sample_size": sample_size,
        "sample_ratio": sample_ratio,
        "total_tokens": int(total_tokens),
        "class_counts": {
            "background": int(background_count),
            "TE": int(te_count)
        },
        "class_ratios": {
            "background": float(background_ratio),
            "TE": float(te_ratio)
        },
        "imbalance_ratio": float(background_count / te_count),
        "class_weights": {
            "background": float(background_weight),
            "TE": float(te_weight)
        },
        "class_weights_normalized": {
            "background": float(background_weight_normalized),
            "TE": float(te_weight_normalized)
        }
    }

    return stats


def print_stats(stats: dict, label: str):
    """Print statistics in a formatted way."""
    logger.info("\n" + "=" * 80)
    logger.info(f"{label}")
    logger.info("=" * 80)

    if stats["source"] == "genomic_statistics_csv":
        logger.info(f"Source: Genomic statistics CSV ({stats['n_species']} species)")
        logger.info(f"Method: {stats['method']}")
        logger.info(f"Assumption: {stats['assumption']}")
        logger.info(f"\nNucleotide level (measured):")
        logger.info(f"  Total genome:  {stats['nucleotide_level']['total_genome_length']:,} bp")
        logger.info(f"  TE bases:      {stats['nucleotide_level']['te_bases']:,} bp ({stats['nucleotide_level']['te_percent']:.2f}%)")
        logger.info(f"  Background:    {stats['nucleotide_level']['background_bases']:,} bp ({stats['nucleotide_level']['background_percent']:.2f}%)")
    else:
        logger.info(f"Source: Dataset sample ({stats['sample_size']:,} samples, {stats['sample_ratio']*100:.1f}%)")
        logger.info(f"\nToken counts (measured):")
        logger.info(f"  Total tokens:  {stats['total_tokens']:,}")
        logger.info(f"  Background:    {stats['class_counts']['background']:,} ({stats['class_ratios']['background']:.2%})")
        logger.info(f"  TE:            {stats['class_counts']['TE']:,} ({stats['class_ratios']['TE']:.2%})")

    logger.info(f"\nImbalance ratio (background/TE): {stats['imbalance_ratio']:.2f}:1")
    logger.info(f"\nClass weights (balanced):")
    logger.info(f"  Background: {stats['class_weights']['background']:.4f}")
    logger.info(f"  TE:         {stats['class_weights']['TE']:.4f}")
    logger.info(f"\nNormalized class weights (min=1.0):")
    logger.info(f"  Background: {stats['class_weights_normalized']['background']:.4f}")
    logger.info(f"  TE:         {stats['class_weights_normalized']['TE']:.4f}")
    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Compute class weights efficiently from CSV statistics or dataset sample"
    )
    parser.add_argument(
        "--stats_csv",
        type=str,
        help="Path to all_species_coverage.csv (FAST: instant calculation)"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="Path to dataset directory (for verification with sample)"
    )
    parser.add_argument(
        "--sample_ratio",
        type=float,
        default=0.02,
        help="Fraction of dataset to sample for verification (default: 0.02 = 2%%)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="class_weights.json",
        help="Output JSON file for class weights"
    )
    args = parser.parse_args()

    if not args.stats_csv and not args.dataset_path:
        parser.error("Must provide either --stats_csv or --dataset_path")

    results = {}

    # Method 1: Calculate from CSV statistics (RECOMMENDED)
    if args.stats_csv:
        csv_stats = compute_weights_from_csv(args.stats_csv)
        print_stats(csv_stats, "CLASS WEIGHTS FROM GENOMIC STATISTICS (RECOMMENDED)")
        results["csv_statistics"] = csv_stats

    # Method 2: Verify with dataset sample (OPTIONAL)
    if args.dataset_path:
        sample_stats = compute_weights_from_sample(args.dataset_path, args.sample_ratio)
        print_stats(sample_stats, "CLASS WEIGHTS FROM DATASET SAMPLE (VERIFICATION)")
        results["sample_verification"] = sample_stats

    # Compare results if both methods used
    if args.stats_csv and args.dataset_path:
        logger.info("\n" + "=" * 80)
        logger.info("COMPARISON: CSV vs Sample")
        logger.info("=" * 80)

        csv_te_weight = csv_stats['class_weights_normalized']['TE']
        sample_te_weight = sample_stats['class_weights_normalized']['TE']
        diff_percent = abs(csv_te_weight - sample_te_weight) / csv_te_weight * 100

        logger.info(f"TE weight (CSV):    {csv_te_weight:.4f}")
        logger.info(f"TE weight (Sample): {sample_te_weight:.4f}")
        logger.info(f"Difference:         {diff_percent:.2f}%")

        if diff_percent < 5:
            logger.info("✅ Results are VERY SIMILAR (<5% difference)")
            logger.info("   CSV-based calculation is validated!")
        elif diff_percent < 10:
            logger.info("⚠️  Results are SIMILAR (5-10% difference)")
            logger.info("   Consider increasing sample size for verification")
        else:
            logger.warning("❌ Results DIFFER significantly (>10%)")
            logger.warning("   This may indicate issues with tokenization assumptions")
            logger.warning("   Consider using sample-based calculation instead")

        logger.info("=" * 80)

    # Save to file
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n✅ Class weights saved to: {output_path}")

    # Print recommendation
    logger.info("\n" + "=" * 80)
    logger.info("RECOMMENDED CONFIGURATION")
    logger.info("=" * 80)

    if "csv_statistics" in results:
        te_weight = results["csv_statistics"]["class_weights_normalized"]["TE"]
    else:
        te_weight = results["sample_verification"]["class_weights_normalized"]["TE"]

    logger.info(f"\nAdd to config.yaml:")
    logger.info(f"  model:")
    logger.info(f"    num_labels: 2")
    logger.info(f"    class_weights: [1.0, {te_weight:.2f}]  # [background, TE]")
    logger.info(f"\nOr modify train_token_classification.py (line ~369):")
    logger.info(f"  class_weights_list = [1.0, {te_weight:.2f}]")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()