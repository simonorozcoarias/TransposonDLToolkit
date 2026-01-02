#!/usr/bin/env python3
"""
Post-processing utilities for DNABERT-2 TE predictions.

Converts token-level predictions to nucleotide-level and detects TE regions.
Handles both overlapping and non-overlapping windows by averaging confidence
scores from all windows covering each nucleotide.
"""

import numpy as np
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class WindowPrediction:
    """Single window prediction result."""
    window_start: int
    window_end: int
    token_predictions: np.ndarray  # Shape: (n_tokens,)
    offset_mapping: List[Tuple[int, int]]  # Token → nucleotide spans
    logits: np.ndarray  # Shape: (n_tokens, 2) for confidence


@dataclass
class TERegion:
    """Predicted TE region."""
    seqid: str
    start: int  # 0-based
    end: int    # 0-based exclusive
    score: float  # Average confidence
    strand: str  # '+', '-', or '.'
    source: str = "DNABERT2-TE-Detector"
    feature_type: str = "transposable_element"


def softmax(logits: np.ndarray) -> np.ndarray:
    """
    Compute softmax for confidence calculation.

    Args:
        logits: Raw model outputs, shape (n_classes,) or (n_samples, n_classes)

    Returns:
        Softmax probabilities
    """
    # Handle both 1D and 2D arrays
    if logits.ndim == 1:
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / np.sum(exp_logits)
    else:
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)


def reconstruct_nucleotide_predictions(
    window_predictions: List[WindowPrediction],
    chromosome_length: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert token-level predictions to nucleotide-level.

    Handles overlapping windows by averaging confidence scores from all
    windows covering each nucleotide. Final prediction is based on
    averaged TE confidence (threshold: 0.5).

    Args:
        window_predictions: List of predictions per window
        chromosome_length: Total chromosome length in bp

    Returns:
        Tuple of (binary_predictions, confidence_scores)
        - binary_predictions: np.ndarray of shape (chromosome_length,), values 0 or 1
        - confidence_scores: np.ndarray of shape (chromosome_length,), values 0-1
          (represents confidence in TE class)
    """
    # Initialize arrays for accumulation
    te_confidence_sum = np.zeros(chromosome_length, dtype=np.float32)
    coverage_count = np.zeros(chromosome_length, dtype=np.int16)

    # Process each window
    for window_pred in window_predictions:
        # Process each token
        for token_idx, (nuc_start, nuc_end) in enumerate(window_pred.offset_mapping):
            # Skip special tokens (offset = (0, 0))
            if nuc_start == nuc_end == 0:
                continue

            # Convert to absolute chromosome coordinates
            abs_start = window_pred.window_start + nuc_start
            abs_end = window_pred.window_start + nuc_end

            # Get prediction and confidence for this token
            token_label = window_pred.token_predictions[token_idx]
            token_logits = window_pred.logits[token_idx]
            token_probs = softmax(token_logits)

            # Get TE confidence (class 1)
            te_prob = token_probs[1] if len(token_probs) > 1 else (1.0 if token_label == 1 else 0.0)

            # Accumulate confidence for all nucleotides covered by this token
            for pos in range(abs_start, min(abs_end, chromosome_length)):
                te_confidence_sum[pos] += te_prob
                coverage_count[pos] += 1

    # Average confidence scores (avoid division by zero)
    confidence = np.zeros(chromosome_length, dtype=np.float32)
    mask = coverage_count > 0
    confidence[mask] = te_confidence_sum[mask] / coverage_count[mask]

    # Binary predictions based on averaged confidence (threshold: 0.5)
    predictions = (confidence > 0.5).astype(np.int8)

    return predictions, confidence


def detect_te_regions(
    nucleotide_predictions: np.ndarray,
    nucleotide_confidence: np.ndarray,
    seqid: str,
    min_length: int = 50,
    merge_gap: int = 10
) -> List[TERegion]:
    """
    Detect TE regions from binary nucleotide array with smart merging.

    Algorithm:
    1. Run-length encoding to find continuous stretches of 1s
    2. Merge nearby regions (gap <= merge_gap)
    3. Filter by minimum length
    4. Calculate average confidence per region

    Args:
        nucleotide_predictions: Binary array (0=background, 1=TE)
        nucleotide_confidence: Confidence scores (0-1)
        seqid: Sequence ID (chromosome name)
        min_length: Minimum TE length in bp (default: 50)
        merge_gap: Maximum gap to merge adjacent regions (default: 10 bp)

    Returns:
        List of TERegion objects
    """
    # Step 1: Run-length encoding to find continuous stretches
    # Find starts: where value changes from 0 to 1
    padded_pred = np.concatenate(([0], nucleotide_predictions, [0]))
    diff = np.diff(padded_pred)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    if len(starts) == 0:
        return []  # No TEs predicted

    # Create initial regions
    raw_regions = []
    for start, end in zip(starts, ends):
        # Calculate average confidence for this region
        if end > start:
            region_confidence = float(nucleotide_confidence[start:end].mean())
        else:
            region_confidence = 0.0

        raw_regions.append({
            'start': int(start),
            'end': int(end),  # Exclusive
            'confidence': region_confidence
        })

    # Step 2: Merge nearby regions
    if len(raw_regions) == 0:
        return []

    merged_regions = [raw_regions[0]]
    for current in raw_regions[1:]:
        prev = merged_regions[-1]
        gap = current['start'] - prev['end']

        if gap <= merge_gap:
            # Merge: extend previous region with weighted average confidence
            prev_len = prev['end'] - prev['start']
            curr_len = current['end'] - current['start']
            total_len = current['end'] - prev['start']

            if total_len > 0:
                merged_confidence = (
                    prev['confidence'] * prev_len +
                    current['confidence'] * curr_len
                ) / total_len
            else:
                merged_confidence = prev['confidence']

            merged_regions[-1] = {
                'start': prev['start'],
                'end': current['end'],
                'confidence': merged_confidence
            }
        else:
            merged_regions.append(current)

    # Step 3: Filter by minimum length and create TERegion objects
    te_regions = []
    for region in merged_regions:
        length = region['end'] - region['start']
        if length >= min_length:
            te_regions.append(TERegion(
                seqid=seqid,
                start=region['start'],
                end=region['end'],
                score=region['confidence'],
                strand='.',  # Unknown strand for now
                source="DNABERT2-TE-Detector",
                feature_type="transposable_element"
            ))

    return te_regions


def coordinate_converter(position: int, from_system: str, to_system: str) -> int:
    """
    Convert between 0-based (Python) and 1-based (GFF3) coordinate systems.

    GFF3 uses 1-based, fully-closed intervals [start, end]
    Python uses 0-based, half-open intervals [start, end)

    Args:
        position: Coordinate value
        from_system: 'zero' (0-based) or 'one' (1-based)
        to_system: 'zero' (0-based) or 'one' (1-based)

    Returns:
        Converted coordinate

    Examples:
        # Convert Python start to GFF3 start
        coordinate_converter(0, 'zero', 'one')  # Returns 1

        # Convert GFF3 start to Python start
        coordinate_converter(1, 'one', 'zero')  # Returns 0

        # Convert Python end (exclusive) to GFF3 end (inclusive)
        # For a region [0, 100) in Python (100 bp):
        # GFF3 would be [1, 100] - same end value!
        coordinate_converter(100, 'zero', 'one')  # Returns 100
    """
    from_system = from_system.lower()
    to_system = to_system.lower()

    if from_system not in ['zero', 'one'] or to_system not in ['zero', 'one']:
        raise ValueError("Coordinate systems must be 'zero' or 'one'")

    if from_system == to_system:
        return position

    if from_system == 'zero' and to_system == 'one':
        # 0-based to 1-based: add 1 for start positions
        # For end positions: keep same value (exclusive to inclusive)
        return position + 1
    else:  # from_system == 'one' and to_system == 'zero'
        # 1-based to 0-based: subtract 1
        return position - 1


def merge_regions_by_distance(
    regions: List[TERegion],
    max_distance: int = 10
) -> List[TERegion]:
    """
    Merge TE regions that are close together.

    Useful for post-processing to combine fragments of the same TE
    that may have been split by the model.

    Args:
        regions: List of TERegion objects (should be from same chromosome)
        max_distance: Maximum distance between regions to merge (bp)

    Returns:
        List of merged TERegion objects
    """
    if len(regions) == 0:
        return []

    # Sort by start position
    sorted_regions = sorted(regions, key=lambda r: r.start)

    merged = [sorted_regions[0]]
    for current in sorted_regions[1:]:
        prev = merged[-1]

        # Check if same chromosome
        if current.seqid != prev.seqid:
            merged.append(current)
            continue

        # Calculate distance
        distance = current.start - prev.end

        if distance <= max_distance:
            # Merge: create new region spanning both
            length_prev = prev.end - prev.start
            length_curr = current.end - current.start
            total_length = current.end - prev.start

            # Weighted average confidence
            merged_score = (
                prev.score * length_prev +
                current.score * length_curr
            ) / total_length

            merged[-1] = TERegion(
                seqid=prev.seqid,
                start=prev.start,
                end=current.end,
                score=merged_score,
                strand=prev.strand,  # Keep first strand
                source=prev.source,
                feature_type=prev.feature_type
            )
        else:
            merged.append(current)

    return merged


def filter_regions_by_length(
    regions: List[TERegion],
    min_length: int = 50,
    max_length: int = None
) -> List[TERegion]:
    """
    Filter TE regions by length.

    Args:
        regions: List of TERegion objects
        min_length: Minimum length in bp (default: 50)
        max_length: Maximum length in bp (default: None = no limit)

    Returns:
        Filtered list of TERegion objects
    """
    filtered = []
    for region in regions:
        length = region.end - region.start
        if length >= min_length:
            if max_length is None or length <= max_length:
                filtered.append(region)
    return filtered


def filter_regions_by_confidence(
    regions: List[TERegion],
    min_confidence: float = 0.0
) -> List[TERegion]:
    """
    Filter TE regions by confidence score.

    Args:
        regions: List of TERegion objects
        min_confidence: Minimum confidence threshold (0-1)

    Returns:
        Filtered list of TERegion objects
    """
    return [r for r in regions if r.score >= min_confidence]


def get_region_stats(regions: List[TERegion]) -> dict:
    """
    Calculate statistics for a list of TE regions.

    Args:
        regions: List of TERegion objects

    Returns:
        Dictionary with statistics
    """
    if len(regions) == 0:
        return {
            'n_regions': 0,
            'total_bp': 0,
            'mean_length': 0,
            'median_length': 0,
            'min_length': 0,
            'max_length': 0,
            'mean_confidence': 0,
            'median_confidence': 0
        }

    lengths = [r.end - r.start for r in regions]
    confidences = [r.score for r in regions]

    return {
        'n_regions': len(regions),
        'total_bp': sum(lengths),
        'mean_length': float(np.mean(lengths)),
        'median_length': float(np.median(lengths)),
        'min_length': min(lengths),
        'max_length': max(lengths),
        'mean_confidence': float(np.mean(confidences)),
        'median_confidence': float(np.median(confidences))
    }
