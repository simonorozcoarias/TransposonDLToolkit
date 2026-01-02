#!/usr/bin/env python3
"""
Evaluation metrics for TE region predictions.

Computes IoU metrics at both per-element and global nucleotide levels.
"""

import numpy as np
from typing import Dict, List, Any, Tuple
from collections import defaultdict

from utils.postprocessing import TERegion


def compute_iou_per_element(
    predicted_regions: List[TERegion],
    true_regions: List[TERegion],
    iou_threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Compute per-element IoU metrics with optimal matching.

    Algorithm:
    1. Group regions by sequence ID
    2. For each true region, find best matching prediction (max IoU)
    3. If IoU >= threshold: mark as True Positive
    4. Unmatched predictions: False Positives
    5. Unmatched true regions: False Negatives

    IoU = intersection_length / union_length

    Args:
        predicted_regions: List of predicted TERegion objects
        true_regions: List of ground truth TERegion objects
        iou_threshold: Minimum IoU for a match (default: 0.5)

    Returns:
        Dictionary with:
            - 'true_positives': List of (true_region, pred_region, iou) tuples
            - 'false_positives': List of TERegion objects
            - 'false_negatives': List of TERegion objects
            - 'per_element_ious': List of IoU scores for TP matches
            - 'mean_iou': Mean IoU across TP matches
            - 'median_iou': Median IoU across TP matches
            - 'precision': TP / (TP + FP)
            - 'recall': TP / (TP + FN)
            - 'f1': F1 score
            - 'TP': Count of true positives
            - 'FP': Count of false positives
            - 'FN': Count of false negatives
    """
    # Group by sequence ID
    pred_by_seq = defaultdict(list)
    true_by_seq = defaultdict(list)

    for r in predicted_regions:
        pred_by_seq[r.seqid].append(r)
    for r in true_regions:
        true_by_seq[r.seqid].append(r)

    # Storage for results
    matches = []  # (true_region, pred_region, iou)
    unmatched_pred = []
    unmatched_true = []

    # Process each sequence
    all_seqids = set(pred_by_seq.keys()) | set(true_by_seq.keys())

    for seqid in all_seqids:
        true_list = true_by_seq.get(seqid, [])
        pred_list = pred_by_seq.get(seqid, [])

        # Track which predictions have been matched
        pred_matched = [False] * len(pred_list)

        # For each true region, find best match
        for true_region in true_list:
            best_match = None
            best_iou = 0.0
            best_idx = -1

            # Find overlapping predictions and calculate IoU
            for idx, pred_region in enumerate(pred_list):
                if pred_matched[idx]:
                    continue

                # Calculate intersection
                intersection_start = max(true_region.start, pred_region.start)
                intersection_end = min(true_region.end, pred_region.end)

                if intersection_start < intersection_end:
                    intersection_len = intersection_end - intersection_start

                    # Calculate union
                    union_len = (
                        (true_region.end - true_region.start) +
                        (pred_region.end - pred_region.start) -
                        intersection_len
                    )

                    # Calculate IoU
                    iou = intersection_len / union_len if union_len > 0 else 0.0

                    if iou > best_iou:
                        best_iou = iou
                        best_match = pred_region
                        best_idx = idx

            # Record match if IoU >= threshold
            if best_iou >= iou_threshold:
                matches.append((true_region, best_match, best_iou))
                pred_matched[best_idx] = True
            else:
                unmatched_true.append(true_region)

        # Collect unmatched predictions (false positives)
        for idx, pred_region in enumerate(pred_list):
            if not pred_matched[idx]:
                unmatched_pred.append(pred_region)

    # Compute aggregate metrics
    TP = len(matches)
    FP = len(unmatched_pred)
    FN = len(unmatched_true)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    iou_scores = [iou for _, _, iou in matches]
    mean_iou = float(np.mean(iou_scores)) if iou_scores else 0.0
    median_iou = float(np.median(iou_scores)) if iou_scores else 0.0

    return {
        'true_positives': matches,
        'false_positives': unmatched_pred,
        'false_negatives': unmatched_true,
        'per_element_ious': iou_scores,
        'mean_iou': mean_iou,
        'median_iou': median_iou,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'TP': TP,
        'FP': FP,
        'FN': FN
    }


def compute_global_nucleotide_iou(
    pred_nucleotides: np.ndarray,
    true_nucleotides: np.ndarray
) -> Dict[str, float]:
    """
    Compute global nucleotide-level IoU.

    Simple element-wise comparison across all nucleotides.

    IoU = intersection / union
    intersection = sum(pred & true)
    union = sum(pred | true)

    Args:
        pred_nucleotides: Binary array (0/1) for predictions
        true_nucleotides: Binary array (0/1) for ground truth

    Returns:
        Dictionary with:
            - 'intersection': Number of bp in intersection
            - 'union': Number of bp in union
            - 'iou': Intersection over Union
            - 'dice': Dice coefficient (2*inter / (sum(pred) + sum(true)))
            - 'jaccard': Jaccard index (same as IoU)
    """
    # Ensure same length
    if len(pred_nucleotides) != len(true_nucleotides):
        raise ValueError("Prediction and ground truth arrays must have same length")

    # Calculate intersection and union
    intersection = np.sum((pred_nucleotides == 1) & (true_nucleotides == 1))
    union = np.sum((pred_nucleotides == 1) | (true_nucleotides == 1))

    # IoU
    iou = float(intersection / union) if union > 0 else 0.0

    # Dice coefficient
    pred_sum = np.sum(pred_nucleotides)
    true_sum = np.sum(true_nucleotides)
    dice = float(2 * intersection / (pred_sum + true_sum)) if (pred_sum + true_sum) > 0 else 0.0

    return {
        'intersection': int(intersection),
        'union': int(union),
        'iou': iou,
        'dice': dice,
        'jaccard': iou  # Same as IoU
    }


def compute_region_confusion_matrix(
    predicted_regions: List[TERegion],
    true_regions: List[TERegion],
    iou_threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Compute region-level confusion matrix.

    Uses per-element matching results to create confusion matrix.

    Args:
        predicted_regions: List of predicted TERegion objects
        true_regions: List of ground truth TERegion objects
        iou_threshold: Minimum IoU for a match (default: 0.5)

    Returns:
        Dictionary with:
            - 'TP': True positives (matched pairs)
            - 'FP': False positives (unmatched predictions)
            - 'FN': False negatives (unmatched ground truth)
            - 'precision': TP / (TP + FP)
            - 'recall': TP / (TP + FN)
            - 'f1': F1 score
    """
    # Reuse per-element IoU calculation
    iou_results = compute_iou_per_element(
        predicted_regions,
        true_regions,
        iou_threshold
    )

    return {
        'TP': iou_results['TP'],
        'FP': iou_results['FP'],
        'FN': iou_results['FN'],
        'precision': iou_results['precision'],
        'recall': iou_results['recall'],
        'f1': iou_results['f1']
    }


def compute_length_distribution_stats(regions: List[TERegion]) -> Dict[str, float]:
    """
    Compute statistics on TE length distribution.

    Args:
        regions: List of TERegion objects

    Returns:
        Dictionary with length statistics
    """
    if len(regions) == 0:
        return {
            'n_regions': 0,
            'mean_length': 0,
            'median_length': 0,
            'std_length': 0,
            'min_length': 0,
            'max_length': 0,
            'total_bp_covered': 0
        }

    lengths = [r.end - r.start for r in regions]

    return {
        'n_regions': len(regions),
        'mean_length': float(np.mean(lengths)),
        'median_length': float(np.median(lengths)),
        'std_length': float(np.std(lengths)),
        'min_length': int(np.min(lengths)),
        'max_length': int(np.max(lengths)),
        'total_bp_covered': int(np.sum(lengths))
    }


def compute_per_chromosome_metrics(
    predicted_regions: List[TERegion],
    true_regions: List[TERegion],
    iou_threshold: float = 0.5
) -> Dict[str, Dict[str, Any]]:
    """
    Compute metrics separately for each chromosome.

    Args:
        predicted_regions: List of predicted TERegion objects
        true_regions: List of ground truth TERegion objects
        iou_threshold: Minimum IoU for a match

    Returns:
        Dictionary mapping seqid → metrics dictionary
    """
    # Group regions by chromosome
    pred_by_chr = defaultdict(list)
    true_by_chr = defaultdict(list)

    for r in predicted_regions:
        pred_by_chr[r.seqid].append(r)
    for r in true_regions:
        true_by_chr[r.seqid].append(r)

    # Compute metrics for each chromosome
    all_seqids = set(pred_by_chr.keys()) | set(true_by_chr.keys())
    per_chr_metrics = {}

    for seqid in all_seqids:
        chr_pred = pred_by_chr.get(seqid, [])
        chr_true = true_by_chr.get(seqid, [])

        # Per-element IoU
        iou_results = compute_iou_per_element(chr_pred, chr_true, iou_threshold)

        # Length stats
        pred_length_stats = compute_length_distribution_stats(chr_pred)
        true_length_stats = compute_length_distribution_stats(chr_true)

        per_chr_metrics[seqid] = {
            'per_element_metrics': {
                'TP': iou_results['TP'],
                'FP': iou_results['FP'],
                'FN': iou_results['FN'],
                'precision': iou_results['precision'],
                'recall': iou_results['recall'],
                'f1': iou_results['f1'],
                'mean_iou': iou_results['mean_iou']
            },
            'predicted': pred_length_stats,
            'true': true_length_stats
        }

    return per_chr_metrics


def calculate_nucleotide_metrics(
    pred_nucleotides: np.ndarray,
    true_nucleotides: np.ndarray
) -> Dict[str, Any]:
    """
    Calculate comprehensive nucleotide-level metrics.

    Args:
        pred_nucleotides: Binary array (0/1) for predictions
        true_nucleotides: Binary array (0/1) for ground truth

    Returns:
        Dictionary with nucleotide-level metrics
    """
    # Confusion matrix values
    TP = np.sum((pred_nucleotides == 1) & (true_nucleotides == 1))
    TN = np.sum((pred_nucleotides == 0) & (true_nucleotides == 0))
    FP = np.sum((pred_nucleotides == 1) & (true_nucleotides == 0))
    FN = np.sum((pred_nucleotides == 0) & (true_nucleotides == 1))

    # Metrics
    precision = float(TP / (TP + FP)) if (TP + FP) > 0 else 0.0
    recall = float(TP / (TP + FN)) if (TP + FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = float((TP + TN) / (TP + TN + FP + FN)) if (TP + TN + FP + FN) > 0 else 0.0

    # IoU
    iou_results = compute_global_nucleotide_iou(pred_nucleotides, true_nucleotides)

    return {
        'confusion_matrix': {
            'TP': int(TP),
            'TN': int(TN),
            'FP': int(FP),
            'FN': int(FN)
        },
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'iou': iou_results['iou'],
        'dice': iou_results['dice']
    }


def compare_length_distributions(
    predicted_regions: List[TERegion],
    true_regions: List[TERegion]
) -> Dict[str, Any]:
    """
    Compare length distributions between predictions and ground truth.

    Args:
        predicted_regions: List of predicted TERegion objects
        true_regions: List of ground truth TERegion objects

    Returns:
        Dictionary with comparison statistics
    """
    pred_lengths = [r.end - r.start for r in predicted_regions]
    true_lengths = [r.end - r.start for r in true_regions]

    # Basic stats
    pred_stats = compute_length_distribution_stats(predicted_regions)
    true_stats = compute_length_distribution_stats(true_regions)

    # Comparison
    return {
        'predicted': pred_stats,
        'true': true_stats,
        'difference': {
            'mean_diff': pred_stats['mean_length'] - true_stats['mean_length'],
            'median_diff': pred_stats['median_length'] - true_stats['median_length'],
            'total_bp_diff': pred_stats['total_bp_covered'] - true_stats['total_bp_covered']
        }
    }


def summarize_evaluation_results(
    predicted_regions: List[TERegion],
    true_regions: List[TERegion],
    pred_nucleotides: np.ndarray = None,
    true_nucleotides: np.ndarray = None,
    iou_threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Generate comprehensive evaluation summary.

    Args:
        predicted_regions: List of predicted TERegion objects
        true_regions: List of ground truth TERegion objects
        pred_nucleotides: Optional binary array for nucleotide-level metrics
        true_nucleotides: Optional binary array for nucleotide-level metrics
        iou_threshold: Minimum IoU for matching

    Returns:
        Dictionary with all evaluation metrics
    """
    summary = {}

    # Per-element metrics
    per_element = compute_iou_per_element(predicted_regions, true_regions, iou_threshold)
    summary['per_element_metrics'] = {
        'TP': per_element['TP'],
        'FP': per_element['FP'],
        'FN': per_element['FN'],
        'precision': per_element['precision'],
        'recall': per_element['recall'],
        'f1': per_element['f1'],
        'mean_iou': per_element['mean_iou'],
        'median_iou': per_element['median_iou'],
        'iou_threshold': iou_threshold,
        'per_element_ious': per_element['per_element_ious']
    }

    # Nucleotide-level metrics (if provided)
    if pred_nucleotides is not None and true_nucleotides is not None:
        nuc_metrics = calculate_nucleotide_metrics(pred_nucleotides, true_nucleotides)
        summary['nucleotide_metrics'] = nuc_metrics

    # Length distribution comparison
    length_comparison = compare_length_distributions(predicted_regions, true_regions)
    summary['length_distributions'] = length_comparison

    # Per-chromosome breakdown
    per_chr = compute_per_chromosome_metrics(predicted_regions, true_regions, iou_threshold)
    summary['per_chromosome'] = per_chr

    return summary
