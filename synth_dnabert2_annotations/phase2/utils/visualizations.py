#!/usr/bin/env python3
"""
Visualization utilities for TE prediction evaluation.

Generates genome browser tracks, IoU distributions, confusion matrices,
and nucleotide-level comparisons.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from pathlib import Path
from typing import List, Dict, Any
import seaborn as sns

from utils.postprocessing import TERegion


def plot_genome_browser_tracks(
    chromosome_id: str,
    true_regions: List[TERegion],
    pred_regions: List[TERegion],
    nucleotide_true: np.ndarray = None,
    nucleotide_pred: np.ndarray = None,
    output_path: Path = None,
    start: int = 0,
    end: int = None,
    window_size: int = 50000,
    figsize: tuple = (16, 10)
):
    """
    Create genome browser style visualization.

    Shows 4 tracks:
    1. True annotations (green bars)
    2. Predicted annotations (blue bars)
    3. Nucleotide-level true (optional heatmap)
    4. Nucleotide-level predicted (optional heatmap)

    Args:
        chromosome_id: Chromosome name
        true_regions: List of ground truth TERegion objects
        pred_regions: List of predicted TERegion objects
        nucleotide_true: Optional binary array for nucleotide-level truth
        nucleotide_pred: Optional binary array for nucleotide-level predictions
        output_path: Output file path
        start: Start position (bp)
        end: End position (bp), None = use chromosome length
        window_size: Maximum window size per plot (bp)
        figsize: Figure size (width, height)
    """
    # Determine end position
    if end is None:
        if nucleotide_true is not None:
            end = len(nucleotide_true)
        elif nucleotide_pred is not None:
            end = len(nucleotide_pred)
        else:
            # Use max region end
            max_true = max([r.end for r in true_regions]) if true_regions else 0
            max_pred = max([r.end for r in pred_regions]) if pred_regions else 0
            end = max(max_true, max_pred)

    # Filter regions to window
    true_in_window = [r for r in true_regions if r.start < end and r.end > start]
    pred_in_window = [r for r in pred_regions if r.start < end and r.end > start]

    # Decide number of tracks
    n_tracks = 2  # Always show region annotations
    if nucleotide_true is not None:
        n_tracks += 1
    if nucleotide_pred is not None:
        n_tracks += 1

    # Create figure
    fig, axes = plt.subplots(n_tracks, 1, figsize=figsize, sharex=True)
    if n_tracks == 1:
        axes = [axes]

    track_idx = 0

    # Track 1: True regions
    ax = axes[track_idx]
    for region in true_in_window:
        ax.add_patch(mpatches.Rectangle(
            (region.start, 0), region.end - region.start, 1,
            facecolor='green', alpha=0.6, edgecolor='darkgreen'
        ))
    ax.set_ylim(0, 1)
    ax.set_ylabel('True TEs', fontsize=10)
    ax.set_yticks([])
    ax.grid(True, alpha=0.3)
    track_idx += 1

    # Track 2: Predicted regions
    ax = axes[track_idx]
    for region in pred_in_window:
        # Color by confidence
        if region.score > 0.8:
            color = 'darkblue'
        elif region.score > 0.5:
            color = 'blue'
        else:
            color = 'lightblue'

        ax.add_patch(mpatches.Rectangle(
            (region.start, 0), region.end - region.start, 1,
            facecolor=color, alpha=0.6, edgecolor='navy'
        ))
    ax.set_ylim(0, 1)
    ax.set_ylabel('Pred TEs', fontsize=10)
    ax.set_yticks([])
    ax.grid(True, alpha=0.3)
    track_idx += 1

    # Track 3: Nucleotide-level true (if provided)
    if nucleotide_true is not None:
        ax = axes[track_idx]
        window_nuc_true = nucleotide_true[start:end]
        # Downsample if too long
        if len(window_nuc_true) > 10000:
            downsample = len(window_nuc_true) // 10000
            window_nuc_true = window_nuc_true[::downsample]
            x_coords = np.arange(start, end, downsample)
        else:
            x_coords = np.arange(start, start + len(window_nuc_true))

        ax.fill_between(x_coords, 0, window_nuc_true,
                        color='green', alpha=0.3, step='mid')
        ax.set_ylim(0, 1)
        ax.set_ylabel('True (bp)', fontsize=10)
        ax.set_yticks([0, 1])
        ax.grid(True, alpha=0.3)
        track_idx += 1

    # Track 4: Nucleotide-level predicted (if provided)
    if nucleotide_pred is not None:
        ax = axes[track_idx]
        window_nuc_pred = nucleotide_pred[start:end]
        # Downsample if too long
        if len(window_nuc_pred) > 10000:
            downsample = len(window_nuc_pred) // 10000
            window_nuc_pred = window_nuc_pred[::downsample]
            x_coords = np.arange(start, end, downsample)
        else:
            x_coords = np.arange(start, start + len(window_nuc_pred))

        ax.fill_between(x_coords, 0, window_nuc_pred,
                        color='blue', alpha=0.3, step='mid')
        ax.set_ylim(0, 1)
        ax.set_ylabel('Pred (bp)', fontsize=10)
        ax.set_yticks([0, 1])
        ax.grid(True, alpha=0.3)
        track_idx += 1

    # Set x-label
    axes[-1].set_xlabel(f'{chromosome_id} Position (bp)', fontsize=12)
    axes[-1].set_xlim(start, end)

    # Title
    fig.suptitle(f'TE Annotations - {chromosome_id}:{start:,}-{end:,}',
                fontsize=14, fontweight='bold')

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved genome browser track to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_iou_distribution(
    iou_scores: List[float],
    output_path: Path,
    iou_threshold: float = 0.5,
    figsize: tuple = (12, 5)
):
    """
    Plot distribution of per-element IoU scores.

    Two subplots:
    - Histogram with bins
    - Boxplot with quartiles

    Args:
        iou_scores: List of IoU scores (0-1)
        output_path: Output file path
        iou_threshold: Threshold line to show
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    if len(iou_scores) == 0:
        # No matches - empty plot
        for ax in axes:
            ax.text(0.5, 0.5, 'No matched regions',
                   ha='center', va='center', fontsize=14)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        fig.suptitle('IoU Distribution', fontsize=14, fontweight='bold')
        plt.tight_layout()
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return

    # Subplot 1: Histogram
    ax = axes[0]
    ax.hist(iou_scores, bins=np.linspace(0, 1, 21),
           color='skyblue', edgecolor='navy', alpha=0.7)

    # Add statistics lines
    mean_iou = np.mean(iou_scores)
    median_iou = np.median(iou_scores)
    ax.axvline(mean_iou, color='red', linestyle='--',
              linewidth=2, label=f'Mean: {mean_iou:.3f}')
    ax.axvline(median_iou, color='orange', linestyle='--',
              linewidth=2, label=f'Median: {median_iou:.3f}')
    ax.axvline(iou_threshold, color='green', linestyle=':',
              linewidth=2, label=f'Threshold: {iou_threshold}')

    ax.set_xlabel('IoU Score', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('IoU Score Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Subplot 2: Boxplot
    ax = axes[1]
    bp = ax.boxplot(iou_scores, vert=False, patch_artist=True,
                   boxprops=dict(facecolor='skyblue', alpha=0.7),
                   medianprops=dict(color='red', linewidth=2))

    # Add threshold line
    ax.axvline(iou_threshold, color='green', linestyle=':',
              linewidth=2, label=f'Threshold: {iou_threshold}')

    ax.set_xlabel('IoU Score', fontsize=12)
    ax.set_title('IoU Score Box Plot', fontsize=12, fontweight='bold')
    ax.set_yticks([])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')

    fig.suptitle(f'Per-Element IoU Distribution (n={len(iou_scores)})',
                fontsize=14, fontweight='bold')

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved IoU distribution plot to {output_path}")

    plt.close()


def plot_confusion_matrix_regions(
    cm_dict: Dict[str, int],
    output_path: Path,
    figsize: tuple = (10, 6)
):
    """
    Visualize region-level confusion matrix.

    Bar plot showing TP, FP, FN with precision, recall, F1 annotations.

    Args:
        cm_dict: Dictionary with TP, FP, FN, precision, recall, f1
        output_path: Output file path
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Data
    categories = ['True Positives', 'False Positives', 'False Negatives']
    values = [cm_dict['TP'], cm_dict['FP'], cm_dict['FN']]
    colors = ['green', 'orange', 'red']

    # Bar plot
    bars = ax.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{int(height)}',
               ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Add metrics annotation
    metrics_text = (
        f"Precision: {cm_dict['precision']:.3f}\n"
        f"Recall: {cm_dict['recall']:.3f}\n"
        f"F1 Score: {cm_dict['f1']:.3f}"
    )
    ax.text(0.98, 0.97, metrics_text,
           transform=ax.transAxes,
           fontsize=11,
           verticalalignment='top',
           horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('Region-Level Confusion Matrix', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved confusion matrix plot to {output_path}")

    plt.close()


def plot_nucleotide_comparison(
    true_nucleotides: np.ndarray,
    pred_nucleotides: np.ndarray,
    output_path: Path,
    chromosome_id: str = "",
    sample_interval: int = 100,
    max_length: int = 100000,
    figsize: tuple = (16, 8)
):
    """
    Nucleotide-by-nucleotide comparison heatmap.

    Three rows:
    1. True labels (white=bg, green=TE)
    2. Predicted labels (white=bg, blue=TE)
    3. Errors (white=correct, orange=FP, red=FN)

    For large chromosomes: sample every N nucleotides to keep plot readable.

    Args:
        true_nucleotides: Binary array (0/1) for ground truth
        pred_nucleotides: Binary array (0/1) for predictions
        output_path: Output file path
        chromosome_id: Chromosome name for title
        sample_interval: Sample every N nucleotides
        max_length: Maximum length to plot
        figsize: Figure size
    """
    # Limit length
    if len(true_nucleotides) > max_length:
        true_nucleotides = true_nucleotides[:max_length]
        pred_nucleotides = pred_nucleotides[:max_length]
        title_suffix = f" (first {max_length:,} bp)"
    else:
        title_suffix = ""

    # Sample if needed
    if len(true_nucleotides) > 10000:
        indices = np.arange(0, len(true_nucleotides), sample_interval)
        true_sampled = true_nucleotides[indices]
        pred_sampled = pred_nucleotides[indices]
    else:
        true_sampled = true_nucleotides
        pred_sampled = pred_nucleotides
        indices = np.arange(len(true_nucleotides))

    # Calculate errors
    # 0 = correct, 1 = FP, 2 = FN
    errors = np.zeros_like(true_sampled)
    errors[(pred_sampled == 1) & (true_sampled == 0)] = 1  # FP
    errors[(pred_sampled == 0) & (true_sampled == 1)] = 2  # FN

    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)

    # Row 1: True
    ax = axes[0]
    ax.imshow(true_sampled.reshape(1, -1), aspect='auto', cmap='Greens',
             vmin=0, vmax=1, interpolation='nearest')
    ax.set_ylabel('True', fontsize=11)
    ax.set_yticks([])

    # Row 2: Predicted
    ax = axes[1]
    ax.imshow(pred_sampled.reshape(1, -1), aspect='auto', cmap='Blues',
             vmin=0, vmax=1, interpolation='nearest')
    ax.set_ylabel('Predicted', fontsize=11)
    ax.set_yticks([])

    # Row 3: Errors
    ax = axes[2]
    # Custom colormap: white=correct, orange=FP, red=FN
    colors = ['white', 'orange', 'red']
    cmap = ListedColormap(colors)
    im = ax.imshow(errors.reshape(1, -1), aspect='auto', cmap=cmap,
                  vmin=0, vmax=2, interpolation='nearest')
    ax.set_ylabel('Errors', fontsize=11)
    ax.set_yticks([])

    # X-axis
    ax.set_xlabel('Position (bp)', fontsize=12)
    # Show actual positions
    n_ticks = 10
    tick_indices = np.linspace(0, len(indices)-1, n_ticks, dtype=int)
    tick_positions = indices[tick_indices]
    ax.set_xticks(tick_indices)
    ax.set_xticklabels([f'{int(pos):,}' for pos in tick_positions], rotation=45)

    # Legend for error row
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='white', edgecolor='black', label='Correct'),
        Patch(facecolor='orange', edgecolor='black', label='False Positive'),
        Patch(facecolor='red', edgecolor='black', label='False Negative')
    ]
    axes[2].legend(handles=legend_elements, loc='upper right', fontsize=9)

    # Title
    title = f'Nucleotide-Level Comparison'
    if chromosome_id:
        title += f' - {chromosome_id}'
    title += title_suffix
    fig.suptitle(title, fontsize=14, fontweight='bold')

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved nucleotide comparison plot to {output_path}")

    plt.close()


def plot_length_distribution_comparison(
    true_regions: List[TERegion],
    pred_regions: List[TERegion],
    output_path: Path,
    figsize: tuple = (12, 6)
):
    """
    Compare length distributions.

    Two overlaid histograms:
    - True TE lengths (green, transparent)
    - Predicted TE lengths (blue, transparent)

    Log scale for x-axis (TE lengths vary widely).

    Args:
        true_regions: List of ground truth TERegion objects
        pred_regions: List of predicted TERegion objects
        output_path: Output file path
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Extract lengths
    true_lengths = [r.end - r.start for r in true_regions]
    pred_lengths = [r.end - r.start for r in pred_regions]

    if len(true_lengths) == 0 and len(pred_lengths) == 0:
        ax.text(0.5, 0.5, 'No regions to compare',
               ha='center', va='center', fontsize=14)
        plt.tight_layout()
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        return

    # Determine bins (log scale)
    all_lengths = true_lengths + pred_lengths
    if len(all_lengths) > 0:
        min_len = max(1, min(all_lengths))
        max_len = max(all_lengths)
        bins = np.logspace(np.log10(min_len), np.log10(max_len), 30)
    else:
        bins = np.logspace(1, 5, 30)

    # Plot histograms
    if len(true_lengths) > 0:
        ax.hist(true_lengths, bins=bins, color='green', alpha=0.5,
               label=f'True (n={len(true_lengths)})', edgecolor='darkgreen')

    if len(pred_lengths) > 0:
        ax.hist(pred_lengths, bins=bins, color='blue', alpha=0.5,
               label=f'Predicted (n={len(pred_lengths)})', edgecolor='navy')

    # Add statistics
    if len(true_lengths) > 0:
        ax.axvline(np.mean(true_lengths), color='darkgreen', linestyle='--',
                  linewidth=2, label=f'True mean: {np.mean(true_lengths):.0f} bp')
    if len(pred_lengths) > 0:
        ax.axvline(np.mean(pred_lengths), color='darkblue', linestyle='--',
                  linewidth=2, label=f'Pred mean: {np.mean(pred_lengths):.0f} bp')

    ax.set_xscale('log')
    ax.set_xlabel('TE Length (bp, log scale)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('TE Length Distribution Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved length distribution comparison to {output_path}")

    plt.close()


def create_evaluation_summary_figure(
    metrics_dict: Dict[str, Any],
    output_path: Path,
    figsize: tuple = (16, 10)
):
    """
    Create comprehensive evaluation summary figure.

    Multiple subplots showing key metrics.

    Args:
        metrics_dict: Dictionary with all evaluation metrics
        output_path: Output file path
        figsize: Figure size
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # Subplot 1: Per-element metrics bar chart
    ax1 = fig.add_subplot(gs[0, 0])
    per_elem = metrics_dict.get('per_element_metrics', {})
    metrics = ['Precision', 'Recall', 'F1', 'Mean IoU']
    values = [
        per_elem.get('precision', 0),
        per_elem.get('recall', 0),
        per_elem.get('f1', 0),
        per_elem.get('mean_iou', 0)
    ]
    bars = ax1.bar(metrics, values, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'],
                  alpha=0.7, edgecolor='black')
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=10)
    ax1.set_ylim(0, 1)
    ax1.set_ylabel('Score', fontsize=11)
    ax1.set_title('Per-Element Metrics', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    # Subplot 2: Confusion matrix
    ax2 = fig.add_subplot(gs[0, 1])
    categories = ['TP', 'FP', 'FN']
    values = [per_elem.get('TP', 0), per_elem.get('FP', 0), per_elem.get('FN', 0)]
    colors = ['green', 'orange', 'red']
    bars = ax2.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10)
    ax2.set_ylabel('Count', fontsize=11)
    ax2.set_title('Region-Level Confusion Matrix', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # Subplot 3: Nucleotide-level metrics (if available)
    ax3 = fig.add_subplot(gs[1, 0])
    nuc_metrics = metrics_dict.get('nucleotide_metrics', {})
    if nuc_metrics:
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'IoU']
        values = [
            nuc_metrics.get('accuracy', 0),
            nuc_metrics.get('precision', 0),
            nuc_metrics.get('recall', 0),
            nuc_metrics.get('f1', 0),
            nuc_metrics.get('iou', 0)
        ]
        bars = ax3.bar(metrics, values, color='lightblue', alpha=0.7, edgecolor='black')
        for bar in bars:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=9)
        ax3.set_ylim(0, 1)
        ax3.set_ylabel('Score', fontsize=11)
        ax3.set_title('Nucleotide-Level Metrics', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
    else:
        ax3.text(0.5, 0.5, 'Nucleotide metrics\nnot available',
                ha='center', va='center', fontsize=12)

    # Subplot 4: Length statistics
    ax4 = fig.add_subplot(gs[1, 1])
    length_dist = metrics_dict.get('length_distributions', {})
    pred_stats = length_dist.get('predicted', {})
    true_stats = length_dist.get('true', {})

    categories = ['N regions', 'Mean length', 'Total bp']
    pred_values = [
        pred_stats.get('n_regions', 0),
        pred_stats.get('mean_length', 0),
        pred_stats.get('total_bp_covered', 0) / 1000  # Convert to kb
    ]
    true_values = [
        true_stats.get('n_regions', 0),
        true_stats.get('mean_length', 0),
        true_stats.get('total_bp_covered', 0) / 1000
    ]

    x = np.arange(len(categories))
    width = 0.35
    ax4.bar(x - width/2, true_values, width, label='True', color='green', alpha=0.7)
    ax4.bar(x + width/2, pred_values, width, label='Predicted', color='blue', alpha=0.7)

    ax4.set_xticks(x)
    ax4.set_xticklabels(categories)
    ax4.set_ylabel('Value', fontsize=11)
    ax4.set_title('Length Statistics', fontsize=12, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    fig.suptitle('Evaluation Summary', fontsize=16, fontweight='bold')

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved evaluation summary figure to {output_path}")

    plt.close()
