#!/usr/bin/env python3
"""
DNABERT-2 TE Region Prediction and Evaluation Pipeline

Main script for:
1. Loading trained DNABERT-2 model
2. Processing FASTA sequences
3. Predicting TE regions
4. Writing GFF3 and BED outputs
5. (Optional) Evaluating against ground truth with comprehensive metrics and visualizations
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

import torch
import numpy as np
from Bio import SeqIO
from transformers import BertForTokenClassification, AutoTokenizer
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.postprocessing import (
    WindowPrediction, TERegion,
    reconstruct_nucleotide_predictions,
    detect_te_regions,
    get_region_stats
)
from utils.format_converters import (
    write_gff3, write_bed, parse_gff3_regions,
    build_nucleotide_array_from_regions,
    load_chromosome_lengths_from_fasta,
    calculate_genome_coverage, sort_regions,
    write_regions_by_chromosome
)
from utils.region_metrics import (
    summarize_evaluation_results,
    compute_per_chromosome_metrics
)
from utils.visualizations import (
    plot_genome_browser_tracks,
    plot_iou_distribution,
    plot_confusion_matrix_regions,
    plot_nucleotide_comparison,
    plot_length_distribution_comparison,
    create_evaluation_summary_figure
)


@dataclass
class ChromosomeResult:
    """Complete chromosome processing result."""
    sequence_id: str
    sequence_length: int
    nucleotide_predictions: np.ndarray  # Binary array (0/1)
    nucleotide_confidence: np.ndarray   # Confidence scores (0-1)
    predicted_regions: List[TERegion]
    n_windows: int
    processing_time: float


def load_model_and_tokenizer(model_path: str, device: str) -> Tuple:
    """
    Load fine-tuned DNABERT-2 model and tokenizer.

    Args:
        model_path: Path to fine-tuned model directory
        device: Device to load model on ('cuda' or 'cpu')

    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"\n{'='*80}")
    print("LOADING MODEL")
    print(f"{'='*80}")
    print(f"Model path: {model_path}")
    print(f"Device: {device}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    print(f"✓ Tokenizer loaded (max_length: {tokenizer.model_max_length})")

    # Load model with token classification head
    # Note: Using BertForTokenClassification directly (same as training script)
    # due to compatibility with DNABERT-2's custom configuration class
    model = BertForTokenClassification.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    model.to(device)
    model.eval()
    print(f"✓ Model loaded and set to eval mode")
    print(f"  Model config: num_labels = {model.config.num_labels}")

    return model, tokenizer


def generate_sliding_windows(sequence: str, window_size: int = 2048, stride: int = 2048):
    """
    Generate sliding windows over sequence.

    Reuses strategy from prepare_dnabert2_data.py.

    Args:
        sequence: DNA sequence string
        window_size: Window size in bp (default: 2048)
        stride: Stride in bp (default: 2048, no overlap)

    Yields:
        Tuple of (start_position, window_sequence)
    """
    seq_len = len(sequence)

    for start in range(0, seq_len - window_size + 1, stride):
        end = start + window_size
        yield (start, sequence[start:end])

    # Last window if we didn't reach the end
    if seq_len % stride != 0:
        start = max(0, seq_len - window_size)
        yield (start, sequence[start:seq_len])


def sliding_window_inference(
    sequence: str,
    model,
    tokenizer,
    window_size: int,
    stride: int,
    batch_size: int,
    device: str
) -> List[WindowPrediction]:
    """
    Run batch inference on sliding windows.

    Args:
        sequence: DNA sequence
        model: Fine-tuned DNABERT-2 model
        tokenizer: DNABERT-2 tokenizer
        window_size: Window size in bp
        stride: Stride in bp
        batch_size: Batch size for inference
        device: Device to run on

    Returns:
        List of WindowPrediction objects
    """
    # Generate all windows
    windows = list(generate_sliding_windows(sequence, window_size, stride))

    if len(windows) == 0:
        return []

    # BERT models have a hard limit of 512 tokens in their architecture
    # (due to internal buffered_token_type_ids with fixed size)
    max_length = 512

    # Process in batches
    all_window_predictions = []

    for batch_start in tqdm(range(0, len(windows), batch_size),
                           desc="Inference", disable=len(windows) < 10):
        batch_windows = windows[batch_start:batch_start + batch_size]

        # Tokenize batch
        batch_sequences = [w[1] for w in batch_windows]
        tokenized = tokenizer(
            batch_sequences,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors='pt',
            return_offsets_mapping=True
        )

        # Move to device
        input_ids = tokenized['input_ids'].to(device)
        attention_mask = tokenized['attention_mask'].to(device)
        offset_mapping = tokenized['offset_mapping']  # Keep on CPU

        # Inference
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # Shape: (batch, seq_len, num_labels)

        # Get predictions
        predictions = torch.argmax(logits, dim=-1)  # Shape: (batch, seq_len)

        # Convert to numpy
        predictions_np = predictions.cpu().numpy()
        logits_np = logits.cpu().numpy()

        # Store results
        for idx, (window_start, window_seq) in enumerate(batch_windows):
            window_end = window_start + len(window_seq)

            all_window_predictions.append(WindowPrediction(
                window_start=window_start,
                window_end=window_end,
                token_predictions=predictions_np[idx],
                offset_mapping=offset_mapping[idx].tolist(),
                logits=logits_np[idx]
            ))

    return all_window_predictions


def process_chromosome(
    seq_record,
    model,
    tokenizer,
    args
) -> ChromosomeResult:
    """
    Process single chromosome end-to-end.

    Args:
        seq_record: BioPython SeqRecord
        model: Fine-tuned DNABERT-2 model
        tokenizer: DNABERT-2 tokenizer
        args: Command line arguments

    Returns:
        ChromosomeResult object
    """
    start_time = time.time()

    sequence_id = seq_record.id
    sequence = str(seq_record.seq).upper()
    sequence_length = len(sequence)

    print(f"\nProcessing {sequence_id}: {sequence_length:,} bp")

    # Step 1: Run sliding window inference
    window_predictions = sliding_window_inference(
        sequence=sequence,
        model=model,
        tokenizer=tokenizer,
        window_size=args.window_size,
        stride=args.stride,
        batch_size=args.batch_size,
        device=args.device
    )

    print(f"  ✓ Generated {len(window_predictions)} window predictions")

    # Step 2: Reconstruct nucleotide-level predictions
    nucleotide_predictions, nucleotide_confidence = reconstruct_nucleotide_predictions(
        window_predictions,
        sequence_length
    )

    print(f"  ✓ Reconstructed nucleotide predictions")
    print(f"    - TE nucleotides: {np.sum(nucleotide_predictions):,} "
          f"({np.sum(nucleotide_predictions) / sequence_length * 100:.2f}%)")

    # Step 3: Detect TE regions
    predicted_regions = detect_te_regions(
        nucleotide_predictions,
        nucleotide_confidence,
        seqid=sequence_id,
        min_length=args.min_te_length,
        merge_gap=args.merge_gap
    )

    print(f"  ✓ Detected {len(predicted_regions)} TE regions")

    # Region statistics
    if len(predicted_regions) > 0:
        region_stats = get_region_stats(predicted_regions)
        print(f"    - Mean length: {region_stats['mean_length']:.0f} bp")
        print(f"    - Total coverage: {region_stats['total_bp']:,} bp")

    processing_time = time.time() - start_time

    return ChromosomeResult(
        sequence_id=sequence_id,
        sequence_length=sequence_length,
        nucleotide_predictions=nucleotide_predictions,
        nucleotide_confidence=nucleotide_confidence,
        predicted_regions=predicted_regions,
        n_windows=len(window_predictions),
        processing_time=processing_time
    )


def process_fasta(
    fasta_path: Path,
    model,
    tokenizer,
    args
) -> Dict[str, ChromosomeResult]:
    """
    Process all sequences in FASTA file.

    Args:
        fasta_path: Path to FASTA file
        model: Fine-tuned DNABERT-2 model
        tokenizer: DNABERT-2 tokenizer
        args: Command line arguments

    Returns:
        Dictionary mapping sequence_id → ChromosomeResult
    """
    print(f"\n{'='*80}")
    print("PROCESSING FASTA")
    print(f"{'='*80}")
    print(f"Input: {fasta_path}")

    results = {}

    for seq_record in SeqIO.parse(fasta_path, "fasta"):
        result = process_chromosome(seq_record, model, tokenizer, args)
        results[result.sequence_id] = result

    return results


def write_outputs(
    results: Dict[str, ChromosomeResult],
    output_dir: Path,
    args
):
    """
    Write predictions to GFF3 and BED formats.

    Args:
        results: Dictionary of ChromosomeResult objects
        output_dir: Output directory
        args: Command line arguments
    """
    print(f"\n{'='*80}")
    print("WRITING OUTPUTS")
    print(f"{'='*80}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect all regions
    all_regions = []
    for result in results.values():
        all_regions.extend(result.predicted_regions)

    # Sort regions
    all_regions = sort_regions(all_regions)

    print(f"Total predicted regions: {len(all_regions)}")

    # Write combined GFF3
    gff3_path = output_dir / "predictions.gff3"
    write_gff3(all_regions, gff3_path)

    # Write combined BED
    bed_path = output_dir / "predictions.bed"
    write_bed(all_regions, bed_path)

    # Write per-chromosome files
    chr_dir = output_dir / "predictions_by_chromosome"
    write_regions_by_chromosome(all_regions, chr_dir, format='gff3')
    write_regions_by_chromosome(all_regions, chr_dir, format='bed')

    # Write summary statistics
    chromosome_lengths = {r.sequence_id: r.sequence_length for r in results.values()}
    coverage = calculate_genome_coverage(all_regions, chromosome_lengths)

    summary = {
        'timestamp': datetime.now().isoformat(),
        'model_path': args.model_path,
        'input_fasta': str(args.fasta),
        'total_chromosomes': len(results),
        'total_genome_bp': coverage['total_genome_bp'],
        'total_te_regions': coverage['n_regions'],
        'total_te_bp': coverage['total_te_bp'],
        'genome_coverage_pct': coverage['genome_coverage_pct'],
        'per_chromosome': {}
    }

    for seq_id, result in results.items():
        region_stats = get_region_stats(result.predicted_regions)
        summary['per_chromosome'][seq_id] = {
            'length_bp': result.sequence_length,
            'n_regions': region_stats['n_regions'],
            'te_bp': region_stats['total_bp'],
            'coverage_pct': (region_stats['total_bp'] / result.sequence_length * 100)
                           if result.sequence_length > 0 else 0,
            'mean_region_length': region_stats['mean_length'],
            'processing_time_sec': result.processing_time
        }

    summary_path = output_dir / "summary_statistics.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"✓ Wrote summary statistics to {summary_path}")


def run_evaluation(
    results: Dict[str, ChromosomeResult],
    gff3_path: Path,
    output_dir: Path,
    args
):
    """
    Run evaluation against ground truth.

    Args:
        results: Dictionary of ChromosomeResult objects
        gff3_path: Path to ground truth GFF3 file
        output_dir: Output directory
        args: Command line arguments
    """
    print(f"\n{'='*80}")
    print("EVALUATION AGAINST GROUND TRUTH")
    print(f"{'='*80}")
    print(f"Ground truth: {gff3_path}")

    # Parse ground truth
    true_regions_by_seq = parse_gff3_regions(gff3_path)

    # Collect predictions
    pred_regions_by_seq = {}
    for seq_id, result in results.items():
        pred_regions_by_seq[seq_id] = result.predicted_regions

    # Flatten to lists
    all_true = []
    all_pred = []
    for seq_id in set(true_regions_by_seq.keys()) | set(pred_regions_by_seq.keys()):
        all_true.extend(true_regions_by_seq.get(seq_id, []))
        all_pred.extend(pred_regions_by_seq.get(seq_id, []))

    print(f"Total true regions: {len(all_true)}")
    print(f"Total predicted regions: {len(all_pred)}")

    # Build nucleotide arrays for global IoU
    print("\nBuilding nucleotide arrays for global IoU...")
    all_true_nuc = []
    all_pred_nuc = []

    for seq_id, result in results.items():
        true_regions = true_regions_by_seq.get(seq_id, [])
        true_nuc = build_nucleotide_array_from_regions(true_regions, result.sequence_length)
        pred_nuc = result.nucleotide_predictions

        all_true_nuc.append(true_nuc)
        all_pred_nuc.append(pred_nuc)

    # Concatenate
    all_true_nuc = np.concatenate(all_true_nuc) if all_true_nuc else np.array([])
    all_pred_nuc = np.concatenate(all_pred_nuc) if all_pred_nuc else np.array([])

    # Compute metrics
    print("\nComputing evaluation metrics...")
    eval_results = summarize_evaluation_results(
        predicted_regions=all_pred,
        true_regions=all_true,
        pred_nucleotides=all_pred_nuc,
        true_nucleotides=all_true_nuc,
        iou_threshold=args.iou_threshold
    )

    # Print summary
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")

    per_elem = eval_results['per_element_metrics']
    print(f"Per-Element Metrics (IoU threshold: {per_elem['iou_threshold']}):")
    print(f"  Precision: {per_elem['precision']:.4f}")
    print(f"  Recall:    {per_elem['recall']:.4f}")
    print(f"  F1 Score:  {per_elem['f1']:.4f}")
    print(f"  Mean IoU:  {per_elem['mean_iou']:.4f}")
    print(f"  TP: {per_elem['TP']}, FP: {per_elem['FP']}, FN: {per_elem['FN']}")

    if 'nucleotide_metrics' in eval_results:
        nuc_metrics = eval_results['nucleotide_metrics']
        print(f"\nNucleotide-Level Metrics:")
        print(f"  IoU:       {nuc_metrics['iou']:.4f}")
        print(f"  Dice:      {nuc_metrics['dice']:.4f}")
        print(f"  Precision: {nuc_metrics['precision']:.4f}")
        print(f"  Recall:    {nuc_metrics['recall']:.4f}")
        print(f"  F1 Score:  {nuc_metrics['f1']:.4f}")

    # Write metrics
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # Overall metrics
    overall_metrics_path = metrics_dir / "overall_metrics.json"
    with open(overall_metrics_path, 'w') as f:
        # Convert for JSON serialization
        metrics_json = {
            'per_element_metrics': eval_results['per_element_metrics'],
            'nucleotide_metrics': eval_results.get('nucleotide_metrics', {}),
            'length_distributions': eval_results['length_distributions']
        }
        json.dump(metrics_json, f, indent=2)

    print(f"✓ Wrote overall metrics to {overall_metrics_path}")

    # Per-chromosome metrics
    per_chr_metrics = eval_results['per_chromosome']
    per_chr_path = metrics_dir / "per_chromosome_metrics.json"
    with open(per_chr_path, 'w') as f:
        json.dump(per_chr_metrics, f, indent=2)

    print(f"✓ Wrote per-chromosome metrics to {per_chr_path}")

    # IoU distribution CSV
    iou_scores = eval_results['per_element_metrics']['per_element_ious']
    if len(iou_scores) > 0:
        iou_csv_path = metrics_dir / "iou_distribution.csv"
        with open(iou_csv_path, 'w') as f:
            f.write("iou_score\n")
            for score in iou_scores:
                f.write(f"{score:.6f}\n")
        print(f"✓ Wrote IoU distribution to {iou_csv_path}")

    # Generate visualizations if requested
    if args.visualize:
        print(f"\n{'='*80}")
        print("GENERATING VISUALIZATIONS")
        print(f"{'='*80}")

        vis_dir = output_dir / "visualizations"
        vis_dir.mkdir(parents=True, exist_ok=True)

        # 1. IoU distribution
        if len(iou_scores) > 0:
            plot_iou_distribution(
                iou_scores,
                vis_dir / "iou_distribution.png",
                iou_threshold=args.iou_threshold
            )

        # 2. Confusion matrix
        cm_dict = {
            'TP': per_elem['TP'],
            'FP': per_elem['FP'],
            'FN': per_elem['FN'],
            'precision': per_elem['precision'],
            'recall': per_elem['recall'],
            'f1': per_elem['f1']
        }
        plot_confusion_matrix_regions(cm_dict, vis_dir / "confusion_matrix.png")

        # 3. Length distribution comparison
        plot_length_distribution_comparison(
            all_true,
            all_pred,
            vis_dir / "length_distribution_comparison.png"
        )

        # 4. Evaluation summary figure
        create_evaluation_summary_figure(
            eval_results,
            vis_dir / "evaluation_summary.png"
        )

        # 5. Genome browser tracks (first few chromosomes)
        tracks_dir = vis_dir / "genome_tracks"
        tracks_dir.mkdir(exist_ok=True)

        for seq_id in list(results.keys())[:3]:  # First 3 chromosomes
            result = results[seq_id]
            true_regions = true_regions_by_seq.get(seq_id, [])
            pred_regions = result.predicted_regions

            # Build nucleotide arrays
            true_nuc = build_nucleotide_array_from_regions(true_regions, result.sequence_length)
            pred_nuc = result.nucleotide_predictions

            # Plot full chromosome or windows
            chr_length = result.sequence_length

            # Always plot full chromosome
            plot_genome_browser_tracks(
                seq_id, true_regions, pred_regions,
                true_nuc, pred_nuc,
                tracks_dir / f"{seq_id}_full.png"
            )

            # If chromosome > 100kb, also create 100kb windows for detailed view
            if chr_length > 100000:
                window_size = 100000
                n_windows = (chr_length + window_size - 1) // window_size  # Ceiling division

                for i in range(n_windows):
                    start = i * window_size
                    end = min((i + 1) * window_size, chr_length)

                    plot_genome_browser_tracks(
                        seq_id, true_regions, pred_regions,
                        true_nuc, pred_nuc,
                        tracks_dir / f"{seq_id}_{start//1000}-{end//1000}kb.png",
                        start=start, end=end
                    )

        # 6. Nucleotide comparison (first chromosome)
        nuc_comp_dir = vis_dir / "nucleotide_comparison"
        nuc_comp_dir.mkdir(exist_ok=True)

        first_seq_id = list(results.keys())[0]
        result = results[first_seq_id]
        true_regions = true_regions_by_seq.get(first_seq_id, [])
        true_nuc = build_nucleotide_array_from_regions(true_regions, result.sequence_length)
        pred_nuc = result.nucleotide_predictions

        plot_nucleotide_comparison(
            true_nuc, pred_nuc,
            nuc_comp_dir / f"{first_seq_id}_comparison.png",
            chromosome_id=first_seq_id,
            max_length=len(true_nuc)  # Show full chromosome instead of default 100kb
        )

        print(f"✓ Generated all visualizations in {vis_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='DNABERT-2 TE Region Prediction and Evaluation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument('--fasta', type=str, required=True,
                       help='Input FASTA file')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to fine-tuned DNABERT-2 model')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory')

    # Optional arguments
    parser.add_argument('--gff3', type=str, default=None,
                       help='Ground truth GFF3 file for evaluation')
    parser.add_argument('--batch_size', type=int, default=192,
                       help='Batch size for inference')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to run inference on')

    # Window parameters
    parser.add_argument('--window_size', type=int, default=2048,
                       help='Window size in bp')
    parser.add_argument('--stride', type=int, default=1024,
                       help='Stride in bp (default: 1024 for 50%% overlap)')

    # Region detection parameters
    parser.add_argument('--min_te_length', type=int, default=50,
                       help='Minimum TE length in bp')
    parser.add_argument('--merge_gap', type=int, default=10,
                       help='Maximum gap to merge adjacent regions')

    # Evaluation parameters
    parser.add_argument('--iou_threshold', type=float, default=0.5,
                       help='IoU threshold for region matching')
    parser.add_argument('--visualize', action='store_true',
                       help='Generate evaluation visualizations')

    args = parser.parse_args()

    # Convert paths
    args.fasta = Path(args.fasta)
    args.output_dir = Path(args.output_dir)
    if args.gff3:
        args.gff3 = Path(args.gff3)

    # Validate inputs
    if not args.fasta.exists():
        print(f"Error: FASTA file not found: {args.fasta}")
        sys.exit(1)

    if not Path(args.model_path).exists():
        print(f"Error: Model path not found: {args.model_path}")
        sys.exit(1)

    if args.gff3 and not args.gff3.exists():
        print(f"Error: GFF3 file not found: {args.gff3}")
        sys.exit(1)

    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        args.device = 'cpu'

    # Print configuration
    print(f"\n{'='*80}")
    print("DNABERT-2 TE PREDICTION PIPELINE")
    print(f"{'='*80}")
    print(f"Configuration:")
    print(f"  FASTA:        {args.fasta}")
    print(f"  Model:        {args.model_path}")
    print(f"  Output:       {args.output_dir}")
    print(f"  GFF3:         {args.gff3 if args.gff3 else 'None (inference only)'}")
    print(f"  Batch size:   {args.batch_size}")
    print(f"  Device:       {args.device}")
    print(f"  Window size:  {args.window_size} bp")
    print(f"  Stride:       {args.stride} bp")
    print(f"  Min TE len:   {args.min_te_length} bp")
    print(f"  Merge gap:    {args.merge_gap} bp")

    start_time = time.time()

    # Step 1: Load model
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.device)

    # Step 2: Process FASTA
    results = process_fasta(args.fasta, model, tokenizer, args)

    # Step 3: Write outputs
    write_outputs(results, args.output_dir, args)

    # Step 4: Evaluation (if GFF3 provided)
    if args.gff3:
        run_evaluation(results, args.gff3, args.output_dir, args)

    # Summary
    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print("PIPELINE COMPLETE")
    print(f"{'='*80}")
    print(f"Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print(f"Output directory: {args.output_dir}")
    print(f"\nGenerated files:")
    print(f"  - predictions.gff3")
    print(f"  - predictions.bed")
    print(f"  - predictions_by_chromosome/")
    print(f"  - summary_statistics.json")
    if args.gff3:
        print(f"  - metrics/")
        if args.visualize:
            print(f"  - visualizations/")


if __name__ == "__main__":
    main()
