#!/usr/bin/env python3
"""
Evaluate Fine-tuned DNABERT-2 Model on Test Set

This script evaluates a fine-tuned DNABERT-2 model on the test set,
computing detailed metrics and generating analysis reports.

Usage:
    python evaluate_model.py --model_path results/dnabert2_te_token_classification
    python evaluate_model.py --model_path results/dnabert2_te_token_classification --by_species
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml
from datasets import load_from_disk
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from tqdm import tqdm
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
)

from utils.data_collator import get_data_collator

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def predict_batch(model, batch, device):
    """
    Run model prediction on a batch.

    Args:
        model: The fine-tuned model
        batch: Batch of inputs
        device: Device to run on (cuda/cpu)

    Returns:
        Tuple of (predictions, labels) as numpy arrays
    """
    # Move batch to device
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)

    # Forward pass (no gradient needed)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

    # Get predictions (argmax)
    predictions = torch.argmax(logits, dim=-1)

    return predictions.cpu().numpy(), labels.cpu().numpy()


def compute_metrics(predictions: np.ndarray, labels: np.ndarray, ignore_index: int = -100) -> Dict:
    """
    Compute evaluation metrics.

    Args:
        predictions: Predicted labels (batch_size, seq_len)
        labels: True labels (batch_size, seq_len)
        ignore_index: Label index to ignore (default: -100)

    Returns:
        Dictionary of metrics
    """
    # Flatten and filter out ignored indices
    pred_flat = predictions.flatten()
    label_flat = labels.flatten()

    mask = label_flat != ignore_index
    pred_flat = pred_flat[mask]
    label_flat = label_flat[mask]

    # Compute metrics
    accuracy = accuracy_score(label_flat, pred_flat)
    precision, recall, f1, support = precision_recall_fscore_support(
        label_flat, pred_flat, average='binary', zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(label_flat, pred_flat)

    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        label_flat, pred_flat, average=None, zero_division=0
    )

    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'confusion_matrix': cm.tolist(),
        'per_class': {
            'background': {
                'precision': float(precision_per_class[0]),
                'recall': float(recall_per_class[0]),
                'f1': float(f1_per_class[0]),
                'support': int(support_per_class[0])
            },
            'TE': {
                'precision': float(precision_per_class[1]) if len(precision_per_class) > 1 else 0.0,
                'recall': float(recall_per_class[1]) if len(recall_per_class) > 1 else 0.0,
                'f1': float(f1_per_class[1]) if len(f1_per_class) > 1 else 0.0,
                'support': int(support_per_class[1]) if len(support_per_class) > 1 else 0
            }
        },
        'total_tokens': int(len(label_flat)),
        'TE_tokens': int((label_flat == 1).sum()),
        'background_tokens': int((label_flat == 0).sum())
    }


def evaluate_by_species(model, tokenizer, dataset, device, batch_size: int = 8) -> Dict[str, Dict]:
    """
    Evaluate model separately for each species.

    Args:
        model: The fine-tuned model
        tokenizer: The tokenizer
        dataset: Test dataset with 'species' field
        device: Device to run on
        batch_size: Batch size for evaluation

    Returns:
        Dictionary mapping species name to metrics
    """
    logger.info("Evaluating by species...")

    # Get unique species
    species_list = sorted(set(dataset['species']))
    logger.info(f"Found {len(species_list)} unique species")

    results = {}

    for species in tqdm(species_list, desc="Evaluating species"):
        # Filter dataset for this species
        species_dataset = dataset.filter(lambda x: x['species'] == species)
        logger.info(f"  {species}: {len(species_dataset)} samples")

        # Evaluate on this species
        all_predictions = []
        all_labels = []

        # Create data collator
        data_collator = get_data_collator(tokenizer)

        # Process in batches
        for i in range(0, len(species_dataset), batch_size):
            batch_data = [species_dataset[idx] for idx in range(i, min(i + batch_size, len(species_dataset)))]
            batch = data_collator(batch_data)

            predictions, labels = predict_batch(model, batch, device)
            # Flatten batch immediately to avoid shape mismatch during concatenation
            all_predictions.append(predictions.flatten())
            all_labels.append(labels.flatten())

        # Concatenate all flattened batches
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        # Reshape to 2D for compute_metrics
        all_predictions = all_predictions.reshape(-1, 1)
        all_labels = all_labels.reshape(-1, 1)

        # Compute metrics for this species
        species_metrics = compute_metrics(all_predictions, all_labels)
        results[species] = species_metrics

    return results


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned DNABERT-2 model")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to fine-tuned model directory"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--by_species",
        action="store_true",
        help="Compute metrics separately for each species"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save evaluation results (default: model_path/evaluation)"
    )
    args = parser.parse_args()

    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)

    # Set output directory
    if args.output_dir is None:
        output_dir = Path(args.model_path) / "evaluation"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Results will be saved to: {output_dir}")

    # Detect device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # ========================================================================
    # Load Model and Tokenizer
    # ========================================================================
    logger.info(f"Loading model from {args.model_path}")

    # Log best checkpoint info if available
    model_path = Path(args.model_path)
    trainer_state_file = model_path / "trainer_state.json"

    if trainer_state_file.exists():
        import json
        with open(trainer_state_file, 'r') as f:
            trainer_state = json.load(f)

        best_metric = trainer_state.get('best_metric', 'N/A')
        best_checkpoint_path = trainer_state.get('best_model_checkpoint', 'N/A')
        if best_checkpoint_path != 'N/A':
            checkpoint_name = Path(best_checkpoint_path).name
            logger.info(f"Training info: Best checkpoint was {checkpoint_name} with F1={best_metric:.4f}")

    # Load tokenizer
    logger.info("Loading tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True
    )
    logger.info("✅ Tokenizer loaded successfully")

    # Load model using HuggingFace from_pretrained
    logger.info(f"Loading model from: {args.model_path}")

    # Import custom model classes
    from train_token_classification import TokenClassificationModel

    model = TokenClassificationModel.from_pretrained(args.model_path)
    model.to(device)
    model.eval()
    logger.info("✅ Model loaded successfully")

    # ========================================================================
    # Load Test Dataset
    # ========================================================================
    dataset_path = Path(config['data']['dataset_path'])
    test_split = config['data']['test_split']

    logger.info(f"Loading test dataset from {dataset_path / test_split}")
    test_dataset = load_from_disk(str(dataset_path / test_split))
    logger.info(f"Loaded {len(test_dataset)} test samples")

    # ========================================================================
    # Overall Evaluation
    # ========================================================================
    logger.info("=" * 80)
    logger.info("Running overall evaluation on test set...")
    logger.info("=" * 80)

    all_predictions = []
    all_labels = []

    # Create data collator
    data_collator = get_data_collator(tokenizer)

    # Process in batches with progress bar
    for i in tqdm(range(0, len(test_dataset), args.batch_size), desc="Evaluating"):
        # Get batch as list of individual samples
        batch_indices = range(i, min(i + args.batch_size, len(test_dataset)))
        batch_data = [test_dataset[idx] for idx in batch_indices]
        batch = data_collator(batch_data)

        predictions, labels = predict_batch(model, batch, device)
        # Flatten batch immediately to avoid shape mismatch during concatenation
        all_predictions.append(predictions.flatten())
        all_labels.append(labels.flatten())

    # Concatenate all flattened batches
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Reshape to 2D for compute_metrics (which expects shape (batch, seq_len))
    # Since we flattened, we need to reshape to (-1, 1) so compute_metrics can flatten again
    all_predictions = all_predictions.reshape(-1, 1)
    all_labels = all_labels.reshape(-1, 1)

    # Compute overall metrics
    overall_metrics = compute_metrics(all_predictions, all_labels)

    # ========================================================================
    # Print Results
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("OVERALL TEST SET RESULTS")
    logger.info("=" * 80)
    logger.info(f"Accuracy: {overall_metrics['accuracy']:.4f}")
    logger.info(f"Precision: {overall_metrics['precision']:.4f}")
    logger.info(f"Recall: {overall_metrics['recall']:.4f}")
    logger.info(f"F1 Score: {overall_metrics['f1']:.4f}")
    logger.info(f"\nTotal tokens evaluated: {overall_metrics['total_tokens']:,}")
    logger.info(f"  Background tokens: {overall_metrics['background_tokens']:,}")
    logger.info(f"  TE tokens: {overall_metrics['TE_tokens']:,}")
    logger.info(f"\nPer-class metrics:")
    logger.info(f"  Background - Precision: {overall_metrics['per_class']['background']['precision']:.4f}, "
                f"Recall: {overall_metrics['per_class']['background']['recall']:.4f}, "
                f"F1: {overall_metrics['per_class']['background']['f1']:.4f}")
    logger.info(f"  TE - Precision: {overall_metrics['per_class']['TE']['precision']:.4f}, "
                f"Recall: {overall_metrics['per_class']['TE']['recall']:.4f}, "
                f"F1: {overall_metrics['per_class']['TE']['f1']:.4f}")
    logger.info(f"\nConfusion Matrix:")
    logger.info(f"  [[TN={overall_metrics['confusion_matrix'][0][0]}, FP={overall_metrics['confusion_matrix'][0][1]}]")
    logger.info(f"   [FN={overall_metrics['confusion_matrix'][1][0]}, TP={overall_metrics['confusion_matrix'][1][1]}]]")
    logger.info("=" * 80)

    # Save overall results
    results_file = output_dir / "test_results.json"
    with open(results_file, 'w') as f:
        json.dump(overall_metrics, f, indent=2)
    logger.info(f"\nOverall results saved to: {results_file}")

    # ========================================================================
    # Species-specific Evaluation (Optional)
    # ========================================================================
    if args.by_species:
        logger.info("\n" + "=" * 80)
        logger.info("SPECIES-SPECIFIC EVALUATION")
        logger.info("=" * 80)

        species_results = evaluate_by_species(model, tokenizer, test_dataset, device, args.batch_size)

        # Print summary
        logger.info("\nSpecies-specific F1 scores:")
        for species, metrics in sorted(species_results.items(), key=lambda x: x[1]['f1'], reverse=True):
            logger.info(f"  {species:30s}: F1={metrics['f1']:.4f}, "
                        f"Precision={metrics['precision']:.4f}, "
                        f"Recall={metrics['recall']:.4f}")

        # Save species results
        species_file = output_dir / "test_results_by_species.json"
        with open(species_file, 'w') as f:
            json.dump(species_results, f, indent=2)
        logger.info(f"\nSpecies-specific results saved to: {species_file}")

    # ========================================================================
    # Summary
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Test samples: {len(test_dataset)}")
    logger.info(f"Overall F1 score: {overall_metrics['f1']:.4f}")
    logger.info(f"Results directory: {output_dir}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
