#!/usr/bin/env python3
"""
Verify Dataset and Environment Before Training

This script performs pre-training checks to ensure:
1. Dataset files exist and are accessible
2. Dataset format is correct
3. Samples can be loaded and batched
4. GPU memory is sufficient

Usage:
    python verify_before_training.py --config config.yaml
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict

import torch
import yaml
from datasets import load_from_disk
from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer

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


def check_dataset_exists(dataset_path: Path, split_name: str) -> bool:
    """Check if dataset split exists."""
    split_path = dataset_path / split_name
    if not split_path.exists():
        logger.error(f"Dataset split not found: {split_path}")
        return False

    # Check for essential files
    essential_files = ['dataset_info.json', 'state.json']
    for file in essential_files:
        if not (split_path / file).exists():
            logger.error(f"Missing essential file: {split_path / file}")
            return False

    logger.info(f"✓ Dataset split exists: {split_path}")
    return True


def check_dataset_format(dataset) -> bool:
    """Check if dataset has correct format."""
    required_fields = ['input_ids', 'attention_mask', 'labels']
    optional_fields = ['sequence_id', 'window_start', 'species', 'n_tes']

    # Check required fields
    for field in required_fields:
        if field not in dataset.features:
            logger.error(f"Missing required field: {field}")
            return False

    logger.info(f"✓ Dataset has all required fields: {required_fields}")

    # Check optional fields (just log warnings)
    for field in optional_fields:
        if field in dataset.features:
            logger.info(f"  - Optional field present: {field}")
        else:
            logger.warning(f"  - Optional field missing: {field}")

    return True


def check_sample_validity(dataset, num_samples: int = 5) -> bool:
    """Check if samples are valid."""
    logger.info(f"Checking {num_samples} random samples...")

    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]

        # Check that input_ids, attention_mask, and labels have same length
        input_len = len(sample['input_ids'])
        mask_len = len(sample['attention_mask'])
        label_len = len(sample['labels'])

        if not (input_len == mask_len == label_len):
            logger.error(f"Sample {i}: Length mismatch - "
                         f"input_ids={input_len}, attention_mask={mask_len}, labels={label_len}")
            return False

        # Check label values (should be 0, 1, or -100)
        valid_labels = set([0, 1, -100])
        sample_labels = set(sample['labels'])
        if not sample_labels.issubset(valid_labels):
            logger.error(f"Sample {i}: Invalid label values: {sample_labels - valid_labels}")
            return False

        # Check attention_mask values (should be 0 or 1)
        valid_mask = set([0, 1])
        sample_mask = set(sample['attention_mask'])
        if not sample_mask.issubset(valid_mask):
            logger.error(f"Sample {i}: Invalid attention_mask values: {sample_mask - valid_mask}")
            return False

    logger.info(f"✓ All {num_samples} samples are valid")
    return True


def check_data_collation(tokenizer, dataset, batch_size: int = 8) -> bool:
    """Check if data collator works correctly."""
    logger.info("Testing data collator...")

    try:
        data_collator = get_data_collator(tokenizer)

        # Create a small batch
        batch_samples = [dataset[i] for i in range(min(batch_size, len(dataset)))]

        # Collate
        batch = data_collator(batch_samples)

        # Check batch format
        required_keys = ['input_ids', 'attention_mask', 'labels']
        for key in required_keys:
            if key not in batch:
                logger.error(f"Batch missing key: {key}")
                return False

            if not isinstance(batch[key], torch.Tensor):
                logger.error(f"Batch[{key}] is not a tensor: {type(batch[key])}")
                return False

        # Check batch dimensions
        batch_size_actual = batch['input_ids'].shape[0]
        seq_len = batch['input_ids'].shape[1]

        logger.info(f"✓ Data collator works correctly")
        logger.info(f"  Batch shape: ({batch_size_actual}, {seq_len})")

        return True

    except Exception as e:
        logger.error(f"Data collator failed: {e}")
        return False


def estimate_memory_requirements(config: Dict) -> None:
    """Estimate GPU memory requirements."""
    logger.info("Estimating memory requirements...")

    # Model parameters (DNABERT-2-117M has ~117M parameters)
    model_params = 117_000_000

    # Memory per parameter (4 bytes for fp32, 2 bytes for fp16)
    bytes_per_param = 2 if config['training']['fp16'] else 4

    # Model memory
    model_memory_gb = (model_params * bytes_per_param) / (1024 ** 3)

    # Optimizer states (Adam typically needs 2x model params for momentum + variance)
    optimizer_memory_gb = model_memory_gb * 2

    # Activations (rough estimate based on batch size and sequence length)
    batch_size = config['training']['per_device_train_batch_size']
    seq_len = config['data']['max_length']
    hidden_size = 768  # DNABERT-2 hidden size

    # Activation memory (batch_size * seq_len * hidden_size * num_layers * dtype)
    num_layers = 12  # DNABERT-2 has 12 layers
    activation_memory_gb = (batch_size * seq_len * hidden_size * num_layers * bytes_per_param) / (1024 ** 3)

    # Total memory (add 20% overhead)
    total_memory_gb = (model_memory_gb + optimizer_memory_gb + activation_memory_gb) * 1.2

    logger.info(f"  Model memory: {model_memory_gb:.2f} GB")
    logger.info(f"  Optimizer memory: {optimizer_memory_gb:.2f} GB")
    logger.info(f"  Activation memory: {activation_memory_gb:.2f} GB")
    logger.info(f"  Total estimated: {total_memory_gb:.2f} GB per GPU")

    # Check available GPU memory
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            total_mem = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
            logger.info(f"  GPU {i} ({torch.cuda.get_device_name(i)}): {total_mem:.2f} GB available")

            if total_mem < total_memory_gb:
                logger.warning(f"  ⚠ GPU {i} may not have enough memory!")
                logger.warning(f"    Consider reducing batch size or enabling gradient checkpointing")
            else:
                logger.info(f"  ✓ GPU {i} has sufficient memory")
    else:
        logger.warning("No GPU available - training will be very slow on CPU")


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Verify dataset and environment before training")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("PRE-TRAINING VERIFICATION")
    logger.info("=" * 80)

    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    logger.info("✓ Configuration loaded successfully")

    # Track verification status
    all_checks_passed = True

    # ========================================================================
    # Check Dataset Existence
    # ========================================================================
    logger.info("\n" + "-" * 80)
    logger.info("1. Checking dataset existence...")
    logger.info("-" * 80)

    dataset_path = Path(config['data']['dataset_path'])
    logger.info(f"Dataset path: {dataset_path}")

    if not dataset_path.exists():
        logger.error(f"Dataset path does not exist: {dataset_path}")
        sys.exit(1)

    # Check each split
    for split_name in ['train_split', 'validation_split', 'test_split']:
        split = config['data'][split_name]
        if not check_dataset_exists(dataset_path, split):
            all_checks_passed = False

    # ========================================================================
    # Load and Check Training Dataset
    # ========================================================================
    logger.info("\n" + "-" * 80)
    logger.info("2. Loading and checking training dataset...")
    logger.info("-" * 80)

    try:
        train_split = config['data']['train_split']
        train_dataset = load_from_disk(str(dataset_path / train_split))
        logger.info(f"✓ Training dataset loaded: {len(train_dataset)} samples")

        # Check format
        if not check_dataset_format(train_dataset):
            all_checks_passed = False

        # Check sample validity
        if not check_sample_validity(train_dataset, num_samples=10):
            all_checks_passed = False

        # Print sample statistics
        logger.info(f"\nDataset statistics:")
        logger.info(f"  Total samples: {len(train_dataset)}")

        if 'species' in train_dataset.features:
            # Fast species counting using Counter (vectorized operation)
            from collections import Counter
            species_counts = Counter(train_dataset['species'])

            logger.info(f"  Number of species: {len(species_counts)}")
            logger.info(f"  Samples per species: {len(train_dataset) // len(species_counts)}")

    except Exception as e:
        logger.error(f"Failed to load training dataset: {e}")
        all_checks_passed = False

    # ========================================================================
    # Check Tokenizer and Model Loading
    # ========================================================================
    logger.info("\n" + "-" * 80)
    logger.info("3. Checking model and tokenizer...")
    logger.info("-" * 80)

    try:
        model_name = config['model']['name_or_path']
        logger.info(f"Loading tokenizer from {model_name}")

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        logger.info("✓ Tokenizer loaded successfully")

        logger.info(f"Loading model config from {model_name}")
        model_config = AutoConfig.from_pretrained(
            model_name,
            num_labels=config['model']['num_labels'],
            trust_remote_code=True
        )
        logger.info("✓ Model config loaded successfully")

        # Note: We don't load the full model here to save time
        logger.info("  (Full model loading will happen during training)")

    except Exception as e:
        logger.error(f"Failed to load model/tokenizer: {e}")
        all_checks_passed = False

    # ========================================================================
    # Check Data Collator
    # ========================================================================
    logger.info("\n" + "-" * 80)
    logger.info("4. Testing data collator...")
    logger.info("-" * 80)

    try:
        if not check_data_collation(tokenizer, train_dataset, batch_size=config['training']['per_device_train_batch_size']):
            all_checks_passed = False
    except Exception as e:
        logger.error(f"Data collation test failed: {e}")
        all_checks_passed = False

    # ========================================================================
    # Check GPU Availability and Memory
    # ========================================================================
    logger.info("\n" + "-" * 80)
    logger.info("5. Checking GPU availability...")
    logger.info("-" * 80)

    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        logger.info(f"✓ CUDA available with {num_gpus} GPU(s)")

        for i in range(num_gpus):
            logger.info(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

        # Estimate memory requirements
        estimate_memory_requirements(config)
    else:
        logger.warning("⚠ No GPU available - training will be very slow")
        logger.warning("  Consider running on a machine with GPU support")

    # ========================================================================
    # Check Output Directory
    # ========================================================================
    logger.info("\n" + "-" * 80)
    logger.info("6. Checking output directory...")
    logger.info("-" * 80)

    output_dir = Path(config['output']['output_dir'])
    logger.info(f"Output directory: {output_dir}")

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("✓ Output directory is accessible")
    except Exception as e:
        logger.error(f"Cannot create output directory: {e}")
        all_checks_passed = False

    # ========================================================================
    # Summary
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("VERIFICATION SUMMARY")
    logger.info("=" * 80)

    if all_checks_passed:
        logger.info("✓ All checks passed! Ready to start training.")
        logger.info("\nTo start training, run:")
        logger.info(f"  sbatch submit_training.sh")
        logger.info("\nOr for local testing:")
        logger.info(f"  python train_token_classification.py --config {args.config} --debug")
        sys.exit(0)
    else:
        logger.error("✗ Some checks failed. Please fix the issues before training.")
        logger.error("\nReview the error messages above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
