#!/usr/bin/env python3
"""
DNABERT-2 Token Classification Fine-tuning for Transposable Element Detection (AutoModel Version)

This script fine-tunes DNABERT-2 on pre-tokenized genomic sequences for token-level
classification of transposable elements (TEs). It uses the standard HuggingFace
AutoModelForTokenClassification architecture with dynamic class weights calculated
from the training dataset distribution.

Key Differences from train_token_classification.py:
- Uses AutoModelForTokenClassification (standard HF) instead of custom model
- Class weights calculated dynamically with sklearn from dataset labels
- WeightedLossTrainer for applying class weights (AutoModel doesn't support in config)

Usage:
    python train_token_classification_automodel.py --config config.yaml
    python train_token_classification_automodel.py --config config.yaml --debug  # Test with small subset
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import yaml
from datasets import load_from_disk
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForTokenClassification,
    AutoTokenizer,
    BertForTokenClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)
from transformers.modeling_outputs import TokenClassifierOutput
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

# Import custom data collator
from utils.data_collator import get_data_collator

# Configure logging with forced flushing
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    force=True,  # Force reconfiguration
)
logger = logging.getLogger(__name__)

# Force unbuffered output for real-time logging
import functools
print = functools.partial(print, flush=True)


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def compute_metrics(eval_pred):
    """
    Compute metrics for token classification.

    Args:
        eval_pred: EvalPrediction object containing predictions and labels

    Returns:
        Dictionary of metrics (accuracy, precision, recall, f1)
    """
    predictions, labels = eval_pred

    # Get predicted class (argmax over logits)
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens with label -100)
    true_predictions = [
        [p for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [l for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    # Flatten lists for sklearn metrics
    flat_predictions = [p for sublist in true_predictions for p in sublist]
    flat_labels = [l for sublist in true_labels for l in sublist]

    # Compute metrics
    accuracy = accuracy_score(flat_labels, flat_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        flat_labels, flat_predictions, average='binary', zero_division=0
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ========================================================================
# WeightedLossTrainer (Custom Trainer for Class Weights)
# ========================================================================
# NOTE: AutoModelForTokenClassification does NOT support class_weights in config
# Therefore, we need a custom Trainer that overrides compute_loss()
# This applies class weights during loss calculation

class WeightedLossTrainer(Trainer):
    """
    Custom Trainer that applies class weights to CrossEntropyLoss.

    Required because AutoModelForTokenClassification doesn't have
    class_weights in its config (unlike our custom TokenClassificationModel).

    Args:
        class_weights: torch.Tensor of shape (num_labels,) with weights for each class
    """

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

        if class_weights is not None:
            logger.info(f"WeightedLossTrainer initialized with class weights: {class_weights.tolist()}")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Override compute_loss to apply class weights.

        The default Trainer computes loss inside the model's forward().
        Here we extract logits and compute weighted loss externally.
        """
        labels = inputs.pop("labels")  # Extract labels before forward pass

        # Forward pass (without labels, so model doesn't compute loss)
        outputs = model(**inputs)
        logits = outputs.logits

        # Compute weighted loss
        if self.class_weights is not None:
            loss_fct = nn.CrossEntropyLoss(
                weight=self.class_weights.to(logits.device),
                ignore_index=-100
            )
        else:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

        # Reshape for loss calculation
        loss = loss_fct(
            logits.view(-1, self.model.config.num_labels),
            labels.view(-1)
        )

        return (loss, outputs) if return_outputs else loss


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Fine-tune DNABERT-2 for token classification")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode with small subset of data"
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from"
    )
    parser.add_argument(
        "--imbalance-ratio",
        type=float,
        default=None,
        help="Class imbalance ratio (background/TE). If not provided, uses config value or global default (2.96)"
    )
    args = parser.parse_args()

    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)

    # Override config with command line arguments
    if args.debug:
        config['advanced']['debug_mode'] = True
        logger.info("Running in DEBUG mode with limited data")

    if args.resume_from_checkpoint:
        config['advanced']['resume_from_checkpoint'] = args.resume_from_checkpoint
        logger.info(f"Resuming from checkpoint: {args.resume_from_checkpoint}")

    if args.imbalance_ratio is not None:
        config['training']['imbalance_ratio'] = args.imbalance_ratio
        logger.info(f"Using imbalance_ratio from command line: {args.imbalance_ratio}")

    # Set seed for reproducibility
    set_seed(config['training']['seed'])

    # Create output directory
    output_dir = Path(config['output']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # ========================================================================
    # Load Tokenizer
    # ========================================================================
    logger.info(f"Loading tokenizer from {config['model']['name_or_path']}")
    tokenizer = AutoTokenizer.from_pretrained(
        config['model']['name_or_path'],
        trust_remote_code=True
    )

    # Ensure tokenizer has pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Set pad_token to eos_token: {tokenizer.pad_token}")

    # ========================================================================
    # Load Datasets
    # ========================================================================
    logger.info(f"Loading datasets from {config['data']['dataset_path']}")

    dataset_path = Path(config['data']['dataset_path'])

    # Load train dataset
    train_dataset = load_from_disk(str(dataset_path / config['data']['train_split']))
    logger.info(f"Loaded training dataset: {len(train_dataset)} samples")

    # Load validation dataset
    eval_dataset = load_from_disk(str(dataset_path / config['data']['validation_split']))
    logger.info(f"Loaded validation dataset: {len(eval_dataset)} samples")

    # Store full eval dataset for rotating subset selection
    full_eval_dataset = eval_dataset
    max_eval_samples = config['advanced'].get('max_eval_samples')
    if max_eval_samples is not None and max_eval_samples > 0 and len(full_eval_dataset) > max_eval_samples:
        logger.info(f"⚠️  Validation set will be limited to {max_eval_samples} samples per evaluation")
        logger.info(f"   Total validation samples: {len(full_eval_dataset)}")
        logger.info(f"   Strategy: Rotating subset - different samples each eval")
        logger.info(f"   Reason: Prevent OOM during evaluation with large validation set (18GB)")
        # Start with first subset (callback will rotate from here)
        eval_dataset = full_eval_dataset.select(range(max_eval_samples))
        logger.info(f"   Initial subset: samples 0-{max_eval_samples}")

    # Debug mode: use only a small subset
    if config['advanced']['debug_mode']:
        debug_samples = config['advanced']['debug_samples']
        train_dataset = train_dataset.select(range(min(debug_samples, len(train_dataset))))
        eval_dataset = eval_dataset.select(range(min(debug_samples // 10, len(eval_dataset))))
        logger.info(f"DEBUG mode: Using {len(train_dataset)} train and {len(eval_dataset)} eval samples")

    # Print dataset info
    logger.info(f"Dataset features: {train_dataset.features}")
    logger.info(f"Sample example: {train_dataset[0]}")

    # ========================================================================
    # Load Model (Token Classification)
    # ========================================================================
    logger.info(f"Loading model from {config['model']['name_or_path']}")

    # Create label mappings for binary classification
    # 0=Background, 1=TE
    id2label = {0: "Background", 1: "TE"}
    label2id = {"Background": 0, "TE": 1}

    logger.info(f"Label mapping: {label2id}")

    # Load DNABERT-2 with token classification head
    # Note: Using BertForTokenClassification directly instead of AutoModelForTokenClassification
    # due to compatibility issues with DNABERT-2's custom configuration class.
    # This is the standard HuggingFace approach and produces identical architecture.
    model = BertForTokenClassification.from_pretrained(
        config['model']['name_or_path'],
        num_labels=config['model']['num_labels'],
        id2label=id2label,
        label2id=label2id,
        trust_remote_code=True,
    )

    # Full fine-tuning approach - all base model parameters remain trainable
    logger.info("Using full fine-tuning - all base model parameters are trainable")

    # Resize token embeddings if needed
    if len(tokenizer) != model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))
        logger.info(f"Resized token embeddings to {len(tokenizer)}")

    logger.info("Model loaded successfully")
    logger.info(f"Model class: {type(model).__name__}")
    logger.info(f"Total model parameters: {model.num_parameters():,}")
    logger.info(f"Trainable parameters: {model.num_parameters(only_trainable=True):,}")

    # ========================================================================
    # Configure Class Weights for Imbalanced Data
    # ========================================================================
    # Two methods available:
    #   1. DYNAMIC: Calculate from training dataset labels (sklearn method)
    #   2. STATIC: Use fixed ratio from config (nucleotide-level proxy)
    #
    # Dynamic method is recommended for AutoModelForTokenClassification
    # because it reflects actual token-level distribution.
    # ========================================================================

    ENABLE_CLASS_WEIGHTS = True  # Master switch for class weighting
    use_dynamic_weights = config['training'].get('use_dynamic_class_weights', True)

    class_weights_tensor = None  # Will be passed to WeightedLossTrainer

    if ENABLE_CLASS_WEIGHTS:
        logger.info("\n" + "=" * 80)
        logger.info("CONFIGURING CLASS WEIGHTS FOR IMBALANCED DATA")
        logger.info("=" * 80)

        if use_dynamic_weights:
            # ============================================================
            # METHOD 1: DYNAMIC CALCULATION (sklearn compute_class_weight)
            # ============================================================
            logger.info("Method: DYNAMIC calculation from training labels")
            logger.info("  Computing class weights from training dataset...")

            # Flatten all labels from training dataset (filter -100)
            logger.info("  Extracting labels from training dataset...")
            train_labels_flat = []

            for sample in tqdm(train_dataset, desc="Processing labels"):
                labels = sample['labels']
                # Filter out -100 (padding/special tokens)
                valid_labels = [label for label in labels if label != -100]
                train_labels_flat.extend(valid_labels)

            logger.info(f"  Total valid tokens: {len(train_labels_flat):,}")

            # Compute class weights using sklearn
            unique_classes = np.unique(train_labels_flat)
            class_weights_array = compute_class_weight(
                class_weight='balanced',
                classes=unique_classes,
                y=train_labels_flat
            )

            # Create full weight array for all labels (in case some missing)
            class_weights_list = [1.0] * config['model']['num_labels']
            for cls_id, weight in zip(unique_classes, class_weights_array):
                class_weights_list[cls_id] = float(weight)

            # Convert to tensor for PyTorch
            class_weights_tensor = torch.FloatTensor(class_weights_list)

            logger.info(f"\nDynamic class weights calculated:")
            logger.info(f"  Background (class 0): {class_weights_list[0]:.4f}")
            logger.info(f"  TE (class 1):         {class_weights_list[1]:.4f}")
            logger.info(f"  Imbalance ratio:      {class_weights_list[1]/class_weights_list[0]:.4f}")

            # Calculate actual distribution
            n_background = sum(1 for l in train_labels_flat if l == 0)
            n_te = sum(1 for l in train_labels_flat if l == 1)
            logger.info(f"\nTraining data distribution:")
            logger.info(f"  Background tokens: {n_background:,} ({100*n_background/len(train_labels_flat):.2f}%)")
            logger.info(f"  TE tokens:         {n_te:,} ({100*n_te/len(train_labels_flat):.2f}%)")
            logger.info(f"  Ratio (bg/te):     {n_background/n_te:.4f}")

        else:
            # ============================================================
            # METHOD 2: STATIC RATIO (from config or default)
            # ============================================================
            logger.info("Method: STATIC ratio from config")

            # Get imbalance ratio from config (can be None/null)
            imbalance_ratio = config['training'].get('imbalance_ratio')

            # Default global ratio if not specified (1,012 species, 67.7 Gbp)
            DEFAULT_IMBALANCE_RATIO = 2.96

            if imbalance_ratio is None:
                imbalance_ratio = DEFAULT_IMBALANCE_RATIO
                logger.info(f"⚠️  No imbalance_ratio specified in config")
                logger.info(f"   Using global default: {DEFAULT_IMBALANCE_RATIO}")
                logger.info(f"   Based on: 1,012 species, 67.7 Gbp genomic data")
            else:
                logger.info(f"✅ Using imbalance_ratio from config: {imbalance_ratio}")

            # Calculate class weights: [background_weight, te_weight]
            class_weights_list = [1.0, float(imbalance_ratio)]
            class_weights_tensor = torch.FloatTensor(class_weights_list)

            logger.info(f"\nStatic class weights:")
            logger.info(f"  Background (class 0): {class_weights_list[0]:.2f}")
            logger.info(f"  TE (class 1):         {class_weights_list[1]:.2f}")

        logger.info("\n✅ Class weights configured")
        logger.info("   Will be applied via WeightedLossTrainer.compute_loss()")
        logger.info("=" * 80)

    else:
        logger.info("\n" + "=" * 80)
        logger.info("CLASS WEIGHTS DISABLED")
        logger.info("Using standard CrossEntropyLoss without class weighting")
        logger.info("⚠️  WARNING: Dataset is imbalanced - model may be biased")
        logger.info("=" * 80)

    # ========================================================================
    # Create Data Collator
    # ========================================================================
    data_collator = get_data_collator(tokenizer, max_length=config['data']['max_length'])
    logger.info("Data collator created")

    # ========================================================================
    # Training Arguments
    # ========================================================================
    training_args = TrainingArguments(
        # Output
        output_dir=str(output_dir),
        run_name=config['output']['run_name'],

        # Training hyperparameters
        learning_rate=config['training']['learning_rate'],
        num_train_epochs=config['training']['num_train_epochs'],
        per_device_train_batch_size=config['training']['per_device_train_batch_size'],
        per_device_eval_batch_size=config['training']['per_device_eval_batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        eval_accumulation_steps=config['training'].get('eval_accumulation_steps'),

        # Optimization
        weight_decay=config['training']['weight_decay'],
        adam_beta1=config['training']['adam_beta1'],
        adam_beta2=config['training']['adam_beta2'],
        adam_epsilon=config['training']['adam_epsilon'],
        max_grad_norm=config['training']['max_grad_norm'],
        optim=config['hardware']['optim'],

        # Learning rate schedule
        lr_scheduler_type=config['training']['lr_scheduler_type'],
        warmup_steps=config['training']['warmup_steps'],

        # Mixed precision
        fp16=config['training']['fp16'],

        # Evaluation
        evaluation_strategy=config['training']['evaluation_strategy'],
        eval_steps=config['training']['eval_steps'],

        # Saving
        save_strategy=config['training']['save_strategy'],
        save_steps=config['training']['save_steps'],
        save_total_limit=config['training']['save_total_limit'],
        load_best_model_at_end=config['training']['load_best_model_at_end'],
        metric_for_best_model=config['training']['metric_for_best_model'],
        greater_is_better=config['training']['greater_is_better'],

        # Logging
        logging_strategy=config['training']['logging_strategy'],
        logging_steps=config['training']['logging_steps'],
        logging_dir=config['output']['logging_dir'],
        report_to=config['training']['report_to'],

        # Misc
        seed=config['training']['seed'],
        dataloader_num_workers=config['training']['dataloader_num_workers'],
        dataloader_pin_memory=config['training']['dataloader_pin_memory'],
        ignore_data_skip=config['training']['ignore_data_skip'],

        # Distributed training
        ddp_backend=config['hardware']['distributed_backend'],

        # Gradient checkpointing for memory efficiency (if enabled)
        gradient_checkpointing=config['hardware']['gradient_checkpointing'],
    )

    logger.info("Training arguments configured")

    # ========================================================================
    # Debug Callback for Checkpoint Verification
    # ========================================================================
    class CheckpointDebugCallback(TrainerCallback):
        """Callback to debug checkpoint saving issues"""

        def on_save(self, args, state, control, **kwargs):
            """Called when a checkpoint is being saved"""
            checkpoint_folder = f"checkpoint-{state.global_step}"
            checkpoint_path = Path(args.output_dir) / checkpoint_folder

            logger.info("=" * 80)
            logger.info(f"🔍 CHECKPOINT SAVE TRIGGERED - Step {state.global_step}")
            logger.info(f"📁 Expected path: {checkpoint_path}")

            # Force flush to ensure this gets logged
            sys.stdout.flush()
            sys.stderr.flush()

            return control

        def on_epoch_end(self, args, state, control, **kwargs):
            """Called at the end of an epoch"""
            logger.info("=" * 80)
            logger.info(f"✅ EPOCH {state.epoch} COMPLETED")
            logger.info(f"   Global step: {state.global_step}")
            logger.info("=" * 80)
            sys.stdout.flush()
            return control

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            """Called after evaluation"""
            logger.info("=" * 80)
            logger.info(f"📊 EVALUATION at step {state.global_step}")
            if metrics:
                logger.info(f"   Metrics: {metrics}")
            logger.info("=" * 80)
            sys.stdout.flush()
            return control

    # ========================================================================
    # Rotating Validation Subset Callback
    # ========================================================================
    class RotatingEvalSubsetCallback(TrainerCallback):
        """Rotates through validation set to use different samples each eval"""

        def __init__(self, full_eval_dataset, max_eval_samples):
            self.full_eval_dataset = full_eval_dataset
            self.max_eval_samples = max_eval_samples
            self.total_samples = len(full_eval_dataset)
            self.eval_count = 0
            self.trainer = None  # Will be set after trainer initialization

        def on_evaluate(self, args, state, control, **kwargs):
            """Rotate eval dataset before evaluation"""
            if self.max_eval_samples >= self.total_samples:
                return control

            # Calculate offset using eval count (rotates sequentially)
            offset = (self.eval_count * self.max_eval_samples) % self.total_samples
            end_idx = offset + self.max_eval_samples

            # Handle wraparound
            if end_idx <= self.total_samples:
                indices = list(range(offset, end_idx))
            else:
                # Wrap around to beginning
                indices = list(range(offset, self.total_samples)) + list(range(0, end_idx - self.total_samples))

            # Update trainer's eval dataset
            if self.trainer is not None:
                self.trainer.eval_dataset = self.full_eval_dataset.select(indices)
                logger.info(f"🔄 Rotated eval subset: samples {offset}-{(offset + self.max_eval_samples) % self.total_samples} ({len(indices)} total)")
            else:
                logger.warning("⚠️  Trainer reference not set in RotatingEvalSubsetCallback")
            sys.stdout.flush()

            self.eval_count += 1
            return control

    # ========================================================================
    # Delayed Early Stopping Callback
    # ========================================================================
    class DelayedEarlyStoppingCallback(EarlyStoppingCallback):
        """Early stopping that only activates after minimum epochs completed"""

        def __init__(self, early_stopping_patience, early_stopping_threshold=0.0, min_epochs=1):
            super().__init__(early_stopping_patience, early_stopping_threshold)
            self.min_epochs = min_epochs

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            """Only apply early stopping after min_epochs completed"""
            # Check if we've completed minimum epochs
            if state.epoch < self.min_epochs:
                logger.info(f"⏸️  Early stopping suspended - epoch {state.epoch:.2f}/{self.min_epochs} (minimum required)")
                return control

            # After min_epochs, apply normal early stopping logic
            return super().on_evaluate(args, state, control, metrics=metrics, **kwargs)

    # ========================================================================
    # Callbacks
    # ========================================================================
    callbacks = [CheckpointDebugCallback()]

    # Rotating eval subset callback (if needed)
    if max_eval_samples is not None and max_eval_samples > 0 and len(full_eval_dataset) > max_eval_samples:
        rotating_callback = RotatingEvalSubsetCallback(full_eval_dataset, max_eval_samples)
        callbacks.append(rotating_callback)
        logger.info(f"Rotating eval subset callback enabled")

    # Early stopping callback with minimum epochs
    if config['early_stopping']['enabled']:
        min_epochs = config['early_stopping'].get('min_epochs', 1)
        early_stopping = DelayedEarlyStoppingCallback(
            early_stopping_patience=config['early_stopping']['patience'],
            early_stopping_threshold=config['early_stopping']['threshold'],
            min_epochs=min_epochs
        )
        callbacks.append(early_stopping)
        logger.info(f"Early stopping enabled with patience={config['early_stopping']['patience']}, min_epochs={min_epochs}")

    # ========================================================================
    # Initialize Trainer (WeightedLossTrainer for AutoModel)
    # ========================================================================
    if ENABLE_CLASS_WEIGHTS and class_weights_tensor is not None:
        logger.info("Using WeightedLossTrainer with class weights")
        trainer = WeightedLossTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            class_weights=class_weights_tensor,  # Pass weights to custom trainer
        )
    else:
        logger.info("Using standard Trainer (no class weights)")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
        )

    logger.info("Trainer initialized successfully")

    # Assign trainer reference to RotatingEvalSubsetCallback if it was created
    for callback in callbacks:
        if isinstance(callback, RotatingEvalSubsetCallback):
            callback.trainer = trainer
            logger.info("Trainer reference assigned to RotatingEvalSubsetCallback")
            break

    # ========================================================================
    # GPU Verification (After Trainer Initialization)
    # ========================================================================
    # Note: The Trainer automatically moves the model to GPU during initialization.
    # This verification confirms the model is on GPU before training starts.
    logger.info("\n" + "=" * 80)
    logger.info("GPU VERIFICATION")
    logger.info("=" * 80)
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"Number of GPUs detected: {torch.cuda.device_count()}")
        logger.info(f"Current GPU device: {torch.cuda.current_device()}")
        logger.info(f"GPU name: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
        logger.info(f"GPU memory reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")

        # Verify that model is on GPU (after Trainer has moved it)
        model_device = next(trainer.model.parameters()).device
        logger.info(f"Model device: {model_device}")

        if model_device.type == 'cuda':
            logger.info("✅ Model is on GPU - training will be fast")
            logger.info(f"   Expected speed: ~1.0-1.5 s/iteration for this configuration")
        else:
            logger.warning("⚠️  WARNING: Model is NOT on GPU!")
            logger.warning("   Training will be extremely slow on CPU (10-30+ s/iteration)")
            logger.warning("   Check TrainingArguments and device placement")
    else:
        logger.error("❌ CUDA NOT AVAILABLE - Training will be VERY slow on CPU!")
        logger.error("   Check: nvidia-smi, CUDA installation, PyTorch GPU version")
    logger.info("=" * 80 + "\n")

    # ========================================================================
    # Training
    # ========================================================================
    logger.info("=" * 80)
    logger.info("Starting training...")
    logger.info("=" * 80)

    # Print training info
    total_steps = len(train_dataset) // (
        training_args.per_device_train_batch_size
        * training_args.gradient_accumulation_steps
        * config['hardware']['num_gpus']
    ) * training_args.num_train_epochs

    logger.info(f"Number of training samples: {len(train_dataset)}")
    logger.info(f"Number of validation samples: {len(eval_dataset)}")
    logger.info(f"Number of epochs: {training_args.num_train_epochs}")
    logger.info(f"Batch size per device: {training_args.per_device_train_batch_size}")
    logger.info(f"Total batch size: {training_args.per_device_train_batch_size * config['hardware']['num_gpus']}")
    logger.info(f"Gradient accumulation steps: {training_args.gradient_accumulation_steps}")
    logger.info(f"Total optimization steps: {total_steps}")
    logger.info(f"Learning rate: {training_args.learning_rate}")
    logger.info(f"Warmup steps: {training_args.warmup_steps}")

    # Check if resuming from checkpoint
    resume_from = config['advanced'].get('resume_from_checkpoint')

    # Train
    try:
        train_result = trainer.train(resume_from_checkpoint=resume_from)

        # Save final model using HuggingFace standard method
        logger.info(f"Saving final model to {output_dir}")

        # Save model using save_pretrained (HuggingFace standard)
        trainer.model.save_pretrained(output_dir)
        logger.info(f"✅ Model saved to {output_dir}")

        # Save tokenizer
        tokenizer.save_pretrained(output_dir)
        logger.info(f"✅ Tokenizer saved to {output_dir}")

        # Save training metrics
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        logger.info("=" * 80)
        logger.info("Training completed successfully!")
        logger.info("=" * 80)

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        logger.info("Saving current model state...")
        trainer.save_model()
        logger.info("Model saved")
        sys.exit(0)

    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise

    # ========================================================================
    # Final Evaluation on Validation Set
    # ========================================================================
    logger.info("Running final evaluation on validation set...")
    eval_result = trainer.evaluate()

    logger.info("Evaluation results:")
    for key, value in eval_result.items():
        logger.info(f"  {key}: {value:.4f}")

    trainer.log_metrics("eval", eval_result)
    trainer.save_metrics("eval", eval_result)

    # ========================================================================
    # Summary
    # ========================================================================
    logger.info("=" * 80)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Model: {config['model']['name_or_path']}")
    logger.info(f"Task: Token Classification (TE Detection)")
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(eval_dataset)}")
    logger.info(f"Best checkpoint: {trainer.state.best_model_checkpoint}")
    if trainer.state.best_metric is not None:
        logger.info(f"Best metric ({training_args.metric_for_best_model}): {trainer.state.best_metric:.4f}")
    else:
        logger.info(f"Best metric ({training_args.metric_for_best_model}): Not available")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 80)

    logger.info("\nTo view training logs with TensorBoard, run:")
    logger.info(f"  tensorboard --logdir {config['output']['logging_dir']}")
    logger.info("\nTo evaluate on test set, run:")
    logger.info(f"  python evaluate_model.py --model_path {output_dir} --config {args.config}")

    return trainer


if __name__ == "__main__":
    main()
