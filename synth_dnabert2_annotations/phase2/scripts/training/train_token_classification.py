#!/usr/bin/env python3
"""
DNABERT-2 Token Classification Fine-tuning for Transposable Element Detection

This script fine-tunes DNABERT-2 on pre-tokenized genomic sequences for token-level
classification of transposable elements (TEs). It follows the official DNABERT-2
training recommendations with full fine-tuning approach.

Usage:
    python train_token_classification.py --config config.yaml
    python train_token_classification.py --config config.yaml --debug  # Test with small subset
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


# ========================================================================
# Focal Loss Implementation
# ========================================================================
class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance in token classification.

    From "Focal Loss for Dense Object Detection" (Lin et al., ICCV 2017)
    Adapted for token classification with class weights.

    Args:
        alpha: Class weights (list or tensor). For 75/25 split: [0.75, 0.25]
        gamma: Focusing parameter. Default 2.0 (from paper)
        ignore_index: Index to ignore in loss computation
    """
    def __init__(self, alpha=None, gamma=2.0, ignore_index=-100):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        """
        Args:
            inputs: (N, C) where N = batch_size * seq_len, C = num_classes
            targets: (N,) with class indices
        """
        # Standard cross entropy
        ce_loss = nn.functional.cross_entropy(
            inputs, targets, reduction='none', ignore_index=self.ignore_index
        )

        # Get probability of true class
        pt = torch.exp(-ce_loss)

        # Apply focal term: (1 - pt)^gamma
        focal_term = (1 - pt) ** self.gamma

        # Only compute mean over non-ignored indices
        mask = targets != self.ignore_index

        # Apply class weights if provided
        if self.alpha is not None:
            if isinstance(self.alpha, (list, tuple)):
                alpha_t = torch.tensor(self.alpha, device=inputs.device, dtype=inputs.dtype)
            else:
                alpha_t = self.alpha

            # Clamp targets to valid range for safe indexing
            # (ignore_index = -100 would cause IndexError)
            # We'll mask these out later, so clamping to 0 is safe
            targets_safe = targets.clamp(min=0)
            alpha_t = alpha_t[targets_safe]

            focal_loss = alpha_t * focal_term * ce_loss
        else:
            focal_loss = focal_term * ce_loss

        if mask.sum() > 0:
            return focal_loss[mask].mean()
        else:
            return focal_loss.sum() * 0.0  # Return 0 if all ignored


# ========================================================================
# Custom Token Classification Model (HuggingFace Compatible)
# ========================================================================
from transformers import PreTrainedModel, PretrainedConfig


class TEClassificationConfig(PretrainedConfig):
    """Configuration for TE Classification Model."""
    model_type = "te_classification"

    def __init__(
        self,
        base_model_name=None,
        num_labels=2,
        class_weights=None,
        use_focal_loss=False,
        focal_loss_gamma=2.0,
        focal_loss_alpha=None,
        classifier_dropout=None,
        initializer_range=0.02,
        hidden_dropout_prob=0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.base_model_name = base_model_name
        self.num_labels = num_labels
        self.class_weights = class_weights  # List of class weights for CE loss
        self.use_focal_loss = use_focal_loss  # Whether to use Focal Loss
        self.focal_loss_gamma = focal_loss_gamma  # Gamma parameter for Focal Loss
        self.focal_loss_alpha = focal_loss_alpha  # Alpha (class weights) for Focal Loss
        self.classifier_dropout = classifier_dropout  # Dropout for classifier layer
        self.initializer_range = initializer_range  # Std for weight initialization
        self.hidden_dropout_prob = hidden_dropout_prob  # Fallback dropout value


class TokenClassificationModel(PreTrainedModel):
    """Custom token classification model with DNABERT-2 base (HuggingFace compatible)."""
    config_class = TEClassificationConfig

    def __init__(self, config, base_model=None):
        super().__init__(config)
        self.num_labels = config.num_labels

        # If base_model is provided, use it; otherwise load from config
        if base_model is not None:
            self.bert = base_model
        else:
            from transformers import AutoModel
            self.bert = AutoModel.from_pretrained(
                config.base_model_name,
                trust_remote_code=True
            )

        # Configurable dropout (follows HuggingFace BertForTokenClassification pattern)
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, config.num_labels)

        # Initialize weights following HuggingFace conventions
        # This calls init_weights() which applies _init_weights() to all modules
        self.post_init()

    def _init_weights(self, module):
        """
        Initialize weights following HuggingFace BERT conventions.

        This method is called by post_init() → init_weights() for each module.
        It ensures the classifier layer is initialized with Normal(0, initializer_range)
        instead of PyTorch's default Kaiming Uniform initialization.

        For Focal Loss, initializes classifier bias to account for class imbalance:
        bias = log(minority_freq / majority_freq)

        References:
            - HuggingFace BertPreTrainedModel._init_weights()
            - https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py
            - Focal Loss paper: Initialize bias for rare class probability
        """
        if isinstance(module, nn.Linear):
            # Initialize weights with normal distribution (mean=0, std=initializer_range)
            # This follows BERT's original initialization scheme
            # Slightly different from TensorFlow which uses truncated_normal
            # See: https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                # Special initialization for classifier bias when using Focal Loss
                if hasattr(self, 'classifier') and module is self.classifier:
                    if self.config.use_focal_loss and hasattr(self.config, 'focal_loss_prior_prob'):
                        # Initialize bias to log(minority_prob / majority_prob)
                        # This accounts for class imbalance in initial predictions
                        # E.g., if 25% TE: bias = log(0.25/0.75) ≈ -1.099
                        prior_prob_te = self.config.focal_loss_prior_prob
                        bias_init = np.log(prior_prob_te / (1 - prior_prob_te))
                        module.bias.data.fill_(bias_init)
                        logger.info(f"Initialized classifier bias to {bias_init:.4f} for Focal Loss (TE prior={prior_prob_te:.4f})")
                    else:
                        module.bias.data.zero_()
                else:
                    module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            # Choose loss function based on configuration
            if self.config.use_focal_loss:
                # Use Focal Loss for better handling of class imbalance
                loss_fct = FocalLoss(
                    alpha=self.config.focal_loss_alpha,
                    gamma=self.config.focal_loss_gamma,
                    ignore_index=-100
                )
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            else:
                # Use standard Cross Entropy Loss with optional class weights
                if self.config.class_weights is not None:
                    weight = torch.FloatTensor(self.config.class_weights).to(logits.device)
                    loss_fct = nn.CrossEntropyLoss(weight=weight, ignore_index=-100)
                else:
                    loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
            attentions=outputs.attentions if hasattr(outputs, 'attentions') else None
        )

    def num_parameters(self, only_trainable=False, exclude_embeddings=True):
        """
        Count number of parameters in the model.

        Args:
            only_trainable: Whether to count only trainable parameters
            exclude_embeddings: Whether to exclude embedding parameters (for FLOPs calculation)

        Returns:
            Number of parameters
        """
        if exclude_embeddings:
            # This is used by Trainer for FLOPs calculation
            # For simplicity, we use the default behavior from PreTrainedModel
            return super().num_parameters(only_trainable=only_trainable, exclude_embeddings=exclude_embeddings)

        if only_trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in self.parameters())


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
    # Load Model (Custom Token Classification)
    # ========================================================================
    logger.info(f"Loading model from {config['model']['name_or_path']}")

    # Load base model without classification head
    base_model = AutoModel.from_pretrained(
        config['model']['name_or_path'],
        trust_remote_code=True,
    )

    # Check if we should freeze the base model (train only classifier head)
    freeze_base_model = config['training'].get('freeze_base_model', False)

    if freeze_base_model:
        # Freeze all base model parameters in one line
        base_model.requires_grad_(False)

        logger.info("=" * 80)
        logger.info("FREEZE BASE MODEL ENABLED")
        logger.info("=" * 80)
        logger.info("All base model parameters are FROZEN")
        logger.info("Only the classifier head will be trained")
        logger.info("This is a feature extraction approach (like a frozen CNN backbone)")
        logger.info("=" * 80)
    else:
        # Full fine-tuning approach - all base model parameters remain trainable
        logger.info("Using full fine-tuning - all base model parameters are trainable")

    # Create model configuration
    model_config = TEClassificationConfig(
        base_model_name=config['model']['name_or_path'],
        num_labels=config['model']['num_labels']
    )

    # Create model using the TokenClassificationModel class (HuggingFace compatible)
    model = TokenClassificationModel(model_config, base_model=base_model)

    # Resize token embeddings if needed
    if len(tokenizer) != base_model.config.vocab_size:
        base_model.resize_token_embeddings(len(tokenizer))
        logger.info(f"Resized token embeddings to {len(tokenizer)}")

    # ========================================================================
    # Model Parameters Summary
    # ========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("MODEL ARCHITECTURE SUMMARY")
    logger.info("=" * 80)

    # Count all parameters by iterating through model
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    logger.info(f"Model: {config['model']['name_or_path']}")
    logger.info(f"Task: Token Classification (2 classes)")
    logger.info("")
    logger.info(f"TOTAL PARAMETERS:          {total_params:>12,}")
    logger.info(f"  Trainable:               {trainable_params:>12,} ({100*trainable_params/total_params:.4f}%)")
    logger.info(f"  Frozen (non-trainable):  {frozen_params:>12,} ({100*frozen_params/total_params:.4f}%)")

    if freeze_base_model:
        logger.info("")
        logger.info("✅ FREEZE MODE ACTIVE: Base model frozen, only classifier head trainable")
        logger.info(f"   Memory savings: ~{100*frozen_params/total_params:.1f}% reduction in gradient memory")
        logger.info(f"   Speed improvement: Expected 2-3x faster per epoch")
    else:
        logger.info("")
        logger.info("✅ FULL FINE-TUNING MODE: All parameters trainable")

    logger.info("=" * 80)

    # ========================================================================
    # Configure Class Weights for Imbalanced Data
    # ========================================================================
    # Two methods available:
    #   1. DYNAMIC: Calculate from training dataset labels (sklearn method)
    #   2. STATIC: Use fixed ratio from config (nucleotide-level proxy)
    #
    # Dynamic method (recommended):
    #   - Reflects actual token-level distribution in training data
    #   - Uses sklearn.utils.class_weight.compute_class_weight
    #   - More accurate but requires iterating through dataset
    #
    # Static method (faster):
    #   - Source: results/all_species_coverage.csv
    #   - Imbalance ratio: background_bases / te_bases (nucleotide level)
    #   - Assumption: Nucleotide ratio ≈ Token ratio (BPE is content-agnostic)
    #   - INSTANT (no dataset loading)
    #
    # See docs/class_weights_calculation.md for full details
    # ========================================================================
    # Master switch: Check if ANY class weighting method is enabled
    use_dynamic_weights = config['training'].get('use_dynamic_class_weights', False)
    use_static_ratio = config['training'].get('imbalance_ratio') is not None
    use_focal_loss = config['training'].get('use_focal_loss', False)
    ENABLE_CLASS_WEIGHTS = use_dynamic_weights or use_static_ratio or use_focal_loss

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

            logger.info(f"\nStatic class weights:")
            logger.info(f"  Background (class 0): {class_weights_list[0]:.2f}")
            logger.info(f"  TE (class 1):         {class_weights_list[1]:.2f}")

        # Configure loss function based on settings
        if use_focal_loss:
            # Use Focal Loss
            model.config.use_focal_loss = True
            model.config.focal_loss_gamma = config['training'].get('focal_loss_gamma', 2.0)

            # Use sklearn balanced class weights for alpha (already calculated above)
            # These are more stable than raw frequency ratio
            # E.g., sklearn gives [0.67, 2.0] instead of [1.0, 3.0]
            focal_alpha = class_weights_list  # Already computed by sklearn
            model.config.focal_loss_alpha = focal_alpha

            # Calculate prior probability for bias initialization
            n_background = sum(1 for l in train_labels_flat if l == 0)
            n_te = sum(1 for l in train_labels_flat if l == 1)
            total = len(train_labels_flat)
            prior_prob_te = n_te / total
            model.config.focal_loss_prior_prob = prior_prob_te

            logger.info("\n✅ FOCAL LOSS configured")
            logger.info(f"   Gamma (focusing parameter): {model.config.focal_loss_gamma}")
            logger.info(f"   Alpha (sklearn balanced class weights):")
            logger.info(f"     Background (class 0): {focal_alpha[0]:.4f}")
            logger.info(f"     TE (class 1):         {focal_alpha[1]:.4f}")
            logger.info(f"   Class distribution:")
            logger.info(f"     Background: {n_background:,} ({100*n_background/total:.2f}%)")
            logger.info(f"     TE:         {n_te:,} ({100*n_te/total:.2f}%)")
            logger.info(f"   Imbalance ratio (bg/te): {n_background/n_te:.4f}")
            logger.info(f"   Prior probability TE: {prior_prob_te:.4f}")
            logger.info(f"   Classifier bias will be initialized to: {np.log(prior_prob_te/(1-prior_prob_te)):.4f}")
            logger.info("=" * 80)
        else:
            # Use standard Cross Entropy with class weights
            model.config.class_weights = class_weights_list
            model.config.use_focal_loss = False

            logger.info("\n✅ Class weights configured")
            logger.info("   Weights will be applied automatically in CrossEntropyLoss")
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
    # Custom Trainer with Discriminative Learning Rates
    # ========================================================================
    class DiscriminativeLRTrainer(Trainer):
        """
        Custom Trainer that supports discriminative learning rates.

        Applies different learning rates to:
        - Base model (pretrained): Lower LR
        - Classifier head (new): Higher LR

        This is essential when fine-tuning a pretrained model with a new classification head.
        """

        def __init__(self, base_lr=None, classifier_lr=None, *args, **kwargs):
            """
            Args:
                base_lr: Learning rate for pretrained base model layers
                classifier_lr: Learning rate for new classifier head
            """
            self.base_lr = base_lr
            self.classifier_lr = classifier_lr
            super().__init__(*args, **kwargs)

        def create_optimizer(self):
            """
            Setup the optimizer with discriminative learning rates.

            Overrides the default Trainer.create_optimizer() to apply different
            learning rates to base model vs classifier head.
            """
            if self.optimizer is None:
                # Only apply discriminative LR if both rates are specified
                if self.base_lr is not None and self.classifier_lr is not None:
                    logger.info("\n" + "=" * 80)
                    logger.info("DISCRIMINATIVE LEARNING RATES")
                    logger.info("=" * 80)

                    # Separate parameters into base model and classifier head
                    # Classifier head includes: dropout and classifier layers
                    classifier_params = []
                    base_params = []

                    for name, param in self.model.named_parameters():
                        if not param.requires_grad:
                            continue

                        # Classifier head: dropout + linear classifier
                        if 'classifier' in name or 'dropout' in name:
                            classifier_params.append(param)
                            logger.info(f"  Classifier head: {name} -> LR={self.classifier_lr}")
                        else:
                            # Base model (BERT/DNABERT-2)
                            base_params.append(param)

                    logger.info(f"\nParameter groups:")
                    logger.info(f"  Base model parameters: {len(base_params):,} layers")
                    logger.info(f"  Classifier parameters: {len(classifier_params):,} layers")
                    logger.info(f"  Base LR: {self.base_lr}")
                    logger.info(f"  Classifier LR: {self.classifier_lr}")
                    logger.info(f"  Ratio (classifier/base): {self.classifier_lr/self.base_lr:.1f}x")
                    logger.info("=" * 80 + "\n")

                    # Create parameter groups with different learning rates
                    optimizer_grouped_parameters = [
                        {
                            "params": base_params,
                            "lr": self.base_lr,
                            "weight_decay": self.args.weight_decay,
                        },
                        {
                            "params": classifier_params,
                            "lr": self.classifier_lr,
                            "weight_decay": 0.0,  # No weight decay for classifier (BERT standard)
                        },
                    ]

                    # Create optimizer (use same optimizer type as configured in TrainingArguments)
                    optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

                    # Remove 'lr' from optimizer_kwargs as we're specifying it per parameter group
                    optimizer_kwargs.pop('lr', None)

                    self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

                else:
                    # Fall back to default single LR behavior
                    logger.info("Using single learning rate (discriminative LR not configured)")
                    super().create_optimizer()

            return self.optimizer

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
    # Initialize Trainer (with discriminative learning rates if configured)
    # ========================================================================
    # Check if discriminative learning rates are enabled in config
    base_lr = config['training'].get('base_learning_rate')
    classifier_lr = config['training'].get('classifier_learning_rate')

    if base_lr is not None and classifier_lr is not None:
        # Use DiscriminativeLRTrainer for different LRs
        trainer = DiscriminativeLRTrainer(
            base_lr=base_lr,
            classifier_lr=classifier_lr,
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
        )
        logger.info(f"Using DiscriminativeLRTrainer (base_lr={base_lr}, classifier_lr={classifier_lr})")
    else:
        # Use standard Trainer with single LR
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
        logger.info("Using standard Trainer with single learning rate")

    logger.info("Trainer initialized")

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
