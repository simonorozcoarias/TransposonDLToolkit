# Training Configuration Files

This directory contains YAML configuration files for DNABERT-2 training.

## Configuration Files

### production.yaml

Production training configuration used for the thesis experiments.

**Key settings**:
- Dataset: 40 species combined
- Training: Feature extraction (frozen base model)
- Learning rate: 3e-4
- Epochs: 10
- Batch size: 24 per device
- Gradient accumulation: 3 steps
- Class weights: Dynamic (sklearn balanced)
- Early stopping: 10 evaluations patience

**Usage**:
```bash
python scripts/training/train_token_classification_automodel.py --config config/production.yaml
```

### debug.yaml

Debug configuration for quick testing and development.

**Characteristics**:
- Smaller dataset or subset
- Fewer epochs
- More frequent evaluation
- Useful for testing code changes

**Usage**:
```bash
python scripts/training/train_token_classification_automodel.py --config config/debug.yaml
```

## Configuration Structure

All YAML configs follow this structure:

```yaml
model:
  name_or_path: "zhihan1996/DNABERT-2-117M"
  num_labels: 2

data:
  dataset_path: "path/to/dataset"
  max_length: 512
  is_pretokenized: true

training:
  freeze_base_model: true/false
  use_dynamic_class_weights: true/false
  learning_rate: float
  num_train_epochs: int
  per_device_train_batch_size: int
  gradient_accumulation_steps: int
  eval_steps: int

early_stopping:
  enabled: true
  patience: int
```

## Creating Custom Configurations

To create a new configuration:

1. Copy `production.yaml` as a template
2. Modify parameters as needed
3. Save with descriptive name (e.g., `custom_experiment.yaml`)
4. Run training with `--config config/custom_experiment.yaml`

## Important Parameters

### Model Parameters

- `freeze_base_model`:
  - `true`: Feature extraction (faster, higher LR)
  - `false`: Full fine-tuning (slower, lower LR)

### Class Imbalance Handling

- `use_dynamic_class_weights: true`: Calculate from training data (sklearn balanced)
- `use_dynamic_class_weights: false`: Use static ratio
- `static_class_weight_ratio`: Weight for TE class when using static weights

### Learning Rate Guidelines

- **Frozen base** (feature extraction): 3e-4 to 5e-4
- **Full fine-tuning**: 2e-5 to 5e-5

### Batch Size and Gradient Accumulation

Effective batch size = `per_device_train_batch_size` × `gradient_accumulation_steps` × `num_gpus`

Example (production):
- Per device: 24
- Accumulation: 3
- GPUs: 1
- **Effective**: 72

Adjust based on GPU memory:
- 16GB GPU: per_device = 16-24
- 40GB GPU: per_device = 24-32

## Templates

For creating configs with placeholders for paths, see the template system in FASE 4 of the repository setup plan.

## Version Control

When modifying configs:
- Save original with `.backup` extension
- Document changes in comments
- Track hyperparameter experiments separately

## References

- HuggingFace Transformers TrainingArguments documentation
- DNABERT-2 paper for model-specific recommendations
- `../README.md` for complete workflow context
