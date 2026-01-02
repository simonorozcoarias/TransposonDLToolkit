"""
Data Collator for Pre-tokenized DNABERT-2 Token Classification

This module provides a custom data collator for handling pre-tokenized
genomic sequences for token classification tasks. The collator properly
handles padding, attention masks, and label alignment for batch processing.

The dataset is already tokenized with DNABERT-2's BPE tokenizer, so we only
need to handle batching and padding.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch
from transformers import PreTrainedTokenizerBase


@dataclass
class DataCollatorForPreTokenizedTokenClassification:
    """
    Data collator for token classification with pre-tokenized sequences.

    This collator handles datasets where sequences are already tokenized
    (input_ids and labels are already computed). It performs:
    - Padding of input_ids, attention_mask, and labels to the same length
    - Conversion to PyTorch tensors
    - Proper handling of label padding with -100 (ignored in loss)

    Args:
        tokenizer: The tokenizer used for tokenization (needed for pad_token_id)
        padding: Padding strategy ('longest', 'max_length', or False)
        max_length: Maximum sequence length (if padding='max_length')
        pad_to_multiple_of: Pad to a multiple of this value
        label_pad_token_id: ID to use for padding labels (default: -100)
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of pre-tokenized samples.

        Args:
            features: List of dictionaries containing:
                - input_ids: List[int] - Token IDs
                - attention_mask: List[int] - Attention mask
                - labels: List[int] - Token labels
                - (optional) species: str - Species name
                - (optional) sequence_id: str - Sequence identifier

        Returns:
            Dictionary with batched tensors:
                - input_ids: torch.Tensor (batch_size, seq_len)
                - attention_mask: torch.Tensor (batch_size, seq_len)
                - labels: torch.Tensor (batch_size, seq_len)
        """

        # Extract all keys from the first feature
        label_name = "labels"

        # Separate labels from other features
        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None

        # Remove labels and non-tensor fields from features for padding
        # Keep only input_ids and attention_mask
        batch = {
            "input_ids": [feature["input_ids"] for feature in features],
            "attention_mask": [feature["attention_mask"] for feature in features]
        }

        # Determine the batch max length
        if self.padding == "max_length" and self.max_length is not None:
            max_len = self.max_length
        else:
            # Find the longest sequence in the batch
            max_len = max(len(ids) for ids in batch["input_ids"])

        # Pad to multiple of specified value
        if self.pad_to_multiple_of is not None:
            max_len = (
                (max_len + self.pad_to_multiple_of - 1)
                // self.pad_to_multiple_of
                * self.pad_to_multiple_of
            )

        # Pad input_ids and attention_mask
        padded_batch = {}

        for key in ["input_ids", "attention_mask"]:
            padded_sequences = []
            for sequence in batch[key]:
                # Calculate padding length
                padding_length = max_len - len(sequence)

                if key == "input_ids":
                    # Pad with tokenizer's pad_token_id
                    pad_value = self.tokenizer.pad_token_id
                else:  # attention_mask
                    # Pad with 0 (no attention)
                    pad_value = 0

                # Add padding to the right
                padded_seq = sequence + [pad_value] * padding_length
                padded_sequences.append(padded_seq)

            # Convert to tensor
            padded_batch[key] = torch.tensor(padded_sequences, dtype=torch.long)

        # Pad labels if present
        if labels is not None:
            padded_labels = []
            for label_seq in labels:
                padding_length = max_len - len(label_seq)
                # Pad labels with label_pad_token_id (-100 by default)
                # -100 is ignored by PyTorch's CrossEntropyLoss
                padded_label = label_seq + [self.label_pad_token_id] * padding_length
                padded_labels.append(padded_label)

            padded_batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)

        return padded_batch


def get_data_collator(tokenizer: PreTrainedTokenizerBase, max_length: int = 512) -> DataCollatorForPreTokenizedTokenClassification:
    """
    Factory function to create a data collator for pre-tokenized token classification.

    Args:
        tokenizer: The DNABERT-2 tokenizer
        max_length: Maximum sequence length (default: 512)

    Returns:
        DataCollatorForPreTokenizedTokenClassification instance
    """
    return DataCollatorForPreTokenizedTokenClassification(
        tokenizer=tokenizer,
        padding="max_length",  # Pad to longest in batch
        max_length=max_length,  # Don't enforce max_length (sequences already at correct length)
        pad_to_multiple_of=None,
        label_pad_token_id=-100
    )


# Example usage for testing
if __name__ == "__main__":
    from transformers import AutoTokenizer

    # Load DNABERT-2 tokenizer
    tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True)

    # Create collator
    collator = get_data_collator(tokenizer, max_length=512)

    # Create dummy batch with different lengths
    dummy_features = [
        {
            "input_ids": [1, 2, 3, 4, 5],
            "attention_mask": [1, 1, 1, 1, 1],
            "labels": [0, 1, 1, 0, 1],
        },
        {
            "input_ids": [1, 2, 3],
            "attention_mask": [1, 1, 1],
            "labels": [0, 0, 1],
        },
    ]

    # Collate batch
    batch = collator(dummy_features)

    print("Collated batch:")
    print(f"Input IDs shape: {batch['input_ids'].shape}")
    print(f"Attention mask shape: {batch['attention_mask'].shape}")
    print(f"Labels shape: {batch['labels'].shape}")
    print(f"\nInput IDs:\n{batch['input_ids']}")
    print(f"\nAttention mask:\n{batch['attention_mask']}")
    print(f"\nLabels:\n{batch['labels']}")
    print(f"\nPad token ID: {tokenizer.pad_token_id}")
