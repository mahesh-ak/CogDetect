from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class CogTranDataCollator:
    """Pad pre-tokenized MSA examples into Hugging Face-style model batches."""

    tokenizer: Any
    max_length: int | None = None
    max_rows: int | None = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features: list[dict[str, Any]], return_tensors: str | None = None) -> dict[str, torch.Tensor]:
        tensor_type = return_tensors or self.return_tensors
        if tensor_type != "pt":
            raise ValueError(f"Only PyTorch tensors are supported, got {tensor_type!r}")
        if not features:
            raise ValueError("Cannot collate an empty batch")

        arrays = [feature["input_ids"] for feature in features]
        row_counts = [len(array) for array in arrays]
        column_counts = [len(row) for array in arrays for row in array]
        max_rows = max(row_counts)
        max_columns = max(column_counts)
        if self.max_rows is not None and max_rows > self.max_rows:
            raise ValueError(f"Batch has {max_rows} forms, exceeding max_rows={self.max_rows}")
        if self.max_length is not None and max_columns > self.max_length:
            raise ValueError(f"Batch sequence length {max_columns} exceeds max_length={self.max_length}")

        batch_size = len(features)
        pad_token_id = self.tokenizer.pad_token_id
        input_ids = torch.full((batch_size, max_rows, max_columns), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_rows, max_columns), dtype=torch.long)
        labels = None
        if any("labels" in feature for feature in features):
            labels = torch.full(
                (batch_size, max_rows, max_rows), self.label_pad_token_id, dtype=torch.long
            )

        for batch_index, feature in enumerate(features):
            ids = torch.as_tensor(feature["input_ids"], dtype=torch.long)
            if ids.ndim != 2:
                raise ValueError("Each input_ids feature must have shape [forms, sequence_length]")
            rows, columns = ids.shape
            input_ids[batch_index, :rows, :columns] = ids
            attention_mask[batch_index, :rows, :columns] = ids.ne(pad_token_id)
            if labels is not None and "labels" in feature:
                target = torch.as_tensor(feature["labels"], dtype=torch.long)
                if tuple(target.shape) != (rows, rows):
                    raise ValueError("Each labels feature must be a square [forms, forms] matrix")
                labels[batch_index, :rows, :rows] = target

        batch = {"input_ids": input_ids, "attention_mask": attention_mask}
        if labels is not None:
            batch["labels"] = labels
        return batch
