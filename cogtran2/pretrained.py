from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .tokenizer import SPECIAL_TOKENS


_WEIGHT_NAMES = ("model.safetensors", "pytorch_model.bin", "model.pt", "model.bin", "model.pth")


def _resolve_checkpoint(
    checkpoint: str | Path,
    revision: str | None = None,
    token: str | bool | None = None,
    cache_dir: str | Path | None = None,
) -> Path:
    local_path = Path(checkpoint).expanduser()
    if local_path.exists():
        return local_path
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise ImportError("Install transformers and huggingface_hub to load model repos from the Hub") from error
    return Path(
        snapshot_download(
            repo_id=str(checkpoint),
            revision=revision,
            token=token,
            cache_dir=cache_dir,
            allow_patterns=[
                "*.json",
                "*.safetensors",
                "*.bin",
                "*.pt",
                "*.pth",
            ],
        )
    )


def _weight_file(path: Path) -> Path:
    if path.is_file():
        return path
    for name in _WEIGHT_NAMES:
        candidate = path / name
        if candidate.is_file():
            return candidate
    checkpoints = sorted(
        (candidate for candidate in path.glob("checkpoint-*") if candidate.is_dir()),
        key=lambda item: int(re.search(r"(\d+)$", item.name).group(1)),  # type: ignore[union-attr]
    )
    for checkpoint in reversed(checkpoints):
        for name in _WEIGHT_NAMES:
            candidate = checkpoint / name
            if candidate.is_file():
                return candidate
    raise FileNotFoundError(f"No supported CogTran checkpoint weights found in {path}")


def _source_tokenizer(weight_path: Path) -> dict[str, int] | None:
    for directory in (weight_path.parent, *weight_path.parents):
        for name in ("tokenizer_config.json", "tokenizer.json", "vocab.json"):
            config_path = directory / name
            if not config_path.is_file():
                continue
            with config_path.open(encoding="utf-8") as stream:
                config = json.load(stream)
            tokens = config.get("chars", config.get("tokens"))
            if tokens is not None:
                vocab = {item: index + len(SPECIAL_TOKENS) for index, item in enumerate(tokens)}
                vocab.update({item: index for index, item in enumerate(SPECIAL_TOKENS)})
                return vocab
            vocab = config.get("model", {}).get("vocab", config.get("vocab"))
            if isinstance(vocab, dict):
                return {item: int(index) for item, index in vocab.items()}
    return None


def _load_state(path: Path) -> dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as error:
            raise ImportError("Install safetensors to load this checkpoint: pip install safetensors") from error
        state: Any = load_file(str(path), device="cpu")
    else:
        state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint {path} does not contain a state dictionary")
    return state


def _find_config(checkpoint: str | Path, **hub_kwargs) -> dict[str, Any]:
    path = Path(checkpoint).expanduser()
    if path.exists():
        directory = path if path.is_dir() else path.parent
        for parent in (directory, *directory.parents):
            config_path = parent / "config.json"
            if config_path.is_file():
                with config_path.open(encoding="utf-8") as stream:
                    return json.load(stream)
        return {}
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError

    try:
        config_path = hf_hub_download(repo_id=str(checkpoint), filename="config.json", **hub_kwargs)
    except EntryNotFoundError:
        return {}
    with Path(config_path).open(encoding="utf-8") as stream:
        return json.load(stream)


def pretrained_config(checkpoint: str | Path, **hub_kwargs) -> dict[str, Any]:
    """Read model dimensions from a local CogTran checkpoint or Hub repository."""
    return _find_config(checkpoint, **hub_kwargs)


def load_cogtran_pretrained(
    model: nn.Module,
    checkpoint: str | Path,
    token_to_id: dict[str, int] | None = None,
    revision: str | None = None,
    token: str | bool | None = None,
    cache_dir: str | Path | None = None,
) -> int:
    """Initialize the model's MSA encoder from a local or Hugging Face Hub checkpoint."""
    resolved = _resolve_checkpoint(checkpoint, revision, token, cache_dir)
    weight_path = _weight_file(resolved)
    state = _load_state(weight_path)
    encoder_target = model.msa.state_dict()
    source_vocab = _source_tokenizer(weight_path)
    compatible: dict[str, torch.Tensor] = {}
    loaded_layer_tensors = 0

    for original_key, value in state.items():
        if not isinstance(value, torch.Tensor):
            continue
        key = original_key.removeprefix("module.")
        while True:
            for prefix in ("base_model.", "model.", "msat.", "msa.", "encoder."):
                if key.startswith(prefix):
                    key = key[len(prefix) :]
                    break
            else:
                break
        if key.startswith(("lm_head.", "contact_head.")) or key not in encoder_target:
            continue
        expected = encoder_target[key]
        if key == "embed_tokens.weight" and value.ndim == 2 and value.shape[1] == expected.shape[1]:
            remapped = expected.clone()
            if source_vocab is not None and token_to_id is not None:
                for item, target_id in token_to_id.items():
                    source_id = source_vocab.get(item)
                    if source_id is not None and source_id < value.shape[0] and target_id < remapped.shape[0]:
                        remapped[target_id] = value[source_id]
            elif value.shape == expected.shape:
                remapped.copy_(value)
            compatible[key] = remapped
        elif (
            key == "embed_positions.weight"
            and value.ndim == 2
            and value.shape[1] == expected.shape[1]
        ):
            remapped = expected.clone()
            overlap = min(value.shape[0], remapped.shape[0])
            remapped[:overlap] = value[:overlap]
            compatible[key] = remapped
        elif value.shape == expected.shape:
            compatible[key] = value
        if key in compatible and key.startswith("layers."):
            loaded_layer_tensors += 1

    if not compatible or loaded_layer_tensors == 0:
        raise ValueError(
            f"No compatible axial MSA encoder weights found in {weight_path}. "
            "Check the checkpoint's hidden size, attention heads, and layer count."
        )
    model.msa.load_state_dict(compatible, strict=False)
    skipped = len(encoder_target) - len(compatible)
    if skipped:
        warnings.warn(f"Loaded {len(compatible)} encoder tensors; {skipped} kept their new initialization.")
    return len(compatible)


def load_pretrained(model: nn.Module, checkpoint: str | Path, **kwargs) -> int:
    """Hub/local convenience alias for loading a CognateTransformer MSA encoder."""
    return load_cogtran_pretrained(model, checkpoint, **kwargs)
