from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Sequence

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from transformers import get_linear_schedule_with_warmup

from .data import (
    Concept,
    align_msa_sca,
    asjp_symbols,
    bcubed,
    initialize_data_resources,
    read_concepts,
    split_concepts,
    upgma_clusters,
)
from .data_collator import CogTranDataCollator
from .models import COGDETECT_VOCAB_SIZE, CogTranTriAttn, MSAConfig
from .pretrained import load_cogtran_pretrained, pretrained_config
from .tokenizer import CogTranTokenizer

DEFAULT_PREDICTION_THRESHOLD = 0.4


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.description = "Train and evaluate CogTran2's CogTranTriAttn model."
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--split", type=int, default=0, choices=range(5))
    parser.add_argument("--train-proportion", choices=("125", "90", "50", "0"), default="125")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Checkpoint directory (defaults to a proportion/split/input-variant-specific "
            "path under runs/triattn)"
        ),
    )
    parser.add_argument("--pretrained", help="Local checkpoint path or Hugging Face model repository")
    parser.add_argument("--revision", help="Hugging Face model revision for --pretrained")
    parser.add_argument(
        "--language-tokens",
        action="store_true",
        help="Prefix each form with CogDetect's [family;language] token",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-concepts", type=int, help="Limit concepts per directory for smoke tests")
    parser.add_argument("--max-examples", type=int, help="Limit concepts per split for smoke tests")
    parser.add_argument(
        "--msa-workers",
        type=int,
        default=1,
        help="Separate CPU processes used to align distinct MSAs before training",
    )
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-rows", type=int, default=256)
    parser.add_argument("--hidden-size", type=int)
    parser.add_argument("--layers", type=int)
    parser.add_argument("--heads", type=int)
    parser.add_argument("--feedforward-size", type=int)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--local-rank", "--local_rank", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_PREDICTION_THRESHOLD,
        help="UPGMA average similarity cut",
    )


def _read_pretrained_config(checkpoint: str | None, revision: str | None) -> dict:
    if checkpoint is None:
        return {}
    return pretrained_config(checkpoint, revision=revision)


def _load_dataset(args: argparse.Namespace) -> tuple[list[Concept], list[Concept]]:
    data_dir: Path = args.data_dir
    train_concepts = read_concepts(data_dir / "train", args.max_concepts)
    if args.train_proportion == "0":
        test = read_concepts(data_dir / "test1", args.max_concepts)
        test.extend(read_concepts(data_dir / "test2", args.max_concepts))
        return train_concepts, test
    split_root = data_dir / f"prop_{args.train_proportion}" / f"split_{args.split}"
    train_concepts.extend(read_concepts(split_root / "train", args.max_concepts))
    return train_concepts, read_concepts(split_root / "test", args.max_concepts)


def _output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir
    directory = Path("runs/triattn") / f"prop_{args.train_proportion}"
    if args.train_proportion != "0":
        directory /= f"split_{args.split}"
    directory /= "language_tokens" if args.language_tokens else "no_language_tokens"
    return directory


def _make_config(args: argparse.Namespace, tokenizer: CogTranTokenizer) -> MSAConfig:
    checkpoint_config = _read_pretrained_config(args.pretrained, args.revision)
    for argument, config_key in (
        ("hidden_size", "hidden_size"),
        ("heads", "num_attention_heads"),
        ("feedforward_size", "intermediate_size"),
    ):
        requested = getattr(args, argument)
        pretrained_value = checkpoint_config.get(config_key)
        if requested is not None and pretrained_value is not None and requested != pretrained_value:
            raise ValueError(
                f"--{argument.replace('_', '-')}={requested} does not match the pretrained checkpoint "
                f"({pretrained_value})"
            )
    pretrained_layers = checkpoint_config.get("num_hidden_layers")
    if args.layers is not None and pretrained_layers is not None and args.layers > pretrained_layers:
        raise ValueError("--layers cannot exceed the number of layers in the pretrained checkpoint")
    vocab_size = max(
        COGDETECT_VOCAB_SIZE,
        len(tokenizer),
        checkpoint_config.get("vocab_size") or 0,
    )
    return MSAConfig(
        vocab_size=vocab_size,
        hidden_size=(
            args.hidden_size
            if args.hidden_size is not None
            else checkpoint_config.get("hidden_size", 128)
        ),
        num_hidden_layers=(
            args.layers
            if args.layers is not None
            else checkpoint_config.get("num_hidden_layers", 2)
        ),
        num_attention_heads=(
            args.heads
            if args.heads is not None
            else checkpoint_config.get("num_attention_heads", 2)
        ),
        intermediate_size=(
            args.feedforward_size
            if args.feedforward_size is not None
            else checkpoint_config.get("intermediate_size", 128)
        ),
        dropout=args.dropout,
        max_rows=args.max_rows,
        max_columns=args.max_length,
        use_language_tokens=args.language_tokens,
    )


def _select_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected = torch.device(device)
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA was requested but is unavailable: {device}")
    return selected


def _feature_for_concept(
    concept: Concept,
    tokenizer: CogTranTokenizer,
    max_length: int,
    use_language_tokens: bool = False,
    aligned_forms: Sequence[Sequence[str]] | None = None,
) -> dict:
    aligned = aligned_forms if aligned_forms is not None else align_msa_sca(concept.forms)
    if use_language_tokens:
        if len(concept.languages) != len(aligned) or any(
            not language for language in concept.languages
        ):
            raise ValueError(
                f"Concept {concept.concept!r} in {concept.family!r} lacks DOCULECT values "
                "required by --language-tokens"
            )
        aligned = [
            [f"[{concept.family};{language}]", *form]
            for language, form in zip(concept.languages, aligned)
        ]
    return {
        "input_ids": [tokenizer.encode_form(form, max_length) for form in aligned],
        "labels": [
            [int(left == right) for right in concept.cognate_ids]
            for left in concept.cognate_ids
        ],
    }


def _align_forms_worker(forms: tuple[tuple[str, ...], ...]) -> list[list[str]]:
    return align_msa_sca(forms)


def _precompute_alignments(
    concepts: Sequence[Concept], workers: int
) -> dict[tuple[tuple[str, ...], ...], list[list[str]]]:
    if workers < 1:
        raise ValueError("msa_workers must be positive")
    unique_forms = list(dict.fromkeys(concept.forms for concept in concepts))
    if workers == 1 or len(unique_forms) < 2:
        return {forms: align_msa_sca(forms) for forms in unique_forms}

    chunksize = max(1, len(unique_forms) // (workers * 4))
    with ProcessPoolExecutor(
        max_workers=min(workers, len(unique_forms)),
        mp_context=multiprocessing.get_context("spawn"),
    ) as executor:
        aligned = executor.map(_align_forms_worker, unique_forms, chunksize=chunksize)
        return dict(zip(unique_forms, aligned))


def _training_alignment_tokens(
    concepts: Sequence[Concept],
    aligned_by_forms: dict[tuple[tuple[str, ...], ...], list[list[str]]] | None = None,
) -> set[str]:
    return {
        token
        for concept in concepts
        for aligned_form in (
            aligned_by_forms[concept.forms]
            if aligned_by_forms is not None
            else align_msa_sca(concept.forms)
        )
        for token in aligned_form
    }


def _loader(
    concepts: Sequence[Concept],
    tokenizer: CogTranTokenizer,
    collator: CogTranDataCollator,
    args: argparse.Namespace,
    shuffle: bool,
    rank: int = 0,
    world_size: int = 1,
    aligned_by_forms: dict[tuple[tuple[str, ...], ...], list[list[str]]] | None = None,
) -> DataLoader:
    examples = [
        _feature_for_concept(
            concept,
            tokenizer,
            args.max_length,
            args.language_tokens,
            aligned_by_forms[concept.forms] if aligned_by_forms is not None else None,
        )
        for concept in concepts
        if len(concept.forms) > 1
    ]
    if args.max_examples:
        examples = examples[: args.max_examples]
    if not examples:
        raise ValueError("No concepts with multiple forms were found")
    sampler = (
        DistributedSampler(
            examples,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            seed=args.seed,
        )
        if world_size > 1
        else None
    )
    return DataLoader(
        examples,
        batch_size=args.batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        collate_fn=collator,
    )


def _run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> float:
    training = optimizer is not None
    model.train(training)
    loss_total = 0.0
    examples_seen = 0
    for batch in loader:
        batch = {name: tensor.to(device) for name, tensor in batch.items()}
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            output = model(**batch)
            if output.loss is None:
                raise RuntimeError("Training batch did not produce a loss")
            if optimizer is not None:
                output.loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
        batch_size = batch["input_ids"].shape[0]
        loss_total += float(output.loss.detach()) * batch_size
        examples_seen += batch_size
    if dist.is_initialized():
        totals = torch.tensor([loss_total, examples_seen], dtype=torch.float64, device=device)
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        loss_total, examples_seen = totals.tolist()
    return loss_total / max(examples_seen, 1)


def _create_optimizer_and_scheduler(
    model: torch.nn.Module, learning_rate: float, total_steps: int
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    if total_steps < 1:
        raise ValueError("total_steps must be positive")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps,
    )
    return optimizer, scheduler


def _predict_matrix(
    model: torch.nn.Module,
    concept: Concept,
    tokenizer: CogTranTokenizer,
    collator: CogTranDataCollator,
    args: argparse.Namespace,
    device: torch.device,
    aligned_by_forms: dict[tuple[tuple[str, ...], ...], list[list[str]]] | None = None,
) -> list[list[float]]:
    feature = _feature_for_concept(
        concept,
        tokenizer,
        args.max_length,
        args.language_tokens,
        aligned_by_forms[concept.forms] if aligned_by_forms is not None else None,
    )
    batch = {name: tensor.to(device) for name, tensor in collator([feature]).items()}
    model.eval()
    with torch.no_grad():
        output = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        if output.logits is None:
            raise RuntimeError("Model did not return cognate-link logits")
        probabilities = output.logits[0].softmax(dim=-1)[..., 1]
    count = len(concept.forms)
    return probabilities[:count, :count].cpu().tolist()


def _evaluate(
    model: torch.nn.Module,
    concepts: Sequence[Concept],
    tokenizer: CogTranTokenizer,
    collator: CogTranDataCollator,
    args: argparse.Namespace,
    device: torch.device,
    aligned_by_forms: dict[tuple[tuple[str, ...], ...], list[list[str]]] | None = None,
) -> dict[str, float]:
    predicted_by_family: dict[str, list[int]] = defaultdict(list)
    gold_by_family: dict[str, list[int]] = defaultdict(list)
    predicted_offset: dict[str, int] = defaultdict(int)
    gold_offset: dict[str, int] = defaultdict(int)

    for concept in concepts:
        if len(concept.forms) < 2:
            continue
        similarities = _predict_matrix(
            model, concept, tokenizer, collator, args, device, aligned_by_forms
        )
        predicted = upgma_clusters(similarities, args.threshold)
        family = concept.family
        predicted_by_family[family].extend(label + predicted_offset[family] for label in predicted)
        predicted_offset[family] += max(predicted, default=-1) + 1
        concept_labels: dict[str, int] = {}
        for cognate_id in concept.cognate_ids:
            if cognate_id not in concept_labels:
                concept_labels[cognate_id] = gold_offset[family] + len(concept_labels)
        gold_by_family[family].extend(concept_labels[label] for label in concept.cognate_ids)
        gold_offset[family] += len(concept_labels)

    if not predicted_by_family:
        raise ValueError("No test concepts with at least two forms were found")
    family_metrics = {
        family: bcubed(predicted_by_family[family], gold_by_family[family])
        for family in predicted_by_family
    }
    results = {family: metrics[2] for family, metrics in family_metrics.items()}
    for index, name in enumerate(("BCPrec", "BCRec", "BCF")):
        results[name] = sum(metrics[index] for metrics in family_metrics.values()) / len(family_metrics)
    return results


def _train(
    args: argparse.Namespace,
    device: torch.device,
    rank: int,
    world_size: int,
) -> dict[str, float]:
    if args.epochs < 1 or args.batch_size < 1 or args.learning_rate <= 0:
        raise ValueError("epochs and batch_size must be positive, and learning_rate must be positive")
    if args.msa_workers < 1:
        raise ValueError("msa_workers must be positive")
    if args.max_concepts is not None and args.max_concepts < 1:
        raise ValueError("max_concepts must be positive")
    if args.max_examples is not None and args.max_examples < 1:
        raise ValueError("max_examples must be positive")
    args.output_dir = _output_dir(args)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    concepts, test_concepts = _load_dataset(args)
    concepts = [concept for concept in concepts if len(concept.forms) > 1]
    aligned_by_forms = _precompute_alignments((*concepts, *test_concepts), args.msa_workers)
    train_concepts, validation_concepts = split_concepts(concepts, args.validation_fraction, args.seed)
    if args.max_examples:
        train_concepts = train_concepts[: max(2, args.max_examples)]
        validation_concepts = validation_concepts[: max(1, args.max_examples // 4)]
    language_tokens = set()
    if args.language_tokens:
        for concept in (*train_concepts, *validation_concepts, *test_concepts):
            if len(concept.languages) != len(concept.forms) or any(not language for language in concept.languages):
                raise ValueError(
                    f"Concept {concept.concept!r} in {concept.family!r} lacks DOCULECT values "
                    "required by --language-tokens"
                )
            language_tokens.update(f"[{concept.family};{language}]" for language in concept.languages)
        tokenizer_tokens = (*asjp_symbols(), *language_tokens)
    else:
        tokenizer_tokens = _training_alignment_tokens(concepts, aligned_by_forms)
    tokenizer = CogTranTokenizer(tokenizer_tokens)
    config = _make_config(args, tokenizer)
    model = CogTranTriAttn(config)
    if args.pretrained:
        loaded = load_cogtran_pretrained(
            model, args.pretrained, token_to_id=tokenizer.token_to_id, revision=args.revision
        )
        if rank == 0:
            print(f"Initialized encoder from {args.pretrained} ({loaded} tensors)")
    model.to(device)
    if world_size > 1:
        model = DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            output_device=device.index if device.type == "cuda" else None,
        )
    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    if device.type == "cuda":
        torch.cuda.manual_seed(args.seed + rank)

    collator = CogTranDataCollator(tokenizer, max_length=args.max_length, max_rows=args.max_rows)
    train_loader = _loader(
        train_concepts,
        tokenizer,
        collator,
        args,
        shuffle=True,
        rank=rank,
        world_size=world_size,
        aligned_by_forms=aligned_by_forms,
    )
    validation_loader = _loader(
        validation_concepts,
        tokenizer,
        collator,
        args,
        shuffle=False,
        rank=rank,
        world_size=world_size,
        aligned_by_forms=aligned_by_forms,
    )
    optimizer, scheduler = _create_optimizer_and_scheduler(
        model, args.learning_rate, len(train_loader) * args.epochs
    )
    for epoch in range(1, args.epochs + 1):
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)
        train_loss = _run_epoch(model, train_loader, optimizer, device, scheduler)
        validation_loss = _run_epoch(model, validation_loader, None, device)
        if rank == 0:
            print(f"epoch={epoch} train_loss={train_loss:.4f} validation_loss={validation_loss:.4f}")
            args.output_dir.mkdir(parents=True, exist_ok=True)
            unwrapped_model = model.module if isinstance(model, DistributedDataParallel) else model
            unwrapped_model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
        if dist.is_initialized():
            dist.barrier()

    if rank != 0:
        return {}
    unwrapped_model = model.module if isinstance(model, DistributedDataParallel) else model
    metrics = _evaluate(
        unwrapped_model,
        test_concepts,
        tokenizer,
        collator,
        args,
        device,
        aligned_by_forms,
    )
    print("test " + " ".join(f"{name}={value:.4f}" for name, value in sorted(metrics.items())))
    with (args.output_dir / "test_metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2)
    return metrics


def train(args: argparse.Namespace) -> dict[str, float]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank or 0))
    if world_size > 1:
        if args.device == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = _select_device(args.device)
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("torchrun requested CUDA but CUDA is unavailable")
            device = torch.device("cuda", local_rank)
            torch.cuda.set_device(device)
        dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")
        if rank == 0:
            initialize_data_resources()
        dist.barrier()
    else:
        device = _select_device(args.device)

    try:
        return _train(args, device, rank, world_size)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
