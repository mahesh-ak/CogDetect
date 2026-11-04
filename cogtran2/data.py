from __future__ import annotations

import csv
import logging
import random
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


def _asjp_converter() -> dict[str, str]:
    try:
        from lingpy import rc
    except ImportError as error:
        raise ImportError("Install the package dependencies with `pip install -e .` (requires lingpy).") from error
    previous_level = logging.root.manager.disable
    logging.disable(logging.INFO)
    try:
        return rc("asjp").converter
    finally:
        logging.disable(previous_level)


def asjp_symbols() -> tuple[str, ...]:
    """Return the fixed ASJP output inventory defined by LingPy's converter."""
    return tuple(sorted(set(_asjp_converter().values())))


def initialize_data_resources() -> None:
    """Initialize LingPy's ASJP converter before distributed workers start."""
    _asjp_converter()


def _to_asjp(tokens: Sequence[str], converter: dict[str, str]) -> tuple[str, ...]:
    converted = []

    def append(symbol: str) -> None:
        converted.append(symbol)

    for token in tokens:
        if token in {"+", "-", "="}:
            continue
        if token in converter:
            append(converter[token])
        elif len(token) > 1:
            for char in token:
                if unicodedata.category(char).startswith("M"):
                    continue
                if char in {"+", "-", "="}:
                    continue
                append(converter.get(char, char))
        elif unicodedata.category(token).startswith("M"):
            continue
        else:
            append(converter.get(token, token))
    return tuple(converted)


def _asjp_to_ipa(asjp: str, ipa_of: dict[str, str]) -> str:
    return "".join(ipa_of[char] for char in asjp.strip() if char in ipa_of)


@dataclass(frozen=True)
class Concept:
    family: str
    concept: str
    forms: tuple[tuple[str, ...], ...]
    cognate_ids: tuple[str, ...]
    languages: tuple[str, ...] = ()


def _family_from_filename(path: Path) -> str:
    parts = path.stem.split("_")
    return parts[2].split("-")[0] if len(parts) >= 3 else path.parent.name


def read_concepts(path: str | Path, limit: int | None = None) -> list[Concept]:
    """Read CogDetect split files and retain forms as raw IPA character sequences."""
    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {root}")
    files_by_stem: dict[str, Path] = {}
    for file in root.iterdir():
        if file.suffix.lower() not in {".csv", ".tsv"}:
            continue
        current = files_by_stem.get(file.stem)
        if current is None or file.suffix.lower() == ".tsv":
            files_by_stem[file.stem] = file
    concepts: list[Concept] = []
    for file in sorted(files_by_stem.values()):
        grouped: dict[str, list[tuple[tuple[str, ...], str, str]]] = {}
        with file.open(encoding="utf-8-sig", newline="") as stream:
            header = stream.readline()
            delimiter = "\t" if "\t" in header else ","
            stream.seek(0)
            reader = csv.DictReader(stream, delimiter=delimiter)
            required = {"TOKENS", "COGID", "CONCEPT"}
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                raise ValueError(f"{file} must contain columns {sorted(required)}")
            for row in reader:
                token_text = (row.get("TOKENS") or "").strip()
                cognate_id = (row.get("COGID") or "").strip()
                concept_name = (row.get("CONCEPT") or "").strip()
                language = (row.get("DOCULECT") or "").strip()
                if not token_text or token_text == "-" or not cognate_id or not concept_name:
                    continue
                tokens = tuple(token_text.replace(" ", ""))
                if tokens:
                    grouped.setdefault(concept_name, []).append((tokens, cognate_id, language))
        family = _family_from_filename(file)
        for name, entries in grouped.items():
            concepts.append(
                Concept(
                    family=family,
                    concept=name,
                    forms=tuple(tokens for tokens, _, _ in entries),
                    cognate_ids=tuple(cognate_id for _, cognate_id, _ in entries),
                    languages=tuple(language for _, _, language in entries),
                )
            )
            if limit is not None and len(concepts) >= limit:
                return concepts
    if not concepts:
        raise ValueError(f"No usable concept records found in {root}")
    return concepts


def collect_tokens(concepts: Iterable[Concept]) -> set[str]:
    return {token for concept in concepts for form in concept.forms for token in form}


def split_concepts(
    concepts: Sequence[Concept], validation_fraction: float, seed: int
) -> tuple[list[Concept], list[Concept]]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    if len(concepts) < 2:
        raise ValueError("At least two training concepts are required for a train/validation split")
    shuffled = list(concepts)
    random.Random(seed).shuffle(shuffled)
    validation_size = min(max(1, round(len(shuffled) * validation_fraction)), len(shuffled) - 1)
    return shuffled[validation_size:], shuffled[:validation_size]


def align_msa_sca(
    forms: Sequence[Sequence[str]], *, asjp: bool = False
) -> list[list[str]]:
    """Align IPA forms, or convert ASJP forms to representative IPA first."""
    if not forms:
        return []

    try:
        from lingpy import Multiple
    except ImportError as error:
        raise ImportError("Install the package dependencies with `pip install -e .` (requires lingpy).") from error

    converter = _asjp_converter()
    if asjp:
        ipa_of: dict[str, str] = {}
        for ipa, symbol in converter.items():
            ipa_of.setdefault(symbol, ipa)
        ipa_forms = [tuple(_asjp_to_ipa("".join(form), ipa_of)) for form in forms]
    else:
        ipa_forms = [tuple(form) for form in forms]
    cleaned_forms = [
        tuple(token for token in form if converter.get(token) != "!")
        for form in ipa_forms
    ]
    alignable_indices = [index for index, form in enumerate(cleaned_forms) if form]
    if not alignable_indices:
        return [["~"] for _ in forms]

    multiple = Multiple(
        ["".join(cleaned_forms[index]) for index in alignable_indices]
    )
    multiple.prog_align()
    aligned: list[list[str] | None] = [None] * len(forms)
    for index, form in zip(alignable_indices, multiple.alm_matrix):
        converted = []
        for symbol in form:
            if symbol in converter:
                converted_symbol = converter[symbol]
                converted.append("~" if converted_symbol == "!" else converted_symbol)
            elif len(symbol) > 1:
                composite = "".join(
                    converter[char]
                    for char in symbol
                    if char in converter and converter[char] != "!"
                )
                converted.append(composite if composite else "~")
            else:
                converted.append("~")
        aligned[index] = converted

    width = len(next(form for form in aligned if form is not None))
    return [form if form is not None else ["~"] * width for form in aligned]


def bcubed(
    predicted: Sequence[int], gold: Sequence[int]
) -> tuple[float, float, float]:
    if len(predicted) != len(gold):
        raise ValueError("Predicted and gold cluster assignments must have the same size")
    if not gold:
        return 0.0, 0.0, 0.0
    precision = recall = 0.0
    for index in range(len(gold)):
        predicted_size = sum(label == predicted[index] for label in predicted)
        gold_size = sum(label == gold[index] for label in gold)
        overlap = sum(
            predicted_label == predicted[index] and gold_label == gold[index]
            for predicted_label, gold_label in zip(predicted, gold)
        )
        precision += overlap / predicted_size
        recall += overlap / gold_size
    precision /= len(gold)
    recall /= len(gold)
    f_score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f_score


def upgma_clusters(similarities: Sequence[Sequence[float]], threshold: float = 0.4) -> list[int]:
    """Symmetrize link scores and cluster with LingPy UPGMA at a similarity cut."""
    size = len(similarities)
    if any(len(row) != size for row in similarities):
        raise ValueError("UPGMA similarities must be a square matrix")
    if size == 0:
        return []

    from lingpy.algorithm.clustering import flat_cluster

    distances = [
        [
            1.0 - (float(similarities[left][right]) + float(similarities[right][left])) / 2.0
            for right in range(size)
        ]
        for left in range(size)
    ]
    clusters = flat_cluster("upgma", 1.0 - threshold, distances)
    labels = [0] * size
    for label, indices in enumerate(clusters.values()):
        for index in indices:
            labels[index] = label
    return labels
