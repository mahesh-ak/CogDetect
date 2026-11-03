#!/usr/bin/env python3
"""Rebuild model datasets and concept-disjoint proportion splits from the archive."""

import argparse
import csv
import logging
import random
from pathlib import Path
from typing import Dict, Iterable, List
from zipfile import ZipFile


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = REPOSITORY_ROOT / "fast-cognate-detection.zip"
DEFAULT_DATA_ROOT = REPOSITORY_ROOT / "data"
REQUIRED_COLUMNS = ("DOCULECT", "CONCEPT", "TOKENS", "COGID")
PROPORTIONS = ((125, 0.125, 5), (50, 0.5, 5), (90, 0.9, 1))

TRAINING_FILES = {
    "2_Training/data/BAI.csv": "D_training_Bai-110-09.tsv",
    "2_Training/data/IEL.csv": "D_training_IndoEuropean-207-20.tsv",
    "2_Training/data/JAP.csv": "D_training_Japanese-200-10.tsv",
    "2_Training/data/OUG.csv": "D_training_ObUgrian-110-21.tsv",
    "2_Training/data/PAN.csv": "D_training_Austronesian-210-20.tsv",
    "2_Training/data/SIN.csv": "D_training_Chinese-140-15.tsv",
}

TEST1_FILES = {
    f"3_Testing-BipSkip/test/{name}": name.replace(".tsv", ".csv")
    for name in (
        "D_test_Bahnaric-200-24.tsv",
        "D_test_Chinese-180-18.tsv",
        "D_test_Huon-140-14.tsv",
        "D_test_Romance-110-43.tsv",
        "D_test_Tujia-109-5.tsv",
        "D_test_Uralic-173-8.tsv",
    )
}

TEST2_FILES = {
    "3_Testing-BipSkip/test2/data-aa-58-200.csv": "D_test_AustroAsiatic-58-200.csv",
    "3_Testing-BipSkip/test2/data-an-45-210.csv": "D_test_Austronesian-45-210.csv",
    "3_Testing-BipSkip/test2/data-ie-42-208.csv": "D_test_IndoEuropean-42-208.csv",
    "3_Testing-BipSkip/test2/data-pn-67-183.csv": "D_test_PamaNyungan-67-183.csv",
    "3_Testing-BipSkip/test2/data-st-64-110.csv": "D_test_SinoTibetan-64-110.csv",
}

TRAINING_COLUMN_NAMES = {
    "MID": "ID",
    "Taxon": "DOCULECT",
    "Taxa": "DOCULECT",
    "Language": "DOCULECT",
    "Gloss": "CONCEPT",
    "concept": "CONCEPT",
    "Tokens": "TOKENS",
    "CogID": "COGID",
}
TEST2_COLUMN_NAMES = {
    "Taxon": "DOCULECT",
    "Gloss": "CONCEPT",
    "tokens": "TOKENS",
    "cogid": "COGID",
}
IPA_TOKENIZER = None


def tokenize_ipa(form: str) -> str:
    global IPA_TOKENIZER
    if IPA_TOKENIZER is None:
        previous_disable = logging.root.manager.disable
        logging.disable(logging.CRITICAL)
        try:
            from lingpy import ipa2tokens
        finally:
            logging.disable(previous_disable)
        IPA_TOKENIZER = ipa2tokens
    return " ".join(IPA_TOKENIZER(form))


def parse_table(lines: Iterable[str], source: str):
    rows = list(csv.reader(lines, delimiter="\t"))
    header_index = None
    for index, row in enumerate(rows):
        if not row:
            continue
        if row[0].startswith("#"):
            continue
        if (
            "DOCULECT" in row
            or "Taxon" in row
            or "Taxa" in row
            or ("Language" in row and "concept" in row)
        ):
            header_index = index
            break
    if header_index is None:
        raise ValueError(f"{source} has no recognizable table header")

    columns = rows[header_index]
    records = []
    for row_number, row in enumerate(rows[header_index + 1 :], start=header_index + 2):
        if not row or (len(row) == 1 and row[0].startswith("#")):
            continue
        if len(row) > len(columns):
            raise ValueError(
                f"{source}:{row_number} has {len(row)} fields; expected {len(columns)}"
            )
        row += [""] * (len(columns) - len(row))
        records.append(dict(zip(columns, row)))
    if not records:
        raise ValueError(f"{source} contains no data rows")
    return columns, records


def validate_rows(source: str, records: List[Dict[str, str]]) -> None:
    for column in REQUIRED_COLUMNS:
        if any(column not in row for row in records):
            raise ValueError(f"{source} is missing required column {column!r}")


def write_table(path: Path, columns: List[str], records: List[Dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output, delimiter="\t", lineterminator="\n")
        writer.writerow(columns)
        writer.writerows(
            [[record.get(column, "") for column in columns] for record in records]
        )


def rebuild_source_data(archive_path: Path, data_root: Path) -> List[Path]:
    split_sources = []
    with ZipFile(archive_path) as archive:
        members = set(archive.namelist())
        expected_members = set(TRAINING_FILES) | set(TEST1_FILES) | set(TEST2_FILES)
        missing_members = expected_members - members
        if missing_members:
            raise FileNotFoundError(
                f"{archive_path} is missing archive entries: {sorted(missing_members)}"
            )

        for member, output_name in TRAINING_FILES.items():
            lines = archive.read(member).decode("utf-8-sig").splitlines()
            _, records = parse_table(lines, member)
            renamed_records = [
                {
                    TRAINING_COLUMN_NAMES.get(name, name): value
                    for name, value in record.items()
                }
                for record in records
            ]
            for record in renamed_records:
                if not record.get("TOKENS", "").strip() and record.get("IPA", "").strip():
                    record["TOKENS"] = tokenize_ipa(record["IPA"])
            columns = ["ID", "DOCULECT", "CONCEPT", "IPA", "TOKENS", "COGID"]
            validate_rows(member, renamed_records)
            output_path = data_root / "train" / output_name
            write_table(output_path, columns, renamed_records)

        for member, output_name in TEST1_FILES.items():
            lines = archive.read(member).decode("utf-8-sig").splitlines()
            columns, records = parse_table(lines, member)
            validate_rows(member, records)
            output_path = data_root / "test1" / output_name
            write_table(output_path, columns, records)
            output_path.with_suffix(".tsv").unlink(missing_ok=True)
            split_sources.append(output_path)

        for member, output_name in TEST2_FILES.items():
            lines = archive.read(member).decode("utf-8-sig").splitlines()
            source_columns, source_records = parse_table(lines, member)
            columns = ["ID"] + [TEST2_COLUMN_NAMES.get(name, name) for name in source_columns]
            records = []
            for row_id, source_record in enumerate(source_records, start=1):
                record = {"ID": str(row_id)}
                record.update(
                    {
                        TEST2_COLUMN_NAMES.get(name, name): value
                        for name, value in source_record.items()
                    }
                )
                records.append(record)
            validate_rows(member, records)
            output_path = data_root / "test2" / output_name
            write_table(output_path, columns, records)
            split_sources.append(output_path)

    return split_sources


def generate_splits(source_files: List[Path], output_root: Path, seed: int) -> None:
    datasets = []
    for path in source_files:
        columns, records = parse_table(
            path.read_text(encoding="utf-8-sig").splitlines(), str(path)
        )
        validate_rows(path.name, records)
        valid = [
            record
            for record in records
            if all(record[column].strip() for column in REQUIRED_COLUMNS)
        ]
        if not valid:
            raise ValueError(f"{path} contains no rows with required values populated")
        datasets.append((path, columns, valid))

    for proportion_id, proportion, num_splits in PROPORTIONS:
        for split_id in range(num_splits):
            for source_path, columns, records in datasets:
                groups: Dict[str, List[Dict[str, str]]] = {}
                for record in records:
                    groups.setdefault(record["CONCEPT"], []).append(record)

                concepts = sorted(groups)
                random.Random(seed + split_id).shuffle(concepts)
                num_train_concepts = round(len(concepts) * proportion)
                train_concepts = set(concepts[:num_train_concepts])
                train_rows = [row for row in records if row["CONCEPT"] in train_concepts]
                test_rows = [row for row in records if row["CONCEPT"] not in train_concepts]

                stem = source_path.stem
                split_root = output_root / f"prop_{proportion_id}" / f"split_{split_id}"
                write_table(split_root / "train" / f"{stem}_train.csv", columns, train_rows)
                write_table(split_root / "test" / f"{stem}_test.csv", columns, test_rows)
        print(
            f"Generated prop_{proportion_id}: "
            f"{num_splits} split(s) from {len(datasets)} source files"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive",
        type=Path,
        default=DEFAULT_ARCHIVE,
        help=f"fast-cognate-detection.zip path (default: {DEFAULT_ARCHIVE})",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"directory to rebuild train/, test1/, test2/, and prop_* (default: {DEFAULT_DATA_ROOT})",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed (default: 42)")
    args = parser.parse_args()

    if not args.archive.is_file():
        raise FileNotFoundError(
            f"Dataset archive not found: {args.archive}. Download the Zenodo archive "
            "to this path or pass its location with --archive."
        )

    split_sources = rebuild_source_data(args.archive, args.data_root)
    generate_splits(split_sources, args.data_root, args.seed)
    print(f"Rebuilt source datasets and proportion splits under {args.data_root}")


if __name__ == "__main__":
    main()
