from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from transformers import PreTrainedTokenizer


SPECIAL_TOKENS = ("[CLS]", "[SEP]", "[BOS]", "[MASK]", "[PAD]", "[RESERVED]", "[UNK]")
CLS_ID, SEP_ID, BOS_ID, MASK_ID, PAD_ID, RESERVED_ID, UNK_ID = range(7)
SCA_UNKNOWN = "~"


class CogTranTokenizer(PreTrainedTokenizer):
    """Hugging Face tokenizer for whitespace-delimited ASJP segments."""

    vocab_files_names = {"vocab_file": "vocab.json"}
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        tokens: Iterable[str] | None = None,
        vocab_file: str | None = None,
        **kwargs,
    ) -> None:
        if vocab_file is not None:
            with Path(vocab_file).open(encoding="utf-8") as stream:
                token_ids = json.load(stream)
            tokens = [token for token, _ in sorted(token_ids.items(), key=lambda item: item[1]) if token not in SPECIAL_TOKENS]
        elif tokens is None:
            from .data import asjp_symbols

            tokens = asjp_symbols()
        self.tokens = sorted(
            {
                token
                for token in (tokens or ())
                if token
                and token not in SPECIAL_TOKENS
                and token != SCA_UNKNOWN
            }
        )
        self.tokens.append(SCA_UNKNOWN)
        self.token_to_id = {token: index + len(SPECIAL_TOKENS) for index, token in enumerate(self.tokens)}
        self.token_to_id.update({token: index for index, token in enumerate(SPECIAL_TOKENS)})
        self.id_to_token = {index: token for token, index in self.token_to_id.items()}

        kwargs.setdefault("cls_token", "[CLS]")
        kwargs.setdefault("sep_token", "[SEP]")
        kwargs.setdefault("bos_token", "[BOS]")
        kwargs.setdefault("mask_token", "[MASK]")
        kwargs.setdefault("pad_token", "[PAD]")
        kwargs.setdefault("unk_token", "[UNK]")
        kwargs.setdefault("additional_special_tokens", ["[RESERVED]"])
        super().__init__(**kwargs)

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def get_vocab(self) -> dict[str, int]:
        return dict(self.token_to_id)

    def _tokenize(self, text: str) -> list[str]:
        return text.split()

    def _convert_token_to_id(self, token: str) -> int:
        return self.token_to_id.get(token, UNK_ID)

    def _convert_id_to_token(self, index: int) -> str:
        return self.id_to_token.get(index, "[UNK]")

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        return " ".join(tokens)

    def build_inputs_with_special_tokens(
        self, token_ids_0: list[int], token_ids_1: list[int] | None = None
    ) -> list[int]:
        result = [CLS_ID, *token_ids_0, SEP_ID]
        if token_ids_1 is not None:
            result.extend([*token_ids_1, SEP_ID])
        return result

    def get_special_tokens_mask(
        self,
        token_ids_0: list[int],
        token_ids_1: list[int] | None = None,
        already_has_special_tokens: bool = False,
    ) -> list[int]:
        if already_has_special_tokens:
            return [int(token_id in {CLS_ID, SEP_ID, BOS_ID, MASK_ID, PAD_ID}) for token_id in token_ids_0]
        mask = [1, *([0] * len(token_ids_0)), 1]
        if token_ids_1 is not None:
            mask.extend([*([0] * len(token_ids_1)), 1])
        return mask

    def create_token_type_ids_from_sequences(
        self, token_ids_0: list[int], token_ids_1: list[int] | None = None
    ) -> list[int]:
        first = [0] * (len(token_ids_0) + 2)
        if token_ids_1 is None:
            return first
        return [*first, *([1] * (len(token_ids_1) + 1))]

    def encode_form(self, tokens: Iterable[str], max_length: int | None = None) -> list[int]:
        form = list(tokens)
        if max_length is not None and len(form) + 2 > max_length:
            raise ValueError(
                f"Form length {len(form)} exceeds max_length={max_length}; increase --max-length to avoid truncation"
            )
        return self.build_inputs_with_special_tokens([self.convert_tokens_to_ids(token) for token in form])

    def save_vocabulary(self, save_directory: str, filename_prefix: str | None = None) -> tuple[str]:
        directory = Path(save_directory)
        directory.mkdir(parents=True, exist_ok=True)
        prefix = f"{filename_prefix}-" if filename_prefix else ""
        vocab_path = directory / f"{prefix}vocab.json"
        with vocab_path.open("w", encoding="utf-8") as stream:
            json.dump(self.token_to_id, stream, ensure_ascii=False, indent=2)
        return (str(vocab_path),)


ASJPTok = CogTranTokenizer
