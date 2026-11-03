from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from transformers import PretrainedConfig, PreTrainedModel
from transformers.utils import ModelOutput

from .alphafold_pairwise import CogDetectPairwiseBlock
from .tokenizer import PAD_ID

COGDETECT_VOCAB_SIZE = 768


class MSAConfig(PretrainedConfig):
    model_type = "cogtran2"

    def __init__(
        self,
        vocab_size: int = COGDETECT_VOCAB_SIZE,
        hidden_size: int = 128,
        num_hidden_layers: int = 2,
        num_attention_heads: int = 2,
        intermediate_size: int = 128,
        dropout: float = 0.1,
        max_rows: int = 256,
        max_columns: int = 256,
        layer_norm_eps: float = 1e-12,
        initializer_range: float = 0.02,
        use_language_tokens: bool = False,
        **kwargs,
    ) -> None:
        kwargs.setdefault("pad_token_id", PAD_ID)
        kwargs.setdefault("num_labels", 2)
        super().__init__(**kwargs)
        if min(vocab_size, hidden_size, num_hidden_layers, num_attention_heads, intermediate_size) < 1:
            raise ValueError("model dimensions and vocab_size must be positive")
        if hidden_size % num_attention_heads:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if min(max_rows, max_columns) < 1:
            raise ValueError("max_rows and max_columns must be positive")
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.max_rows = max_rows
        self.max_columns = max_columns
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range
        self.use_language_tokens = use_language_tokens


CogTranConfig = MSAConfig


@dataclass
class CogTranOutput(ModelOutput):
    loss: Tensor | None = None
    logits: Tensor | None = None


class _AxialAttention(nn.Module):
    def __init__(self, config: MSAConfig, axis: str) -> None:
        super().__init__()
        self.axis = axis
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.dropout = config.dropout
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout_module = nn.Dropout(config.dropout)

    def forward(self, hidden: Tensor, attention_mask: Tensor) -> Tensor:
        batch, rows, columns, width = hidden.shape
        shape = (batch, rows, columns, self.num_heads, self.head_dim)
        query = self.q_proj(hidden).reshape(shape)
        key = self.k_proj(hidden).reshape(shape)
        value = self.v_proj(hidden).reshape(shape)

        if self.axis == "row":
            query = query * (self.head_dim * rows) ** -0.5
            query = query * attention_mask[..., None, None]
            query = query.permute(0, 1, 3, 2, 4)
            key = key.permute(0, 1, 3, 2, 4)
            value = value.permute(0, 1, 3, 2, 4)
            weights = torch.matmul(query, key.transpose(-1, -2)).sum(dim=1)
            column_mask = attention_mask[:, 0, :]
            weights = weights.masked_fill(~column_mask[:, None, None, :], -1e4)
            probabilities = self.dropout_module(weights.softmax(dim=-1))
            context = torch.matmul(probabilities[:, None], value)
            context = context.permute(0, 1, 3, 2, 4).reshape(batch, rows, columns, width)
        else:
            query = query * self.head_dim**-0.5
            query = query.permute(0, 2, 3, 1, 4)
            key = key.permute(0, 2, 3, 1, 4)
            value = value.permute(0, 2, 3, 1, 4)
            weights = torch.matmul(query, key.transpose(-1, -2))
            key_mask = attention_mask.permute(0, 2, 1)
            weights = weights.masked_fill(~key_mask[:, :, None, None, :], -1e4)
            probabilities = self.dropout_module(weights.softmax(dim=-1))
            context = torch.matmul(probabilities, value)
            context = context.permute(0, 3, 1, 2, 4).reshape(batch, rows, columns, width)
        return self.out_proj(context)


class _FeedForward(nn.Module):
    def __init__(self, config: MSAConfig) -> None:
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout_module = nn.Dropout(config.dropout)

    def forward(self, hidden: Tensor) -> Tensor:
        return self.fc2(self.dropout_module(F.gelu(self.fc1(hidden))))


class _NormalizedResidual(nn.Module):
    def __init__(self, layer: nn.Module, config: MSAConfig) -> None:
        super().__init__()
        self.layer = layer
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.dropout_module = nn.Dropout(config.dropout)

    def forward(self, hidden: Tensor, *args) -> Tensor:
        return hidden + self.dropout_module(self.layer(self.layer_norm(hidden), *args))


class _AxialTransformerLayer(nn.Module):
    def __init__(self, config: MSAConfig) -> None:
        super().__init__()
        self.row_self_attention = _NormalizedResidual(_AxialAttention(config, "row"), config)
        self.column_self_attention = _NormalizedResidual(_AxialAttention(config, "column"), config)
        self.feed_forward_layer = _NormalizedResidual(_FeedForward(config), config)

    def forward(self, hidden: Tensor, attention_mask: Tensor) -> Tensor:
        hidden = self.row_self_attention(hidden, attention_mask)
        hidden = self.column_self_attention(hidden, attention_mask)
        hidden = self.feed_forward_layer(hidden)
        return hidden


class _MSAEncoder(nn.Module):
    """Native PyTorch axial MSA encoder with fair-esm-compatible parameter names."""

    def __init__(self, config: MSAConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=PAD_ID)
        self.embed_positions = nn.Embedding(
            config.max_columns + PAD_ID + 1, config.hidden_size, padding_idx=PAD_ID
        )
        self.emb_layer_norm_before = nn.LayerNorm(config.hidden_size)
        self.emb_layer_norm_after = nn.LayerNorm(config.hidden_size)
        self.dropout_module = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(_AxialTransformerLayer(config) for _ in range(config.num_hidden_layers))

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        batch, rows, columns = input_ids.shape
        if rows > self.config.max_rows:
            raise ValueError(f"MSA has {rows} forms, exceeding max_rows={self.config.max_rows}")
        if columns > self.config.max_columns:
            raise ValueError(f"MSA length {columns} exceeds max_columns={self.config.max_columns}")
        positions = attention_mask.long().cumsum(dim=-1) + PAD_ID
        positions = positions.masked_fill(~attention_mask, PAD_ID)
        hidden = self.embed_tokens(input_ids) + self.embed_positions(positions)
        hidden = self.dropout_module(self.emb_layer_norm_before(hidden))
        hidden = hidden * attention_mask.unsqueeze(-1)
        for layer in self.layers:
            hidden = layer(hidden, attention_mask)
        return self.emb_layer_norm_after(hidden)

    def initialize(self, config: MSAConfig) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    nn.init.zeros_(module.bias)
                if isinstance(module, nn.Embedding) and module.padding_idx is not None:
                    with torch.no_grad():
                        module.weight[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)


class CogTranTriAttn(PreTrainedModel):
    config_class = MSAConfig
    base_model_prefix = "cogtran"
    main_input_name = "input_ids"

    def __init__(self, config: MSAConfig) -> None:
        super().__init__(config)
        self.msa = _MSAEncoder(config)
        self.outer_norm = nn.LayerNorm(config.hidden_size, eps=1e-5)
        self.left_projection = nn.Linear(config.hidden_size, config.intermediate_size)
        self.right_projection = nn.Linear(config.hidden_size, config.intermediate_size)
        self.outer_projection = nn.Linear(config.intermediate_size, config.hidden_size)
        self.pair_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.triangle_layers = nn.ModuleList(
            CogDetectPairwiseBlock(
                config.hidden_size,
                config.intermediate_size,
                config.num_attention_heads,
                config.layer_norm_eps,
            )
            for _ in range(config.num_hidden_layers)
        )
        self.pairwise_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.classifier = nn.Linear(config.hidden_size, 2)
        self.post_init()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
            if isinstance(module, nn.Embedding) and module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _outer_mean(self, hidden: Tensor, token_mask: Tensor, eps: float = 1e-12) -> Tensor:
        hidden = self.outer_norm(hidden).transpose(1, 2)
        valid = token_mask.transpose(1, 2).to(hidden.dtype)
        left = self.left_projection(hidden)
        right = self.right_projection(hidden)
        pair = torch.einsum("bcrh,bcsh->brsh", left, right)
        counts = torch.einsum("bcr,bcs->brs", valid, valid).unsqueeze(-1) + eps
        return self.outer_projection(pair / counts)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        labels: Tensor | None = None,
        return_dict: bool | None = None,
    ) -> CogTranOutput | tuple[Tensor, ...]:
        if input_ids.ndim != 3:
            raise ValueError("input_ids must have shape [batch, forms, aligned_length]")
        if attention_mask is None:
            attention_mask = input_ids.ne(self.config.pad_token_id)
        token_mask = attention_mask.bool()
        forms_mask = token_mask.any(dim=-1)
        hidden = self.msa(input_ids, token_mask)
        pair = self.pair_norm(self._outer_mean(hidden, token_mask))
        pair_mask = forms_mask[:, :, None] & forms_mask[:, None, :]
        for layer in self.triangle_layers:
            pair = layer(pair, pair_mask)
        pair = self.pairwise_norm(pair)
        logits = self.classifier(pair)
        loss = None if labels is None else F.cross_entropy(
            logits.reshape(-1, 2), labels.reshape(-1), ignore_index=-100
        )
        if return_dict is False:
            output = (logits,)
            return (loss, *output) if loss is not None else output
        return CogTranOutput(loss=loss, logits=logits)

    @classmethod
    def from_cogtran_pretrained(
        cls, config: MSAConfig, checkpoint: str, token_to_id: dict[str, int] | None = None, **kwargs
    ) -> "CogTranTriAttn":
        from .pretrained import load_cogtran_pretrained

        model = cls(config)
        load_cogtran_pretrained(model, checkpoint, token_to_id=token_to_id, **kwargs)
        return model


def load_cogtran_pretrained(model, checkpoint, token_to_id=None, **kwargs) -> int:
    """Load a compatible local or Hub CogTran masked-LM encoder into the MSA backbone."""
    from .pretrained import load_cogtran_pretrained as loader

    return loader(model, checkpoint, token_to_id=token_to_id, **kwargs)
