"""Adapted from CogDetect's vendored alphafold2-pytorch pairwise modules.

Copyright (c) 2020 Phil Wang, Eric Alcaide

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Source: https://github.com/mahesh-ak/CogDetect/tree/main/alphafold2
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class _Attention(nn.Module):
    def __init__(self, dim: int, heads: int, dim_head: int) -> None:
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.dim_head = dim_head
        self.scale = dim_head**-0.5
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        self.gating = nn.Linear(dim, inner_dim)
        self.dropout = nn.Dropout(0.0)

    def forward(
        self,
        x: Tensor,
        mask: Tensor,
        attn_bias: Tensor,
        tie_dim: int | None = None,
    ) -> Tensor:
        batch, positions, _ = x.shape
        query = self.to_q(x).reshape(batch, positions, self.heads, self.dim_head)
        key, value = self.to_kv(x).chunk(2, dim=-1)
        key = key.reshape(batch, positions, self.heads, self.dim_head)
        value = value.reshape(batch, positions, self.heads, self.dim_head)
        query = query.permute(0, 2, 1, 3) * self.scale
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)

        if tie_dim is None:
            scores = torch.matmul(query, key.transpose(-1, -2))
        else:
            query = query.reshape(-1, tie_dim, self.heads, positions, self.dim_head)
            key = key.reshape(-1, tie_dim, self.heads, positions, self.dim_head)
            query = query.mean(dim=1)
            scores = torch.einsum("bhid,brhjd->brhij", query, key)
            scores = scores.reshape(batch, self.heads, positions, positions)

        scores = scores + attn_bias
        pair_mask = mask[:, None, :, None] & mask[:, None, None, :]
        scores = scores.masked_fill(~pair_mask, -torch.finfo(scores.dtype).max)
        probabilities = self.dropout(scores.softmax(dim=-1))
        output = torch.matmul(probabilities, value).transpose(1, 2).reshape(
            batch, positions, self.heads * self.dim_head
        )
        output = output * self.gating(x).sigmoid()
        return self.to_out(output)


class _AxialAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        *,
        row_attn: bool,
        accept_edges: bool = True,
        global_query_attn: bool = False,
    ) -> None:
        super().__init__()
        self.row_attn = row_attn
        self.global_query_attn = global_query_attn
        self.norm = nn.LayerNorm(dim)
        self.attn = _Attention(dim, heads, dim_head)
        self.edges_to_attn_bias = nn.Linear(dim, heads, bias=False) if accept_edges else None
        self.heads = heads

    def forward(self, x: Tensor, edges: Tensor, mask: Tensor) -> Tensor:
        batch, rows, columns, dim = x.shape
        normalized = self.norm(x)
        if self.row_attn:
            axial_dim = rows
            folded = normalized.reshape(batch * rows, columns, dim)
            folded_mask = mask.reshape(batch * rows, columns)
        else:
            axial_dim = columns
            folded = normalized.transpose(1, 2).reshape(batch * columns, rows, dim)
            folded_mask = mask.transpose(1, 2).reshape(batch * columns, rows)

        if self.edges_to_attn_bias is None:
            bias = folded.new_zeros((batch * axial_dim, self.heads, rows, columns))
        else:
            bias = self.edges_to_attn_bias(edges).permute(0, 3, 1, 2)
            bias = bias.unsqueeze(1).expand(-1, axial_dim, -1, -1, -1)
            bias = bias.reshape(batch * axial_dim, self.heads, rows, columns)

        if self.row_attn:
            output = self.attn(folded, folded_mask, bias)
            return output.reshape(batch, rows, columns, dim)

        output = self.attn(
            folded,
            folded_mask,
            bias,
            tie_dim=axial_dim if self.global_query_attn else None,
        )
        return output.reshape(batch, columns, rows, dim).transpose(1, 2)


class _TriangleMultiplicative(nn.Module):
    def __init__(self, dim: int, mix: str) -> None:
        super().__init__()
        if mix not in {"outgoing", "ingoing"}:
            raise ValueError(f"Unsupported triangle multiplication direction: {mix}")
        self.mix = mix
        self.norm = nn.LayerNorm(dim)
        self.left_proj = nn.Linear(dim, dim)
        self.right_proj = nn.Linear(dim, dim)
        self.left_gate = nn.Linear(dim, dim)
        self.right_gate = nn.Linear(dim, dim)
        self.out_gate = nn.Linear(dim, dim)
        self.to_out_norm = nn.LayerNorm(dim)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        normalized = self.norm(x)
        pair_mask = mask.unsqueeze(-1).to(x.dtype)
        left = self.left_proj(normalized) * pair_mask
        right = self.right_proj(normalized) * pair_mask
        left = left * self.left_gate(normalized).sigmoid()
        right = right * self.right_gate(normalized).sigmoid()
        if self.mix == "outgoing":
            output = torch.einsum("bikd,bjkd->bijd", left, right)
        else:
            output = torch.einsum("bkjd,bkid->bijd", left, right)
        output = self.to_out_norm(output) * self.out_gate(normalized).sigmoid()
        return self.to_out(output)


class CogDetectPairwiseBlock(nn.Module):
    """CogDetect's AlphaFold pairwise block and its following residual FFN."""

    def __init__(self, dim: int, intermediate_size: int, heads: int, layer_norm_eps: float) -> None:
        super().__init__()
        self.triangle_multiply_outgoing = _TriangleMultiplicative(dim, "outgoing")
        self.triangle_multiply_ingoing = _TriangleMultiplicative(dim, "ingoing")
        self.triangle_attention_outgoing = _AxialAttention(
            dim, heads, intermediate_size, row_attn=True
        )
        self.triangle_attention_ingoing = _AxialAttention(
            dim, heads, intermediate_size, row_attn=False, global_query_attn=True
        )
        self.layer_norm = nn.LayerNorm(dim, eps=layer_norm_eps)
        self.linear1 = nn.Linear(dim, intermediate_size)
        self.linear2 = nn.Linear(intermediate_size, dim)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        x = self.triangle_multiply_outgoing(x, mask) + x
        x = self.triangle_multiply_ingoing(x, mask) + x
        x = self.triangle_attention_outgoing(x, x, mask) + x
        x = self.triangle_attention_ingoing(x, x, mask) + x
        residual = x
        x = self.linear2(F.relu(self.linear1(self.layer_norm(x))))
        return x + residual
