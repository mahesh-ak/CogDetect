import torch
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import einsum, nn
import torch.nn.functional as F


def exists(value):
    return value is not None


def default(value, fallback):
    return value if exists(value) else fallback


def init_zero_(layer):
    nn.init.constant_(layer.weight, 0.0)
    if exists(layer.bias):
        nn.init.constant_(layer.bias, 0.0)


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        seq_len=None,
        heads=8,
        dim_head=64,
        dropout=0.0,
        gating=True,
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.seq_len = seq_len
        self.heads = heads
        self.scale = dim_head**-0.5

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.gating = nn.Linear(dim, inner_dim)
        nn.init.constant_(self.gating.weight, 0.0)
        nn.init.constant_(self.gating.bias, 1.0)

        self.dropout = nn.Dropout(dropout)
        init_zero_(self.to_out)

    def forward(
        self,
        x,
        mask=None,
        attn_bias=None,
        context=None,
        context_mask=None,
        tie_dim=None,
    ):
        device, _, heads, has_context = x.device, x.shape, self.heads, exists(context)
        context = default(context, x)

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-1))
        q, k, v = (
            rearrange(tensor, "b n (h d) -> b h n d", h=heads)
            for tensor in (q, k, v)
        )
        q = q * self.scale

        if exists(tie_dim):
            q, k = (
                rearrange(tensor, "(b r) ... -> b r ...", r=tie_dim)
                for tensor in (q, k)
            )
            q = q.mean(dim=1)
            dots = einsum("b h i d, b r h j d -> b r h i j", q, k)
            dots = rearrange(dots, "b r ... -> (b r) ...")
        else:
            dots = einsum("b h i d, b h j d -> b h i j", q, k)

        if exists(attn_bias):
            dots = dots + attn_bias

        if exists(mask):
            context_mask = (
                mask
                if not has_context
                else default(
                    context_mask,
                    lambda: torch.ones(1, k.shape[-2], device=device).bool(),
                )
            )
            mask = mask[:, None, :, None] * context_mask[:, None, None, :]
            dots = dots.masked_fill(~mask, -torch.finfo(dots.dtype).max)

        attention = self.dropout(dots.softmax(dim=-1))
        output = einsum("b h i j, b h j d -> b h i d", attention, v)
        output = rearrange(output, "b h n d -> b n (h d)")
        output = output * self.gating(x).sigmoid()
        return self.to_out(output)


class AxialAttention(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        row_attn=True,
        col_attn=True,
        accept_edges=False,
        global_query_attn=False,
        **kwargs,
    ):
        super().__init__()
        if not row_attn and not col_attn:
            raise ValueError("row or column attention must be turned on")

        self.row_attn = row_attn
        self.col_attn = col_attn
        self.global_query_attn = global_query_attn
        self.norm = nn.LayerNorm(dim)
        self.attn = Attention(dim=dim, heads=heads, **kwargs)
        self.edges_to_attn_bias = (
            nn.Sequential(
                nn.Linear(dim, heads, bias=False),
                Rearrange("b i j h -> b h i j"),
            )
            if accept_edges
            else None
        )

    def forward(self, x, edges=None, mask=None):
        if self.row_attn == self.col_attn:
            raise ValueError("exactly one of row_attn and col_attn must be enabled")

        _, height, width, _ = x.shape
        x = self.norm(x)

        if self.col_attn:
            axial_dim = width
            mask_fold = "b h w -> (b w) h"
            input_fold = "b h w d -> (b w) h d"
            output_fold = "(b w) h d -> b h w d"
        else:
            axial_dim = height
            mask_fold = "b h w -> (b h) w"
            input_fold = "b h w d -> (b h) w d"
            output_fold = "(b h) w d -> b h w d"

        x = rearrange(x, input_fold)
        if exists(mask):
            mask = rearrange(mask, mask_fold)

        attn_bias = None
        if exists(self.edges_to_attn_bias) and exists(edges):
            attn_bias = self.edges_to_attn_bias(edges)
            attn_bias = repeat(attn_bias, "b h i j -> (b x) h i j", x=axial_dim)

        tie_dim = axial_dim if self.global_query_attn else None
        output = self.attn(x, mask=mask, attn_bias=attn_bias, tie_dim=tie_dim)
        return rearrange(output, output_fold, h=height, w=width)


class TriangleMultiplicativeModule(nn.Module):
    def __init__(self, *, dim, hidden_dim=None, mix="ingoing"):
        super().__init__()
        if mix not in {"ingoing", "outgoing"}:
            raise ValueError("mix must be either ingoing or outgoing")

        hidden_dim = default(hidden_dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.left_proj = nn.Linear(dim, hidden_dim)
        self.right_proj = nn.Linear(dim, hidden_dim)
        self.left_gate = nn.Linear(dim, hidden_dim)
        self.right_gate = nn.Linear(dim, hidden_dim)
        self.out_gate = nn.Linear(dim, hidden_dim)

        for gate in (self.left_gate, self.right_gate, self.out_gate):
            nn.init.constant_(gate.weight, 0.0)
            nn.init.constant_(gate.bias, 1.0)

        self.mix_einsum_eq = (
            "... i k d, ... j k d -> ... i j d"
            if mix == "outgoing"
            else "... k j d, ... k i d -> ... i j d"
        )
        self.to_out_norm = nn.LayerNorm(hidden_dim)
        self.to_out = nn.Linear(hidden_dim, dim)

    def forward(self, x, mask=None):
        if x.shape[1] != x.shape[2]:
            raise ValueError("feature map must be symmetrical")
        if exists(mask):
            mask = rearrange(mask, "b i j -> b i j ()")

        x = self.norm(x)
        left = self.left_proj(x)
        right = self.right_proj(x)
        if exists(mask):
            left = left * mask
            right = right * mask

        left = left * self.left_gate(x).sigmoid()
        right = right * self.right_gate(x).sigmoid()
        output = einsum(self.mix_einsum_eq, left, right)
        output = self.to_out_norm(output)
        output = output * self.out_gate(x).sigmoid()
        return self.to_out(output)


class OuterMean(nn.Module):
    def __init__(self, dim, hidden_dim=None, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.norm = nn.LayerNorm(dim)
        hidden_dim = default(hidden_dim, dim)
        self.left_proj = nn.Linear(dim, hidden_dim)
        self.right_proj = nn.Linear(dim, hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, dim)

    def forward(self, x, mask=None):
        x = self.norm(x)
        left = self.left_proj(x)
        right = self.right_proj(x)
        outer = rearrange(left, "b m i d -> b m i () d") * rearrange(
            right, "b m j d -> b m () j d"
        )

        if exists(mask):
            mask = rearrange(mask, "b m i -> b m i () ()") * rearrange(
                mask, "b m j -> b m () j ()"
            )
            outer = outer.masked_fill(~mask, 0.0)
            outer = outer.sum(dim=1) / (mask.sum(dim=1) + self.eps)
        else:
            outer = outer.mean(dim=1)

        return self.proj_out(outer)


class PairwiseAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        seq_len,
        heads,
        dim_head,
        dropout=0.0,
        global_column_attn=False,
    ):
        super().__init__()
        self.outer_mean = OuterMean(dim)
        self.triangle_attention_outgoing = AxialAttention(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            row_attn=True,
            col_attn=False,
            accept_edges=True,
        )
        self.triangle_attention_ingoing = AxialAttention(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            row_attn=False,
            col_attn=True,
            accept_edges=True,
            global_query_attn=global_column_attn,
        )
        self.triangle_multiply_outgoing = TriangleMultiplicativeModule(
            dim=dim, mix="outgoing"
        )
        self.triangle_multiply_ingoing = TriangleMultiplicativeModule(
            dim=dim, mix="ingoing"
        )

    def forward(self, x, mask=None, msa_repr=None, msa_mask=None):
        if exists(msa_repr):
            x = x + self.outer_mean(msa_repr, mask=msa_mask)

        x = self.triangle_multiply_outgoing(x, mask=mask) + x
        x = self.triangle_multiply_ingoing(x, mask=mask) + x
        x = self.triangle_attention_outgoing(x, edges=x, mask=mask) + x
        x = self.triangle_attention_ingoing(x, edges=x, mask=mask) + x
        return x
