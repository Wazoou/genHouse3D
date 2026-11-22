"""Corner transformer: predicts the noise on every corner of a plan.

Architecture after HouseDiffusion (Shabani, Hosseini and Furukawa, CVPR 2023): one token per
corner; in every layer the room, global and relational attentions read the same normalised
input and their outputs are summed. Written from the paper, not from its code. Two changes:

- the timestep conditions each layer through adaptive layer norm with zero-initialised gates
  (adaLN-Zero, from DiT, Peebles and Xie 2023) instead of being added once to the input;
- coordinates enter through Fourier features of the corner and of the vector to the next
  corner of the room (HouseDiffusion samples points along that wall instead).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from housegen.data.schema import ROOM_TYPES
from housegen.model.encoding import MAX_CORNERS, MAX_ROOMS, Batch
from housegen.model.masks import ATTENTIONS, attention_masks


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10_000.0) -> torch.Tensor:
    """Sinusoidal embedding (B, dim) of integer timesteps (B,)."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    args = t.float()[:, None] * freqs[None, :]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


def fourier_features(x: torch.Tensor, n: int) -> torch.Tensor:
    """``x`` followed by sin / cos of ``x`` at ``n`` octaves: (..., d) -> (..., d (1 + 2n))."""
    if n == 0:
        return x
    freqs = (2.0 ** torch.arange(n, device=x.device, dtype=x.dtype)) * math.pi
    angles = x[..., None] * freqs
    return torch.cat([x, angles.sin().flatten(-2), angles.cos().flatten(-2)], dim=-1)


class Attention(nn.Module):
    """Multi-head self-attention restricted by a boolean mask (True = may attend)."""

    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be a multiple of heads")
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b, length, dim = x.shape
        q, k, v = (
            self.qkv(x).view(b, length, 3, self.heads, dim // self.heads).permute(2, 0, 3, 1, 4)
        )
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.proj(out.transpose(1, 2).reshape(b, length, dim))


class Block(nn.Module):
    """Summed masked attentions, then a feed-forward layer, both modulated by the timestep."""

    def __init__(self, dim: int, heads: int, ff_mult: int, attentions: Sequence[str]) -> None:
        super().__init__()
        self.attentions = tuple(attentions)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.ModuleDict({name: Attention(dim, heads) for name in self.attentions})
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_mult * dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ff_mult * dim, dim),
        )
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim))
        nn.init.zeros_(self.modulation[1].weight)
        nn.init.zeros_(self.modulation[1].bias)

    def forward(
        self, x: torch.Tensor, c: torch.Tensor, masks: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        shift1, scale1, gate1, shift2, scale2, gate2 = self.modulation(c).chunk(6, dim=-1)
        h = self.norm1(x) * (1 + scale1) + shift1
        x = x + gate1 * sum(self.attn[name](h, masks[name]) for name in self.attentions)
        h = self.norm2(x) * (1 + scale2) + shift2
        return x + gate2 * self.ff(h)


class CornerTransformer(nn.Module):
    """Noise (or clean corner) prediction for every corner token of a batch of plans."""

    def __init__(
        self,
        dim: int = 256,
        depth: int = 6,
        heads: int = 8,
        ff_mult: int = 4,
        fourier: int = 8,
        attentions: Sequence[str] = ATTENTIONS,
    ) -> None:
        super().__init__()
        unknown = set(attentions) - set(ATTENTIONS)
        if unknown or not attentions:
            raise ValueError(f"attentions must be a non-empty subset of {ATTENTIONS}")
        self.dim, self.fourier = dim, fourier
        self.coord_in = nn.Linear(2 * 2 * (1 + 2 * fourier), dim)
        self.type_emb = nn.Embedding(len(ROOM_TYPES), dim)
        self.corner_emb = nn.Embedding(MAX_CORNERS, dim)
        self.count_emb = nn.Embedding(MAX_CORNERS + 1, dim)
        self.room_emb = nn.Embedding(MAX_ROOMS, dim)
        self.entrance_emb = nn.Embedding(2, dim)
        self.time_mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.blocks = nn.ModuleList(Block(dim, heads, ff_mult, attentions) for _ in range(depth))
        self.norm_out = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.modulation_out = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim))
        self.out = nn.Linear(dim, 2)
        for layer in (self.modulation_out[1], self.out):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor, batch: Batch) -> torch.Tensor:
        """``x``: noisy corners (B, L, 2) in model units, ``t``: timesteps (B,)."""
        masks = attention_masks(batch.room, batch.valid, batch.adjacency)
        edge = torch.gather(x, 1, batch.next[..., None].expand_as(x)) - x
        geometry = torch.cat(
            [fourier_features(x, self.fourier), fourier_features(edge, self.fourier)], dim=-1
        )
        h = (
            self.coord_in(geometry)
            + self.type_emb(batch.type)
            + self.corner_emb(batch.corner)
            + self.count_emb(batch.count)
            + self.room_emb(batch.room_id)
            + self.entrance_emb(batch.entrance)
        )
        c = self.time_mlp(timestep_embedding(t, self.dim))[:, None, :]
        for block in self.blocks:
            h = block(h, c, masks)
        shift, scale = self.modulation_out(c).chunk(2, dim=-1)
        return self.out(self.norm_out(h) * (1 + scale) + shift)
