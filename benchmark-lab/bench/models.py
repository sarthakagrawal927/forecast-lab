"""The model ladder. Shapes: B batch, L lookback, H horizon, C channels.
Every model maps a history [B, L, C] to a forecast [B, H, C] (standardised units).

Ladder: naive → seasonal-naive → DLinear → Transformer (time-step tokens, the
co2-transformer design) → Transformer (patch tokens, PatchTST-style) → hybrid
(linear anchor + Transformer correction).
"""
from __future__ import annotations

import math

import torch
from torch import nn

# ----------------------------------------------------------------- baselines


def naive(hist: torch.Tensor, horizon: int) -> torch.Tensor:
    """Repeat the last value ("Repeat" in the DLinear paper's tables)."""
    return hist[:, -1:, :].expand(-1, horizon, -1)


def seasonal_naive(hist: torch.Tensor, horizon: int, season: int) -> torch.Tensor:
    """Repeat the last full season (e.g. yesterday, hour by hour)."""
    last = hist[:, -season:, :]
    reps = math.ceil(horizon / season)
    return last.repeat(1, reps, 1)[:, :horizon, :]


class DLinear(nn.Module):
    """Zeng et al. 2022: split the window into a moving-average trend and the
    remainder, map each to the horizon with one linear layer shared by all channels."""

    def __init__(self, lookback: int, horizon: int, kernel: int = 25):
        super().__init__()
        self.kernel = kernel
        self.trend = nn.Linear(lookback, horizon)
        self.season = nn.Linear(lookback, horizon)

    def forward(self, x):  # [B, L, C]
        pad = (self.kernel - 1) // 2
        padded = torch.cat([x[:, :1].expand(-1, pad, -1), x, x[:, -1:].expand(-1, pad, -1)], dim=1)
        trend = padded.unfold(1, self.kernel, 1).mean(-1)          # [B, L, C]
        season = x - trend
        out = self.trend(trend.transpose(1, 2)) + self.season(season.transpose(1, 2))
        return out.transpose(1, 2)                                   # [B, H, C]


# ------------------------------------------------- Transformer building blocks
# Same hand-written attention as co2-transformer/co2tx/model.py (labs in this
# repo share ideas, not code).


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.h, self.dh = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, N, D = x.shape
        q, k, v = self.qkv(x).view(B, N, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        w = torch.softmax(q @ k.transpose(-2, -1) / math.sqrt(self.dh), dim=-1)
        return self.out((self.drop(w) @ v).transpose(1, 2).reshape(B, N, D))


class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.drop(self.attn(self.ln1(x)))
        return x + self.drop(self.ff(self.ln2(x)))


def sinusoidal(n: int, d: int) -> torch.Tensor:
    pos = torch.arange(n).float()[:, None]
    div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
    pe = torch.zeros(n, d)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
    return pe


class RevIN(nn.Module):
    """Reversible instance normalisation (Kim et al. 2022): remove each window's
    own mean/std per channel before the model, put them back after. Handles the
    level shifts between train and test periods that sink many Transformers."""

    def norm(self, x):
        self.mu = x.mean(1, keepdim=True).detach()
        self.sd = (x.var(1, keepdim=True, unbiased=False) + 1e-5).sqrt().detach()
        return (x - self.mu) / self.sd

    def denorm(self, y):
        return y * self.sd + self.mu


class StepTransformer(nn.Module):
    """Informer-era tokenisation, as in co2-transformer: one token per time step
    embedding all channels jointly; read out from the last + mean token."""

    def __init__(self, lookback, horizon, n_channels, d_model=64, n_heads=4, n_layers=2, d_ff=128,
                 dropout=0.1):
        super().__init__()
        self.H, self.C = horizon, n_channels
        self.revin = RevIN()
        self.inp = nn.Linear(n_channels, d_model)
        self.register_buffer("pos", sinusoidal(lookback, d_model), persistent=False)
        self.blocks = nn.Sequential(*[EncoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(2 * d_model, horizon * n_channels)

    def forward(self, x):
        h = self.ln(self.blocks(self.inp(self.revin.norm(x)) + self.pos))
        z = torch.cat([h[:, -1], h.mean(1)], dim=-1)
        return self.revin.denorm(self.head(z).view(-1, self.H, self.C))


class PatchTransformer(nn.Module):
    """PatchTST-style (Nie et al. 2023): each channel is its own sequence
    (channel independence), cut into overlapping patches; each patch is a token.
    Small-dataset sizes follow the paper's ETTh1 configuration."""

    def __init__(self, lookback, horizon, n_channels, patch=16, stride=8, d_model=16, n_heads=4, n_layers=3,
                 d_ff=128, dropout=0.3, zero_head=False):
        super().__init__()
        self.patch, self.stride, self.H = patch, stride, horizon
        self.n_patch = (lookback - patch) // stride + 2  # +1 for the end padding
        self.revin = RevIN()
        self.embed = nn.Linear(patch, d_model)
        self.pos = nn.Parameter(torch.randn(self.n_patch, d_model) * 0.02)
        self.blocks = nn.Sequential(*[EncoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(self.n_patch * d_model, horizon)
        if zero_head:  # used by the hybrid: start as "no correction"
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def forward(self, x):  # [B, L, C]
        B, L, C = x.shape
        z = self.revin.norm(x).transpose(1, 2).reshape(B * C, L)                 # one series per channel
        z = torch.cat([z, z[:, -1:].expand(-1, self.stride)], dim=1)             # pad end, as PatchTST
        tokens = z.unfold(1, self.patch, self.stride)                           # [B*C, P, patch]
        h = self.ln(self.blocks(self.embed(tokens) + self.pos))
        y = self.head(h.flatten(1)).view(B, C, self.H).transpose(1, 2)
        return self.revin.denorm(y)


class Hybrid(nn.Module):
    """A frozen, already-trained linear forecast plus a Transformer that learns
    only the correction. The Transformer's head starts at zero, so the hybrid
    starts exactly at the linear model: the same "anchor on a strong simple
    forecast, learn the departures" idea as co2-transformer's residual head."""

    def __init__(self, anchor: nn.Module, residual: PatchTransformer):
        super().__init__()
        self.anchor = anchor.requires_grad_(False).eval()
        self.residual = residual

    def forward(self, x):
        with torch.no_grad():
            base = self.anchor(x)
        # The residual is RevIN-denormalised, so remove its added window mean:
        # it should model a correction, not a second copy of the level.
        corr = self.residual(x) - self.residual.revin.mu
        return base + corr

    def train(self, mode: bool = True):
        super().train(mode)
        self.anchor.eval()  # the anchor never switches to training mode
        return self
