"""A Transformer forecaster written from scratch (no nn.Transformer*, no
nn.MultiheadAttention, no external model code), plus a GRU twin that shares
the same input and head so the backbone is the only thing that differs.

Shape legend: B batch, L lookback steps, F input features, D model width,
H forecast horizons, P=6 sampling points, Q quantiles.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F_

QUANTILES = (0.1, 0.5, 0.9)


class MultiHeadSelfAttention(nn.Module):
    """softmax(QK^T / sqrt(d_head)) V, split across heads (Vaswani et al. 2017)."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.dh = n_heads, d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.last_weights: torch.Tensor | None = None  # [B, heads, L, L] for interpretability

    def forward(self, x: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        q, k, v = self.qkv(x).view(B, L, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)  # 3×[B,h,L,dh]
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.dh)                      # [B,h,L,L]
        # Padded steps can never be attended to. Every window ends on a real step,
        # so each query row keeps at least one finite score.
        scores = scores.masked_fill(pad[:, None, None, :], float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        self.last_weights = weights.detach()
        ctx = (self.drop(weights) @ v).transpose(1, 2).reshape(B, L, D)
        return self.out(ctx)


class EncoderBlock(nn.Module):
    """Pre-LayerNorm block: x + Attn(LN(x)); x + FFN(LN(x)). Pre-LN trains
    stably without learning-rate warmup tricks (Xiong et al. 2020)."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, pad):
        x = x + self.drop(self.attn(self.ln1(x), pad))
        return x + self.drop(self.ff(self.ln2(x)))


def sinusoidal_positions(length: int, d_model: int) -> torch.Tensor:
    pos = torch.arange(length).float()[:, None]
    div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe = torch.zeros(length, d_model)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
    return pe


class QuantileResidualHead(nn.Module):
    """Predicts, per horizon and sampling point, a correction to the last
    observed value (persistence) plus non-crossing quantiles around it.

    Output [B, H, P, Q] in CO2 %. Residual learning means an untrained head
    already equals the persistence baseline, so the network only has to learn
    how the process moves away from "nothing changes".
    """

    def __init__(self, d_in: int, horizon: int, n_points: int = 6, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.H, self.P = horizon, n_points
        self.mlp = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, horizon * n_points * 3))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, z: torch.Tensor, last: torch.Tensor) -> torch.Tensor:
        o = self.mlp(z).view(-1, self.H, self.P, 3)
        median = last[:, None, :] + o[..., 0]
        lower = median - F_.softplus(o[..., 1]) - 1e-3
        upper = median + F_.softplus(o[..., 2]) + 1e-3
        return torch.stack([lower, median, upper], dim=-1)


class CO2Transformer(nn.Module):
    def __init__(self, n_features: int, lookback: int, horizon: int, d_model: int = 64,
                 n_heads: int = 4, n_layers: int = 2, d_ff: int = 128, dropout: float = 0.1):
        super().__init__()
        self.inp = nn.Linear(n_features, d_model)
        self.register_buffer("pos", sinusoidal_positions(lookback, d_model), persistent=False)
        self.blocks = nn.ModuleList(EncoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers))
        self.ln = nn.LayerNorm(d_model)
        self.head = QuantileResidualHead(2 * d_model, horizon, dropout=dropout)

    def encode(self, x: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        h = self.inp(x) + self.pos[: x.shape[1]]
        for block in self.blocks:
            h = block(h, pad)
        h = self.ln(h)
        keep = (~pad).float()[..., None]
        mean = (h * keep).sum(1) / keep.sum(1)            # what the whole window looked like
        return torch.cat([h[:, -1], mean], dim=-1)        # + what "now" looks like

    def forward(self, x, pad, last):
        return self.head(self.encode(x, pad), last)

    def attention_maps(self) -> list[torch.Tensor]:
        return [b.attn.last_weights for b in self.blocks]


class CO2GRU(nn.Module):
    """Recurrent baseline with the identical head: isolates the backbone choice."""

    def __init__(self, n_features: int, lookback: int, horizon: int, d_model: int = 64,
                 n_layers: int = 2, dropout: float = 0.1, **_):
        super().__init__()
        self.gru = nn.GRU(n_features, d_model, n_layers, batch_first=True, dropout=dropout)
        self.ln = nn.LayerNorm(d_model)
        self.head = QuantileResidualHead(2 * d_model, horizon, dropout=dropout)

    def forward(self, x, pad, last):
        h = self.ln(self.gru(x)[0])  # left padding is zeros, so the GRU sees a quiet warm-up
        keep = (~pad).float()[..., None]
        mean = (h * keep).sum(1) / keep.sum(1)
        return self.head(torch.cat([h[:, -1], mean], dim=-1), last)


def pinball_loss(pred: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Quantile loss averaged over observed (horizon, point) cells only.

    pred [B,H,P,Q], y/mask [B,H,P]. Unobserved cells contribute nothing, so the
    model is never trained on invented labels.
    """
    q = torch.tensor(QUANTILES, device=pred.device)
    err = y[..., None] - pred
    loss = torch.maximum(q * err, (q - 1) * err).sum(-1)
    return (loss * mask).sum() / mask.sum().clamp_min(1)


class LogSpace(nn.Module):
    """Runs any forecaster on log1p(CO2): the residual head then learns relative
    changes, so a run at 10 % and a run at 3 % share one scale (root cause RC3).
    ``raw`` stays in log space for the loss; ``forward`` returns CO2 % again.
    Quantiles survive the transform because expm1 is monotone."""

    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner

    def raw(self, x, pad, last):
        return self.inner(x, pad, torch.log1p(last))

    def forward(self, x, pad, last):
        return torch.expm1(self.raw(x, pad, last))

    def attention_maps(self):
        return self.inner.attention_maps()


MODELS = {"transformer": CO2Transformer, "gru": CO2GRU}
