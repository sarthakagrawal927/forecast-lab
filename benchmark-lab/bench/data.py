"""Benchmark datasets with the split and scaling protocol used by Informer,
Autoformer, DLinear and PatchTST, so numbers are comparable with published tables.

Protocol (Zhou et al. 2021, kept by every later paper):
* ETTh*: 12 / 4 / 4 months train / val / test (hourly); ETTm*: same months at 15 min.
* Weather, Electricity: 70 / 10 / 20 % chronological split.
* Each val/test segment starts ``lookback`` steps early, so the first window's
  history comes from the previous segment, but every *target* lies inside it.
* Standardise every channel with mean/std of the training segment only.
* Metrics are MSE / MAE on the standardised data, averaged over all channels
  ("multivariate" setting). We score every test window (no drop_last).
"""
from __future__ import annotations

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ETT_URL = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/{name}.csv"
SEASON = {"ETTh1": 24, "ETTh2": 24, "ETTm1": 96, "ETTm2": 96}  # one day in steps


def load_frame(name: str) -> pd.DataFrame:
    path = DATA_DIR / f"{name}.csv"
    if not path.exists():
        if not name.startswith("ETT"):
            raise FileNotFoundError(f"{path} missing; see README for where to get {name}")
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(ETT_URL.format(name=name), path)
    return pd.read_csv(path, parse_dates=["date"], index_col="date")


def borders(name: str, n: int, lookback: int) -> list[tuple[int, int]]:
    """(start, end) row ranges for train, val, test, matching the reference loaders."""
    if name.startswith("ETTh") or name.startswith("ETTm"):
        unit = 30 * 24 * (4 if name.startswith("ETTm") else 1)
        ends = [12 * unit, 16 * unit, 20 * unit]
    else:
        n_train, n_test = int(n * 0.7), int(n * 0.2)
        ends = [n_train, n - n_test, n]
    starts = [0, ends[0] - lookback, ends[1] - lookback]
    return list(zip(starts, ends))


@dataclass
class Split:
    x: np.ndarray        # [T, C] standardised series for this segment (incl. lookback lead-in)
    lookback: int
    horizon: int

    def __len__(self) -> int:
        return len(self.x) - self.lookback - self.horizon + 1

    def windows(self, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """History [B, L, C] and target [B, H, C] for window start indices."""
        hist = np.stack([self.x[i:i + self.lookback] for i in idx])
        targ = np.stack([self.x[i + self.lookback:i + self.lookback + self.horizon] for i in idx])
        return hist, targ


def make_splits(name: str, lookback: int, horizon: int):
    df = load_frame(name)
    values = df.to_numpy(np.float32)
    (a0, a1), (b0, b1), (c0, c1) = borders(name, len(values), lookback)
    mean, std = values[a0:a1].mean(0), values[a0:a1].std(0) + 1e-8  # train segment only
    z = (values - mean) / std
    return (Split(z[a0:a1], lookback, horizon), Split(z[b0:b1], lookback, horizon),
            Split(z[c0:c1], lookback, horizon), list(df.columns))
