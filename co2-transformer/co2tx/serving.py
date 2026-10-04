"""Inference-time wrapper around the trained seed ensemble.

The API hands this a run's history (wide frame, one row per analyser step, up
to the forecast origin) and gets back quantile forecasts. All preprocessing is
the training code path: Preprocessor.step_features + window_at.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import Preprocessor, window_at
from .model import QUANTILES
from .train import TrainConfig, build, cold_quantiles, finalize

DEFAULT_ARTIFACT = Path(__file__).resolve().parent.parent / "artifacts" / "co2_transformer.pt"
STEP_SECONDS = 43  # analyser cadence; horizon h lands ~h*43 s after the origin


@dataclass
class Forecast:
    origin: pd.Timestamp
    quantiles: np.ndarray  # [H, 6, Q] CO2 %
    members_spread: np.ndarray  # [H, 6] std of member medians: epistemic-uncertainty monitor
    seen: np.ndarray  # [6] point already read in this run at the origin; False = cold-start estimate

    def records(self) -> list[dict]:
        rows = []
        H = self.quantiles.shape[0]
        for h in range(H):
            for p in range(6):
                q = self.quantiles[h, p]
                rows.append({
                    "horizon_step": h + 1,
                    "target_time": (self.origin + pd.Timedelta(seconds=STEP_SECONDS * (h + 1))).isoformat(),
                    "sampling_point": p + 1,
                    "p10": round(float(q[0]), 4), "p50": round(float(q[1]), 4), "p90": round(float(q[2]), 4),
                    "ensemble_std": round(float(self.members_spread[h, p]), 4),
                    # Points not yet read this run come from the linear soft sensor (notebook RC2).
                    "point_read_this_run": bool(self.seen[p]),
                    "estimator": "transformer" if self.seen[p] else "cold-start-linear",
                })
        return rows


class Ensemble:
    def __init__(self, path: Path = DEFAULT_ARTIFACT):
        blob = torch.load(path, map_location="cpu", weights_only=False)
        self.cfg = TrainConfig(**blob["config"])
        p = blob["preprocessor"]
        self.pre = Preprocessor(p["features"], p["mean"], p["std"], p["point_mean"])
        n_features = len(self.pre.features) + 24
        self.models = []
        for state in blob["members"]:
            m = build(self.cfg, n_features)
            m.load_state_dict(state)
            m.eval()
            self.models.append(m)
        cold = blob["cold_start"]  # linear soft sensor for points not yet read this run
        self.cold_coef, self.cold_intercept = cold["coef"], cold["intercept"]
        self.cold_offsets = cold["offsets"]  # [2, 6] conformal residual quantiles
        # Content hash: a retrained artifact gets a new version even with the same
        # config, so /monitoring never mixes forecasts from two different models.
        digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()[:8]
        self.version = f"{path.stem}-L{self.cfg.lookback}-H{self.cfg.horizon}-n{len(self.models)}-{digest}"

    @property
    def required_tags(self) -> list[str]:
        return self.pre.features

    @torch.no_grad()
    def forecast(self, history: pd.DataFrame) -> Forecast:
        """``history``: rows from the start of the run up to and including the
        origin, with the training tag columns plus ``co2`` and ``point``."""
        missing = [c for c in self.pre.features + ["co2", "point"] if c not in history.columns]
        if missing:
            raise ValueError(f"history is missing tags: {missing[:5]}")
        feats, raw = self.pre.step_features(history)
        win, pad = window_at(feats, len(history) - 1, self.cfg.lookback)
        x, padt = torch.from_numpy(win)[None], torch.from_numpy(pad)[None]
        last = torch.from_numpy(raw["last"][-1:].astype(np.float32))
        member_q = np.stack([m(x, padt, last)[0].numpy() for m in self.models])  # [M,H,6,Q]
        q = member_q.mean(0)
        seen = np.isin(np.arange(1, 7), history["point"].to_numpy())
        if not seen.all():  # same routing as co2tx.train.route
            med = last[0].numpy() + np.einsum("d,hpd->hp", win.reshape(-1), self.cold_coef) + self.cold_intercept
            q[:, ~seen] = cold_quantiles(med, self.cold_offsets)[:, ~seen]
        q = finalize(q)
        return Forecast(pd.Timestamp(history.index[-1]), q, member_q[..., QUANTILES.index(0.5)].std(0), seen)
