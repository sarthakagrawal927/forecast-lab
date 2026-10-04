"""Helpers for the analysis notebook: per-cell error tables, permutation
importance and attention summaries for the trained ensemble."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import torch

from .data import AGE_CAP, TEST_RUN, Windows, load_runs, make_windows, split_runs
from .serving import Ensemble
from .train import RidgeBank, cold_quantiles, finalize, metrics, persistence, predict, ridge, route


def load_test_context():
    """Everything the notebook needs about the held-out run, built once."""
    runs = load_runs()
    train_runs, test_runs = split_runs(runs, [TEST_RUN])
    ens = Ensemble()
    L, H = ens.cfg.lookback, ens.cfg.horizon
    w = make_windows(ens.pre, test_runs, L, H)
    wtr = make_windows(ens.pre, train_runs, L, H)
    raw = ensemble_predict(ens, w)
    cold_med = RidgeBank(ens.cold_coef, ens.cold_intercept).predict(w)
    q = finalize(raw)  # Transformer alone
    routed = finalize(route(raw, cold_quantiles(cold_med, ens.cold_offsets), w))  # exactly what the API serves
    preds = {"routed": routed[..., 1], "transformer": q[..., 1], "ridge": ridge(wtr, w), "persistence": persistence(w)}
    return dict(runs=runs, ens=ens, w=w, wtr=wtr, q=q, routed=routed, preds=preds)


def ensemble_predict(ens: Ensemble, w: Windows) -> np.ndarray:
    return np.mean([predict(m, w) for m in ens.models], axis=0)


def cell_table(ctx) -> pd.DataFrame:
    """One row per measured (origin, horizon) cell with context for slicing errors."""
    w, q, df = ctx["w"], ctx["routed"], ctx["runs"][TEST_RUN]
    point, co2 = df["point"].to_numpy(), df["co2"].to_numpy()
    visit = (df["point"] != df["point"].shift()).cumsum().to_numpy()
    pos = pd.Series(visit).groupby(visit).cumcount().to_numpy()
    n_sensor = len(ctx["ens"].pre.features)
    rows = []
    for i, t in enumerate(w.origin):
        for h in range(w.y.shape[1]):
            tt = t + h + 1
            if tt >= len(df):
                continue
            p = point[tt] - 1
            row = dict(origin=t, time=df.index[tt], h=h + 1, point=p + 1, pos_in_visit=pos[tt], y=co2[tt],
                       age_at_origin=float(w.x[i, -1, n_sensor + 6 + p] * AGE_CAP),
                       p10=q[i, h, p, 0], p90=q[i, h, p, 2])
            for name, pred in ctx["preds"].items():
                row[name] = pred[i, h, p]
            rows.append(row)
    cells = pd.DataFrame(rows)
    for name in ctx["preds"]:
        cells[f"err_{name}"] = cells[name] - cells["y"]
    cells["covered"] = (cells["y"] >= cells["p10"]) & (cells["y"] <= cells["p90"])
    return cells


def rmse(x) -> float:
    return float(np.sqrt(np.mean(np.square(x))))


def permutation_importance(ctx, repeats: int = 5, seed: int = 0) -> pd.Series:
    """Increase in test RMSE (points 5-6) when one input group is shuffled
    across samples. Groups: each sensor tag, plus the four observation-state
    blocks (last value, age, seen flag, current analyser position)."""
    ens, w = ctx["ens"], ctx["w"]
    base = metrics(ctx["q"], w)["rmse_p56"]
    names = ens.pre.features + ["last observed CO2"] * 6 + ["age of reading"] * 6 + ["seen flag"] * 6 \
        + ["analyser position"] * 6
    groups: dict[str, list[int]] = {}
    for j, n in enumerate(names):
        groups.setdefault(n, []).append(j)
    rng = np.random.default_rng(seed)
    out = {}
    for name, cols in groups.items():
        deltas = []
        for _ in range(repeats):
            x = w.x.copy()
            x[:, :, cols] = w.x[rng.permutation(len(x))][:, :, cols]
            w2 = replace(w, x=x)
            deltas.append(metrics(ensemble_predict(ens, w2), w2)["rmse_p56"] - base)
        out[name] = float(np.mean(deltas))
    return pd.Series(out).sort_values(ascending=False)


@torch.no_grad()
def mean_attention(ctx) -> np.ndarray:
    """[layers, L, L] attention averaged over heads, samples and ensemble members."""
    ens, w = ctx["ens"], ctx["w"]
    x, pad, last = torch.from_numpy(w.x), torch.from_numpy(w.pad), torch.from_numpy(w.last)
    maps = []
    for m in ens.models:
        m(x, pad, last)
        maps.append(torch.stack([a.mean(1) for a in m.attention_maps()]).mean(1).numpy())
    return np.mean(maps, axis=0)
