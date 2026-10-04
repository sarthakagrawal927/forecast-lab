"""Training loop, metrics and the baselines every model must beat."""
from __future__ import annotations

import copy
import random
from dataclasses import asdict, dataclass

import numpy as np
import torch

from .data import Windows
from .model import MODELS, QUANTILES, LogSpace, pinball_loss

INFORMATIVE = [4, 5]  # zero-based points 5 and 6; points 1-4 sit near 0 % CO2


@dataclass
class TrainConfig:
    arch: str = "transformer"
    lookback: int = 18
    horizon: int = 12
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 128
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-2
    batch_size: int = 64
    max_epochs: int = 150
    patience: int = 25
    input_noise: float = 0.05  # Gaussian jitter on inputs: cheap regulariser for 800 samples
    seed: int = 0
    name: str = "base"  # label for experiment bookkeeping only
    warm_only: bool = False  # train/early-stop only on cells whose point was already read (the router's job)
    log_target: bool = False  # model log1p(CO2) so error does not scale with the CO2 level


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_tensors(w: Windows):
    return (torch.from_numpy(w.x), torch.from_numpy(w.pad), torch.from_numpy(w.last.astype(np.float32)),
            torch.from_numpy(w.y), torch.from_numpy(w.mask))


def build(cfg: TrainConfig, n_features: int) -> torch.nn.Module:
    model = MODELS[cfg.arch](n_features=n_features, lookback=cfg.lookback, horizon=cfg.horizon,
                             d_model=cfg.d_model, n_heads=cfg.n_heads, n_layers=cfg.n_layers,
                             d_ff=cfg.d_ff, dropout=cfg.dropout)
    return LogSpace(model) if cfg.log_target else model


def training_loss(model, x, pad, last, y, mask) -> torch.Tensor:
    """Pinball loss in the space the model learns in (log1p CO2 for LogSpace)."""
    if isinstance(model, LogSpace):
        return pinball_loss(model.raw(x, pad, last), torch.log1p(y), mask)
    return pinball_loss(model(x, pad, last), y, mask)


@torch.no_grad()
def predict(model: torch.nn.Module, w: Windows, batch: int = 512) -> np.ndarray:
    """Quantile forecasts [N, H, 6, Q] in CO2 %."""
    model.eval()
    x, pad, last, _, _ = to_tensors(w)
    out = [model(x[i:i + batch], pad[i:i + batch], last[i:i + batch]) for i in range(0, len(x), batch)]
    return torch.cat(out).numpy()


def loss_mask(w: Windows, warm_only: bool) -> torch.Tensor:
    m = w.mask * (~cold_cells(w)) if warm_only else w.mask
    return torch.from_numpy(np.ascontiguousarray(m, dtype=np.float32))


@torch.no_grad()
def val_loss(model, w: Windows, warm_only: bool = False) -> float:
    model.eval()
    x, pad, last, y, _ = to_tensors(w)
    return training_loss(model, x, pad, last, y, loss_mask(w, warm_only)).item()


def fit(cfg: TrainConfig, train: Windows, val: Windows | None = None, epochs: int | None = None):
    """Early-stop on ``val`` if given, else stop after exactly ``epochs``.

    The learning-rate schedule always spans ``cfg.max_epochs``, so a final fit
    that stops at the CV-chosen epoch sees the same learning rates CV did.
    Returns (model, history); history["best_epoch"] tells the final fit how long to train.
    """
    seed_everything(cfg.seed)
    model = build(cfg, train.x.shape[-1])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    n_epochs = cfg.max_epochs if epochs is None else epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.max_epochs)
    x, pad, last, y, _ = to_tensors(train)
    m = loss_mask(train, cfg.warm_only)
    n_sensor = train.x.shape[-1] - 24  # the last 24 columns are observation-state features
    history = {"train": [], "val": [], "best_epoch": n_epochs}
    best, best_state, stale = float("inf"), None, 0
    if val is not None:
        # Epoch 0 is the untrained model, which is exactly persistence (zero-init
        # residual head). If no epoch beats it, training stops at 0: do no harm.
        best, best_state = val_loss(model, val, cfg.warm_only), copy.deepcopy(model.state_dict())
        history["val"].append(best)
        history["best_epoch"] = 0
    for epoch in range(1, n_epochs + 1):
        model.train()
        perm = torch.randperm(len(x))
        total = 0.0
        for i in range(0, len(x), cfg.batch_size):
            idx = perm[i:i + cfg.batch_size]
            xb = x[idx].clone()
            if cfg.input_noise:
                xb[..., :n_sensor] += cfg.input_noise * torch.randn_like(xb[..., :n_sensor]) * (~pad[idx])[..., None]
            loss = training_loss(model, xb, pad[idx], last[idx], y[idx], m[idx])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(idx)
        sched.step()
        history["train"].append(total / len(x))
        if val is not None:
            v = val_loss(model, val, cfg.warm_only)
            history["val"].append(v)
            if v < best - 1e-5:
                best, best_state, stale = v, copy.deepcopy(model.state_dict()), 0
                history["best_epoch"] = epoch
            else:
                stale += 1
                if stale >= cfg.patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history


# ---------------------------------------------------------------- metrics

def clip_physical(pred: np.ndarray) -> np.ndarray:
    """CO2 % cannot be negative. Applied identically to every model and in serving."""
    return np.maximum(pred, 0.0)


def metrics(pred_q: np.ndarray, w: Windows) -> dict:
    """Errors only where the analyser actually measured. pred_q [N,H,6,Q] or [N,H,6]."""
    pred_q = clip_physical(pred_q)
    med = pred_q[..., QUANTILES.index(0.5)] if pred_q.ndim == 4 else pred_q
    m = w.mask.astype(bool)
    err = (med - w.y)[m]
    point = np.broadcast_to(np.arange(6), m.shape)[m]
    hor = np.broadcast_to(np.arange(1, m.shape[1] + 1)[:, None], m.shape)[m]
    cold = cold_cells(w)[m]
    informative = np.isin(point, INFORMATIVE)
    out = {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "rmse_p56": float(np.sqrt(np.mean(err[informative] ** 2))),
        "rmse_p56_warm": float(np.sqrt(np.mean(err[informative & ~cold] ** 2))),
        "rmse_p56_cold": float(np.sqrt(np.mean(err[informative & cold] ** 2))) if (informative & cold).any() else None,
        "cold_share": float(np.mean(cold[informative])),
        "rmse_by_point": {int(p + 1): float(np.sqrt(np.mean(err[point == p] ** 2))) for p in range(6)},
        "rmse_by_horizon": {int(h): float(np.sqrt(np.mean(err[hor == h] ** 2))) for h in np.unique(hor)},
        "rmse_p56_by_horizon": {
            int(h): float(np.sqrt(np.mean(err[(hor == h) & np.isin(point, INFORMATIVE)] ** 2))) for h in np.unique(hor)
        },
        "n": int(m.sum()),
    }
    if pred_q.ndim == 4:
        lo, hi = pred_q[..., 0][m], pred_q[..., -1][m]
        out["coverage_80"] = float(np.mean((w.y[m] >= lo) & (w.y[m] <= hi)))
        out["width_80"] = float(np.mean(hi - lo))
    return out


# ---------------------------------------------------------------- baselines

def persistence(w: Windows) -> np.ndarray:
    """Tomorrow looks like today: last observed value of each point, all horizons."""
    return np.repeat(w.last[:, None, :], w.y.shape[1], axis=1)


def climatology(w: Windows, point_mean: np.ndarray) -> np.ndarray:
    return np.broadcast_to(point_mean[None, None, :], w.y.shape).copy()


def cold_cells(w: Windows) -> np.ndarray:
    """[N, H, 6] True where the target point had not been read yet in this run
    at the forecast origin: its "last value" is only the training-mean fill, so
    the cell measures soft sensing from process variables, not forecasting."""
    seen = w.x[:, -1, -12:-6] > 0.5  # seen flags of the origin step
    return np.broadcast_to(~seen[:, None, :], w.y.shape)


@dataclass
class RidgeBank:
    """One linear model per (horizon, point) on the flattened window, predicting
    the residual over persistence. Plain arrays, so serving needs no sklearn."""

    coef: np.ndarray       # [H, 6, L*F]
    intercept: np.ndarray  # [H, 6]

    def predict(self, w: Windows) -> np.ndarray:
        x = w.x.reshape(len(w.x), -1)
        return persistence(w) + np.einsum("nd,hpd->nhp", x, self.coef) + self.intercept


def fit_ridge(train: Windows, alpha: float = 30.0) -> RidgeBank:
    from sklearn.linear_model import Ridge  # analysis-only dependency, not in the serving image

    xtr = train.x.reshape(len(train.x), -1)
    H = train.y.shape[1]
    coef, intercept = np.zeros((H, 6, xtr.shape[1])), np.zeros((H, 6))
    for h in range(H):
        for p in range(6):
            sel = train.mask[:, h, p] > 0
            if sel.sum() < 10:
                continue  # leave as pure persistence
            reg = Ridge(alpha=alpha).fit(xtr[sel], train.y[sel, h, p] - train.last[sel, p])
            coef[h, p], intercept[h, p] = reg.coef_, reg.intercept_
    return RidgeBank(coef, intercept)


def ridge(train: Windows, test: Windows, alpha: float = 30.0) -> np.ndarray:
    return fit_ridge(train, alpha).predict(test)


def cold_quantiles(median: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Soft-sensor median ± conformal residual quantiles (offsets [2, 6]) → [..., 6, 3]."""
    return np.stack([median + offsets[0], median, median + offsets[1]], axis=-1)


def finalize(q: np.ndarray) -> np.ndarray:
    """Order the quantiles, then clip at 0 %: the last step in training-side
    evaluation and in serving alike."""
    return clip_physical(np.sort(q, axis=-1))


def route(warm: np.ndarray, cold: np.ndarray, w: Windows) -> np.ndarray:
    """Transformer where the point has been read this run; the linear soft
    sensor where it has not. Works on [N,H,6] or [N,H,6,Q] arrays."""
    mask = cold_cells(w)
    if warm.ndim == 4:
        mask = mask[..., None]
    return np.where(mask, cold, warm)


def config_dict(cfg: TrainConfig) -> dict:
    return asdict(cfg)
