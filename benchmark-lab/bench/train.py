"""Training (MSE, Adam, early stopping on the validation segment) and evaluation."""
from __future__ import annotations

import copy
import random

import numpy as np
import torch

from .data import Split


def device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def batches(split: Split, batch: int, shuffle: bool, rng: np.random.Generator | None = None):
    idx = np.arange(len(split))
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, len(idx), batch):
        hist, targ = split.windows(idx[i:i + batch])
        yield torch.from_numpy(hist), torch.from_numpy(targ)


@torch.no_grad()
def predict(fn, split: Split, batch: int = 512) -> np.ndarray:
    """Forecasts [N, H, C] for every window of ``split``. ``fn`` maps a history
    tensor to a forecast tensor (a model or a baseline closure)."""
    dev = device()
    if isinstance(fn, torch.nn.Module):
        fn.eval()
    out = [fn(h.to(dev)).float().cpu().numpy() for h, _ in batches(split, batch, shuffle=False)]
    return np.concatenate(out)


def targets(split: Split) -> np.ndarray:
    return split.windows(np.arange(len(split)))[1]


def scores(pred: np.ndarray, y: np.ndarray) -> dict:
    err = pred - y
    return {"mse": float(np.mean(err ** 2)), "mae": float(np.mean(np.abs(err)))}


def fit(model: torch.nn.Module, train: Split, val: Split, *, lr: float, batch: int, max_epochs: int,
        patience: int, seed: int, halve_lr: bool = False) -> tuple[torch.nn.Module, dict]:
    """Adam on MSE; keep the epoch with the best validation MSE.

    Epoch 0 (the untrained model) is scored too: for the hybrid, whose
    correction starts at zero, that is the linear model itself, so the hybrid
    can never be selected worse than its anchor on validation.
    """
    seed_everything(seed)
    dev = device()
    model.to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    # The DLinear reference code halves the learning rate after every epoch ("type1").
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.5) if halve_lr else None
    y_val = targets(val)
    best = scores(predict(model, val), y_val)["mse"]
    best_state, best_epoch, stale = copy.deepcopy(model.state_dict()), 0, 0
    rng = np.random.default_rng(seed)
    history = [best]
    for epoch in range(1, max_epochs + 1):
        model.train()
        for hist, targ in batches(train, batch, shuffle=True, rng=rng):
            loss = torch.mean((model(hist.to(dev)) - targ.to(dev)) ** 2)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if sched:
            sched.step()
        v = scores(predict(model, val), y_val)["mse"]
        history.append(v)
        if v < best - 1e-6:
            best, best_state, best_epoch, stale = v, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    return model, {"best_epoch": best_epoch, "val_mse": best, "val_curve": history}
