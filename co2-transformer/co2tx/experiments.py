"""End-to-end experiment: leave-one-run-out CV over lookback windows, ablations,
final seed-ensemble fit, held-out test evaluation and artifacts.

    uv run python -m co2tx.experiments            # full run (~10 min on a laptop)
    uv run python -m co2tx.experiments --quick    # 1 seed, 3 lookbacks (smoke)

Outputs: results/*.json|csv (tracked) and artifacts/co2_transformer.pt (used by
the API). The test run is never seen by model selection.
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS, ARTIFACTS = ROOT / "results", ROOT / "artifacts"
LOOKBACKS = [3, 6, 12, 18, 24, 36]
HORIZON = 12  # 12 × 43 s ≈ 8.6 min ahead, about two full analyser cycles
SEEDS = [0, 1, 2]
FINAL_SEEDS = [0, 1, 2, 3, 4]
# Capacity/regularisation variants tried before the lookback sweep.
CAPACITY = {
    "base": {},
    "small": dict(d_model=32, d_ff=64, n_layers=1),
    "regularised": dict(lr=3e-4, dropout=0.2, weight_decay=0.05),
    "small_regularised": dict(d_model=32, d_ff=64, n_layers=1, lr=3e-4, dropout=0.2, weight_decay=0.05),
}


def _fold_job(args):
    """One (arch, lookback, feature_set, val_run, seed) fit. Runs in a worker process."""
    import torch

    torch.set_num_threads(1)
    from .data import RUNS, Preprocessor, TEST_RUN, load_runs, make_windows, split_runs
    from .train import TrainConfig, climatology, fit, metrics, persistence, predict, ridge, route

    arch, lookback, feature_set, val_run, seed, with_baselines, *rest = args
    overrides = rest[0] if rest else {}
    runs = load_runs()
    train_runs, _ = split_runs(runs, [TEST_RUN])
    fit_runs, val = split_runs(train_runs, [val_run])
    pre = Preprocessor.fit(fit_runs, feature_set)
    wtr, wva = make_windows(pre, fit_runs, lookback, HORIZON), make_windows(pre, val, lookback, HORIZON)
    cfg = TrainConfig(arch=arch, lookback=lookback, horizon=HORIZON, seed=seed, **overrides)
    # Nested: the epoch count is chosen on an inner run (the next training run
    # after the held-out one), then the model is refit on all 6 fitting runs for
    # that many epochs. The held-out run is never used to stop training, which
    # is exactly how the final model is trained.
    cycle = [r for r in RUNS if r != TEST_RUN]
    inner = cycle[(cycle.index(val_run) + 1) % len(cycle)]
    inner_fit, inner_val = split_runs(fit_runs, [inner])
    _, hist = fit(cfg, make_windows(pre, inner_fit, lookback, HORIZON),
                  make_windows(pre, inner_val, lookback, HORIZON))
    model, _ = fit(cfg, wtr, epochs=hist["best_epoch"])
    pred = predict(model, wva)
    tag = dict(lookback=lookback, features=feature_set, val_run=val_run, config=overrides.get("name", "base"))
    rows = [dict(model=arch, seed=seed, best_epoch=hist["best_epoch"], **tag, **_flat(metrics(pred, wva)))]
    if arch == "transformer":
        ridge_pred = ridge(wtr, wva)
        # Router: Transformer once a point has been read this run, linear soft sensor before.
        rows.append(dict(model="routed", seed=seed, best_epoch=hist["best_epoch"], **tag,
                         **_flat(metrics(route(pred[..., 1], ridge_pred, wva), wva))))
    if with_baselines:
        for name, pred in [("persistence", persistence(wva)), ("climatology", climatology(wva, pre.point_mean)),
                           ("ridge", ridge_pred)]:
            rows.append(dict(model=name, seed=-1, best_epoch=0, **tag, **_flat(metrics(pred, wva))))
    return rows


def _flat(m: dict) -> dict:
    out = {k: v for k, v in m.items() if not isinstance(v, dict)}
    for k, v in m.items():
        if isinstance(v, dict):
            out.update({f"{k}_{kk}": vv for kk, vv in v.items()})
    return out


def run_cv(jobs, workers):
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return pd.DataFrame([row for rows in pool.map(_fold_job, jobs) for row in rows])


def final_fit(lookback: int, feature_set: str, epochs: int, seeds, overrides: dict, gru_epochs: int) -> dict:
    """Train the seed ensemble on all 7 training runs, evaluate once on the test run."""
    import torch

    from .data import Preprocessor, TEST_RUN, load_runs, make_windows, split_runs
    from .train import (TrainConfig, climatology, cold_quantiles, config_dict, finalize, fit, fit_ridge, metrics,
                        persistence, predict, route)

    runs = load_runs()
    train_runs, test_runs = split_runs(runs, [TEST_RUN])
    pre = Preprocessor.fit(train_runs, feature_set)
    wtr, wte = make_windows(pre, train_runs, lookback, HORIZON), make_windows(pre, test_runs, lookback, HORIZON)
    cfg = TrainConfig(lookback=lookback, horizon=HORIZON, **overrides)
    members, preds = [], []
    for seed in seeds:
        model, _ = fit(replace(cfg, seed=seed), wtr, epochs=epochs)
        members.append(model.state_dict())
        preds.append(predict(model, wte))
    ens = finalize(np.mean(preds, axis=0))
    # Same ensemble size for the GRU, so the backbone comparison is like for like.
    gru_ens = finalize(np.mean([predict(fit(replace(cfg, arch="gru", seed=s), wtr, epochs=gru_epochs)[0], wte)
                                for s in seeds], axis=0))
    bank = fit_ridge(wtr)
    ridge_pred = bank.predict(wte)
    offsets = cold_interval_offsets(train_runs, feature_set, lookback)
    routed = finalize(route(np.mean(preds, axis=0), cold_quantiles(ridge_pred, offsets), wte))
    report = {
        "test_run": TEST_RUN, "lookback": lookback, "features": feature_set, "epochs": epochs,
        "n_members": len(seeds), "horizon_minutes": [round(h * 43 / 60, 1) for h in range(1, HORIZON + 1)],
        "transformer_ensemble": metrics(ens, wte),
        "routed": metrics(routed, wte),
        "transformer_single_seeds": [metrics(p, wte)["rmse_p56"] for p in preds],
        "gru": metrics(gru_ens, wte),
        "ridge": metrics(ridge_pred, wte),
        "persistence": metrics(persistence(wte), wte),
        "climatology": metrics(climatology(wte, pre.point_mean), wte),
    }
    ARTIFACTS.mkdir(exist_ok=True)
    torch.save({"config": config_dict(cfg), "epochs": epochs, "members": members,
                "cold_start": {"coef": bank.coef.astype(np.float32), "intercept": bank.intercept,
                               "offsets": offsets},
                "preprocessor": {"features": pre.features, "mean": pre.mean, "std": pre.std,
                                 "point_mean": pre.point_mean}},
               ARTIFACTS / "co2_transformer.pt")
    # Per-sample predictions for the notebook's analysis.
    np.savez_compressed(RESULTS / "test_predictions.npz", ens=ens, routed=routed, members=np.stack(preds), y=wte.y,
                        mask=wte.mask, last=wte.last, origin=wte.origin,
                        times=wte.times.astype("datetime64[s]").astype(str))
    return report


def cold_interval_offsets(train_runs: dict, feature_set: str, lookback: int) -> np.ndarray:
    """[2, 6] residual quantiles (10 %, 90 %) of the linear soft sensor on cold
    cells, from leave-one-run-out predictions: split-conformal-style intervals
    for the cold-start path, which has no quantile head of its own. The
    preprocessor is refit per fold so the held-out run never shapes its inputs."""
    from .data import Preprocessor, make_windows
    from .train import cold_cells, fit_ridge

    resid = [[] for _ in range(6)]
    for held in train_runs:
        rest = {k: v for k, v in train_runs.items() if k != held}
        pre = Preprocessor.fit(rest, feature_set)
        wtr = make_windows(pre, rest, lookback, HORIZON)
        wva = make_windows(pre, {held: train_runs[held]}, lookback, HORIZON)
        err = wva.y - fit_ridge(wtr).predict(wva)
        sel = wva.mask.astype(bool) & cold_cells(wva)
        for p in range(6):
            resid[p].extend(err[..., p][sel[..., p]])
    return np.array([[np.quantile(r, q) if len(r) else 0.0 for r in resid] for q in (0.1, 0.9)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()
    from .data import RUNS, TEST_RUN, load_runs

    load_runs()  # download once before workers start
    RESULTS.mkdir(exist_ok=True)
    lookbacks = [3, 12, 36] if args.quick else LOOKBACKS
    seeds = SEEDS[:1] if args.quick else SEEDS
    val_runs = [r for r in RUNS if r != TEST_RUN]
    tf_rows = lambda df: df[df.model == "transformer"]

    # 1. Inputs and capacity at a mid lookback (12 steps ≈ one analyser cycle).
    stage1 = run_cv([("transformer", 12, fs, v, s, False, dict(CAPACITY[c], name=c))
                     for fs in ("all", "key") for c in CAPACITY for v in val_runs for s in seeds
                     if fs == "key" or c == "base"], args.workers)
    by_variant = tf_rows(stage1).groupby(["features", "config"])["rmse_p56"].mean()
    best_fs, best_cfg = by_variant.idxmin()
    print(by_variant.round(4).to_string(), f"\nselected features={best_fs} config={best_cfg}", flush=True)
    overrides = dict(CAPACITY[best_cfg], name=best_cfg)

    # 2. Target space × lookback grid for both backbones (log1p target was the
    #    top fix proposed by the root-cause analysis: error scaled with CO2 level).
    variants = {"linear": overrides, "log": dict(overrides, log_target=True, name=f"{best_cfg}+log")}
    stage2 = run_cv([(arch, L, best_fs, v, s, arch == "transformer" and s == seeds[0] and t == "linear",
                      dict(o, name=t)) for arch in ("transformer", "gru") for t, o in variants.items()
                     for L in lookbacks for v in val_runs for s in seeds], args.workers)
    stage2 = stage2.rename(columns={"config": "target"})
    cv = pd.concat([stage1.assign(stage="inputs_capacity"), stage2.assign(stage="target_lookback")], ignore_index=True)
    cv.to_csv(RESULTS / "cv_folds.csv", index=False)
    print(stage2.groupby(["model", "target", "lookback"])["rmse_p56"].mean().unstack().round(4).to_string())

    # 3. Pick (target, lookback) with a paired one-standard-error rule: among
    #    cells whose fold-by-fold gap to the best cell is within one SE, take the
    #    shortest window (then the lower mean). Raw argmin would select noise:
    #    fold-to-fold spread dwarfs the gaps.
    grid = tf_rows(stage2).pivot_table(index=["val_run", "seed"], columns=["target", "lookback"], values="rmse_p56")
    best_raw = grid.mean().idxmin()
    gap = grid.sub(grid[best_raw], axis=0)
    se = gap.std() / np.sqrt(len(gap))
    ok = [c for c in grid.columns if gap[c].mean() <= se[c]]
    chosen = min(ok, key=lambda c: (c[1], grid[c].mean()))
    chosen_t, chosen_L = chosen[0], int(chosen[1])
    overrides = variants[chosen_t]
    cell = lambda arch: stage2[(stage2.model == arch) & (stage2.target == chosen_t) & (stage2.lookback == chosen_L)]
    epochs = int(np.median(cell("transformer")["best_epoch"]))
    gru_epochs = int(np.median(cell("gru")["best_epoch"]))
    print(f"raw best {best_raw}, one-SE choice {chosen}; epochs {epochs} (gru {gru_epochs})", flush=True)

    # 4. Final fit on all training runs, one look at the test run.
    report = final_fit(chosen_L, best_fs, epochs, FINAL_SEEDS[:2] if args.quick else FINAL_SEEDS,
                       overrides, gru_epochs)
    at = stage2[(stage2.target == chosen_t) & (stage2.lookback == chosen_L)
                | (stage2.model.isin(["persistence", "climatology", "ridge"]) & (stage2.lookback == chosen_L))]
    report["selection"] = {
        "features_capacity_cv_rmse_p56": {f"{k[0]}/{k[1]}": float(v) for k, v in by_variant.items()},
        "target": chosen_t,
        "cv_rmse_p56_by_target_lookback": {f"{t}/{int(L)}": float(v) for (t, L), v in grid.mean().items()},
        "paired_gap_to_best": {f"{t}/{int(L)}": float(v) for (t, L), v in gap.mean().items()},
        "paired_se": {f"{t}/{int(L)}": float(v) for (t, L), v in se.items()},
        "raw_best": f"{best_raw[0]}/{int(best_raw[1])}",
        "cv_at_chosen": {m: {k: float(v) for k, v in g.items()} for m, g in
                         at.groupby("model")[["rmse_p56", "rmse_p56_warm", "rmse_p56_cold"]].mean().iterrows()},
        "rule": "shortest lookback (then lowest mean) among (target, lookback) cells within one paired SE of the best",
    }
    (RESULTS / "test_metrics.json").write_text(json.dumps(report, indent=2))
    for k in ("routed", "transformer_ensemble", "gru", "ridge", "persistence", "climatology"):
        print(f"{k:22s} test rmse_p56={report[k]['rmse_p56']:.3f} rmse={report[k]['rmse']:.3f}")


if __name__ == "__main__":
    main()
