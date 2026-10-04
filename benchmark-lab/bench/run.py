"""Run the whole ladder on one dataset and write results/<dataset>.{json,md}.

    uv run python -m bench.run --dataset ETTh1
    uv run python -m bench.run --dataset ETTh1 --horizons 96 --seeds 0   # quick

Every choice (epochs, blend weight) is made on the validation segment; the test
segment is only scored.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .data import SEASON, make_splits
from .models import DLinear, Hybrid, PatchTransformer, StepTransformer, naive, seasonal_naive
from .train import fit, predict, scores, targets

RESULTS = Path(__file__).resolve().parent.parent / "results"
# Training settings follow each paper's ETTh1 setup, with a common epoch cap.
LINEAR = dict(lr=5e-3, batch=32, max_epochs=20, patience=3, halve_lr=True)  # DLinear reference schedule
TRANSFORMER = dict(lr=1e-4, batch=128, max_epochs=60, patience=8)
ORDER = ["naive", "seasonal_naive", "dlinear", "step_transformer", "patch_transformer", "hybrid", "blend"]


def run_horizon(name: str, lookback: int, horizon: int, seeds: list[int]) -> dict:
    train, val, test, cols = make_splits(name, lookback, horizon)
    C, season = len(cols), SEASON.get(name, 24)
    y_val, y_test = targets(val), targets(test)
    out: dict[str, list] = {m: [] for m in ORDER}

    for label, fn in [("naive", lambda h: naive(h, horizon)),
                      ("seasonal_naive", lambda h: seasonal_naive(h, horizon, season))]:
        out[label].append({"test": scores(predict(fn, test), y_test), "val": scores(predict(fn, val), y_val)})

    for seed in seeds:
        record = lambda m, model, info, t0: out[m].append({
            "seed": seed, "test": scores(predict(model, test), y_test), "val": {"mse": info["val_mse"]},
            "best_epoch": info["best_epoch"], "seconds": round(time.time() - t0, 1)})

        t0 = time.time()
        lin, info = fit(DLinear(lookback, horizon), train, val, seed=seed, **LINEAR)
        record("dlinear", lin, info, t0)

        t0 = time.time()
        step, info = fit(StepTransformer(lookback, horizon, C), train, val, seed=seed, **TRANSFORMER)
        record("step_transformer", step, info, t0)

        t0 = time.time()
        patch, info = fit(PatchTransformer(lookback, horizon, C), train, val, seed=seed, **TRANSFORMER)
        record("patch_transformer", patch, info, t0)

        t0 = time.time()
        hyb, info = fit(Hybrid(lin, PatchTransformer(lookback, horizon, C, zero_head=True)), train, val,
                        seed=seed, **TRANSFORMER)
        record("hybrid", hyb, info, t0)

        # Blend: w·linear + (1−w)·patch, w chosen on validation (grid of 0.1).
        lv, pv, lt, pt = predict(lin, val), predict(patch, val), predict(lin, test), predict(patch, test)
        grid = np.round(np.linspace(0, 1, 11), 1)
        w = float(grid[np.argmin([scores(g * lv + (1 - g) * pv, y_val)["mse"] for g in grid])])
        out["blend"].append({"seed": seed, "weight_linear": w, "test": scores(w * lt + (1 - w) * pt, y_test),
                             "val": scores(w * lv + (1 - w) * pv, y_val)})
        print(f"  H={horizon} seed={seed} done", flush=True)
    return {"n_test_windows": len(test), "n_val_windows": len(val), "models": out}


def summarise(res: dict) -> dict:
    table = {}
    for H, r in res["horizons"].items():
        for m in ORDER:
            runs = r["models"][m]
            mse = [x["test"]["mse"] for x in runs]
            mae = [x["test"]["mae"] for x in runs]
            table.setdefault(m, {})[H] = {"mse": float(np.mean(mse)), "mse_std": float(np.std(mse)),
                                          "mae": float(np.mean(mae)), "mae_std": float(np.std(mae))}
    return table


def markdown(name: str, res: dict) -> str:
    t = res["summary"]
    Hs = list(res["horizons"])
    lines = [f"# {name} — test MSE / MAE (standardised, all channels; mean over seeds {res['seeds']})", "",
             f"Lookback {res['lookback']}. Lower is better; **bold** = best per horizon.", "",
             "| model | " + " | ".join(f"H={h} MSE | H={h} MAE" for h in Hs) + " |",
             "|---|" + "---|---|" * len(Hs)]
    best = {h: min(t[m][h]["mse"] for m in ORDER) for h in Hs}
    for m in ORDER:
        cells = []
        for h in Hs:
            v = t[m][h]
            mse = f"{v['mse']:.3f}" + (f" ±{v['mse_std']:.3f}" if v["mse_std"] > 0 else "")
            cells += [f"**{mse}**" if v["mse"] == best[h] else mse, f"{v['mae']:.3f}"]
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ETTh1")
    ap.add_argument("--lookback", type=int, default=336)
    ap.add_argument("--horizons", type=int, nargs="+", default=[96, 192, 336, 720])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--tag", default="", help="suffix for result files (e.g. a quick run)")
    args = ap.parse_args()
    res = {"dataset": args.dataset, "lookback": args.lookback, "seeds": args.seeds,
           "linear_cfg": LINEAR, "transformer_cfg": TRANSFORMER, "horizons": {}}
    for H in args.horizons:
        print(f"{args.dataset} H={H}", flush=True)
        res["horizons"][str(H)] = run_horizon(args.dataset, args.lookback, H, args.seeds)
    res["summary"] = summarise(res)
    RESULTS.mkdir(exist_ok=True)
    stem = args.dataset + (f"-{args.tag}" if args.tag else "")
    (RESULTS / f"{stem}.json").write_text(json.dumps(res, indent=1))
    (RESULTS / f"{stem}.md").write_text(markdown(args.dataset, res))
    print(markdown(args.dataset, res))


if __name__ == "__main__":
    main()
