"""Load the Imperial College carbon-capture pilot plant records and build causal,
multi-horizon training windows.

Facts that shape this module (see docs/learning/01-problem-and-data.md):

* One gas analyser (AT400) is switched between six sampling points, so at any
  timestep exactly one point's CO2 is known. ``point`` says which.
* Every file is a separate operating run, sampled every ~43 s. Runs are never
  stitched together: a window never crosses a run boundary.
* Nothing here looks into the future. Unlike the reference notebook, missing
  point readings are carried forward (with their age), not linearly
  interpolated between past and future samples.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

N_POINTS = 6
CO2_COL = "AT400(CO2 %)"
TEST_RUN = "140207_1"  # recommended by the challenge
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
SOURCE_URL = (
    "https://github.com/tonyzyl/CO2-Soft-sensor-for-a-carbon-capture-pilot-plant"
    "/raw/main/data/withLabel/{run}.xlsx"
)
RUNS = ["140120_1", "140206_1", "140207_1", "140207_2",
        "140214_1", "140214_2", "140227_1", "140313_1"]

# Absorber-column instruments named in the reference notebook's P&ID notes.
KEY_FEATURES = [
    "FT103(kg/hr)", "FT104(kg/hr)", "TT210(0C)", "TT211(0C)", "FT301m3/hr",
    "FT302(kg/hr)", "FT303m3/hr", "FT304(kg/hr)", "PT402(barg)", "PT403(barg)",
    "TT104(0C)", "TT304(0C)", "PT111(barg)",
]
AGE_CAP = 24.0  # steps; ages beyond ~1.4 analyser cycles carry no extra signal


def download(raw_dir: Path = RAW_DIR) -> None:
    """Fetch any missing run files from the original public repository."""
    import urllib.request

    raw_dir.mkdir(parents=True, exist_ok=True)
    for run in RUNS:
        target = raw_dir / f"{run}.xlsx"
        if not target.exists():
            urllib.request.urlretrieve(SOURCE_URL.format(run=run), target)


def load_run(path: Path) -> pd.DataFrame:
    """One run as a frame indexed by timestamp with sensor columns + co2 + point."""
    df = pd.read_excel(path, sheet_name=0, index_col=0, header=[0, 1])
    df.columns = df.columns.map("".join)
    df = df.rename(columns={df.columns[-1]: "point", CO2_COL: "co2"})
    df.index.name = "time"
    df["point"] = df["point"].astype(int)
    return df


def load_runs(raw_dir: Path = RAW_DIR) -> dict[str, pd.DataFrame]:
    """All eight runs, cached as a pickle next to the raw files (Excel parsing is slow)."""
    cache = raw_dir.parent / "runs.pkl"
    if cache.exists():
        return pd.read_pickle(cache)
    if not all((raw_dir / f"{r}.xlsx").exists() for r in RUNS):
        download(raw_dir)
    runs = {r: load_run(raw_dir / f"{r}.xlsx") for r in RUNS}
    pd.to_pickle(runs, cache)
    return runs


def observation_state(co2: np.ndarray, point: np.ndarray, fill: np.ndarray):
    """Causal per-point state at each step: last seen value, its age, seen flag.

    ``fill`` (shape [6]) stands in for points not yet observed in this run.
    Returns last [T,6], age [T,6] (steps since observed, capped), seen [T,6].
    """
    T = len(co2)
    last = np.tile(fill, (T, 1)).astype(np.float32)
    age = np.full((T, N_POINTS), AGE_CAP, dtype=np.float32)
    seen = np.zeros((T, N_POINTS), dtype=np.float32)
    cur_last, cur_age, cur_seen = fill.astype(np.float32).copy(), np.full(N_POINTS, AGE_CAP), np.zeros(N_POINTS)
    for t in range(T):
        cur_age = np.minimum(cur_age + 1, AGE_CAP)
        p = point[t] - 1
        cur_last[p], cur_age[p], cur_seen[p] = co2[t], 0.0, 1.0
        last[t], age[t], seen[t] = cur_last, cur_age, cur_seen
    return last, age, seen


@dataclass
class Preprocessor:
    """Fitted on training runs only, then applied unchanged to anything else."""

    features: list[str]
    mean: np.ndarray
    std: np.ndarray
    point_mean: np.ndarray  # per-point training mean CO2 (fill + climatology baseline)
    names: list[str] = field(default_factory=list)

    @classmethod
    def fit(cls, runs: dict[str, pd.DataFrame], feature_set: str = "all") -> "Preprocessor":
        frame = pd.concat(runs.values())
        if feature_set == "key":
            features = [c for c in KEY_FEATURES if c in frame.columns]
        else:
            candidates = [c for c in frame.columns if c not in ("co2", "point")]
            std = frame[candidates].std()
            features = [c for c in candidates if std[c] > 1e-9]  # drop dead tags
        x = frame[features].to_numpy(np.float64)
        point_mean = frame.groupby("point")["co2"].mean().reindex(range(1, 7)).to_numpy()
        return cls(features, x.mean(0), x.std(0) + 1e-6, point_mean)

    def step_features(self, df: pd.DataFrame) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Per-step model input [T, F] plus the raw arrays later stages need."""
        co2 = df["co2"].to_numpy(np.float32)
        point = df["point"].to_numpy(np.int64)
        sensors = ((df[self.features].to_numpy(np.float64) - self.mean) / self.std).astype(np.float32)
        sensors = np.clip(sensors, -8, 8)  # one bad tag must not dominate attention
        last, age, seen = observation_state(co2, point, self.point_mean)
        onehot = np.eye(N_POINTS, dtype=np.float32)[point - 1]
        x = np.concatenate([sensors, last, age / AGE_CAP, seen, onehot], axis=1)
        self.names = (self.features + [f"last_p{i}" for i in range(1, 7)]
                      + [f"age_p{i}" for i in range(1, 7)] + [f"seen_p{i}" for i in range(1, 7)]
                      + [f"at_p{i}" for i in range(1, 7)])
        return x, {"co2": co2, "point": point, "last": last}


@dataclass
class Windows:
    """Model-ready samples. N samples, lookback L, F features, H horizons."""

    x: np.ndarray        # [N, L, F] left-padded with zeros
    pad: np.ndarray      # [N, L] True where padded
    last: np.ndarray     # [N, 6] last observed CO2 per point at the origin step
    y: np.ndarray        # [N, H, 6] target CO2 (0 where unobserved)
    mask: np.ndarray     # [N, H, 6] 1 where the analyser measured that point
    run: np.ndarray      # [N] run id per sample
    origin: np.ndarray   # [N] index of the forecast origin inside its run
    times: np.ndarray    # [N] origin timestamp


def window_at(feats: np.ndarray, t: int, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    """The ``lookback`` steps ending at ``t`` (inclusive), left-padded with zeros.

    Training and the API both call this, so the model sees identical inputs.
    """
    lo = max(0, t - lookback + 1)
    n = t - lo + 1
    win = np.zeros((lookback, feats.shape[1]), np.float32)
    pad = np.ones(lookback, bool)
    win[lookback - n:], pad[lookback - n:] = feats[lo:t + 1], False
    return win, pad


def make_windows(pre: Preprocessor, runs: dict[str, pd.DataFrame], lookback: int, horizon: int) -> Windows:
    """Every step of every run is a forecast origin; targets are steps t+1..t+H.

    Left padding means every lookback sees the same set of origins and targets,
    so lookback comparisons are apples to apples.
    """
    xs, pads, lasts, ys, masks, run_ids, origins, times = [], [], [], [], [], [], [], []
    for run, df in runs.items():
        feats, raw = pre.step_features(df)
        T = feats.shape[0]
        for t in range(T - 1):  # the last step has no future target
            win, pad = window_at(feats, t, lookback)
            y = np.zeros((horizon, N_POINTS), np.float32)
            m = np.zeros((horizon, N_POINTS), np.float32)
            for h in range(1, horizon + 1):
                if t + h < T:
                    p = raw["point"][t + h] - 1
                    y[h - 1, p], m[h - 1, p] = raw["co2"][t + h], 1.0
            xs.append(win); pads.append(pad); lasts.append(raw["last"][t])
            ys.append(y); masks.append(m); run_ids.append(run); origins.append(t)
            times.append(df.index[t])
    return Windows(np.stack(xs), np.stack(pads), np.stack(lasts), np.stack(ys),
                   np.stack(masks), np.array(run_ids), np.array(origins), np.array(times))


def split_runs(runs: dict[str, pd.DataFrame], test_runs: list[str]):
    train = {k: v for k, v in runs.items() if k not in test_runs}
    test = {k: v for k, v in runs.items() if k in test_runs}
    return train, test
