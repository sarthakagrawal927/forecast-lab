# 01 · The benchmarks and the protocol

## Where these datasets come from

The AI-Challenge repository ships download scripts in `src/utils/` for the
standard long-horizon forecasting benchmarks. These are the datasets almost
every Transformer-for-time-series paper since 2021 reports on.

| dataset | what | channels | step | rows | in this lab |
|---|---|---|---|---|---|
| ETTh1, ETTh2 | Electricity transformer: 6 load features + oil temperature (OT) | 7 | 1 h | 17,420 | ETTh1 first |
| ETTm1, ETTm2 | The same transformers at 15-minute resolution | 7 | 15 min | 69,680 | next |
| Weather (WTH) | Meteorological station measurements | 12 or 21, depending on the version | 10 min to 1 h | — | needs a mirror (the challenge script links Google Drive) |
| Electricity (ECL) | Consumption of 321 customers | 321 | 1 h | 26,304 | heaviest |
| yfinance | Live stock prices | — | 5 min | last 60 days only | skipped: a near random walk, and not reproducible |

**Sources:**

- ETT: Zhou et al. 2021, *Informer*: <https://arxiv.org/abs/2012.07436>.
  Data: <https://github.com/zhouhaoyi/ETDataset>.
- Weather and Electricity in their benchmark form: Wu et al. 2021, *Autoformer*:
  <https://arxiv.org/abs/2106.13008>.

## The protocol, and why every detail matters

Our numbers are only comparable with published tables if we follow the same rules.

**1. Chronological splits.**
- ETTh is split 12 / 4 / 4 months (train, validation, test); ETTm uses the same
  months at 15-minute steps.
- The other datasets use 70 / 10 / 20 %.
- Code: `bench/data.py:borders`. Test: `test_ett_hourly_borders_match_reference_loader`.

**2. Lead-in.**
- The validation and test segments start `lookback` steps early, so the first
  forecast's history comes from the previous segment.
- Only targets have to fall inside the segment.

**3. Train-only scaling.**
- Every channel is standardised with the *training* mean and std.
- Errors are reported on that standardised scale, so an MSE of 0.38 means about
  0.38 of the training variance.
- Test: `test_scaler_uses_training_segment_only`.

**4. Multivariate scoring.**
- MSE and MAE are averaged over all 7 channels and all horizon steps.
- Test: `test_test_window_count_matches_published_protocol` (2,785 windows for ETTh1 at L=336, H=96).

**5. Fixed horizons.**
- 96, 192, 336 and 720 steps (4 days to 30 days for hourly ETT).
- A lookback of **336** (two weeks), as in the DLinear and PatchTST papers.

**6. Every test window counts.**
- Many reference loaders used `drop_last=True` on the test set, which silently
  drops the last partial batch. We score every window.
- Read: Qiu et al. 2024, *TFB: Towards Comprehensive and Fair Benchmarking*,
  which documents this and other pitfalls: <https://arxiv.org/abs/2403.20150>.

### Sanity checks that our protocol matches

| check | ours | published |
|---|---|---|
| Naive ("Repeat") ETTh1, H=96 | 1.294 | 1.295 (DLinear paper, Table 2) |
| DLinear ETTh1, H=96 (single seed, reference learning-rate schedule) | 0.377 | 0.375 |

The full published rows, transcribed from the papers' PDFs, are in `results/published_etth1.json`.

If the naive number didn't match, nothing else in the tables would mean anything.

## How this differs from the CO₂ challenge

| | CO₂ absorber | these benchmarks |
|---|---|---|
| Rows | 898 | 17k–70k |
| Unit of generalisation | an unseen operating *run* | an unseen *later period* of the same system |
| Labels | sparse: one point measured per step | dense: every channel at every step |
| Horizon | 0.7–8.6 min (12 steps) | 4–30 days (96–720 steps) |
| Hybrid we used | router: soft sensor for points not yet read | anchor + correction: linear forecast plus Transformer residual |

The CO₂ router has no counterpart here, because every channel is observed at
every step. The general idea carries over: anchor on a strong simple forecast
and learn only the departures from it.
