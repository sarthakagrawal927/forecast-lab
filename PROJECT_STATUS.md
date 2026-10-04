# forecast-lab — PROJECT STATUS

> **REACTIVATED 2026-10-04** (was parked 2026-07-04) for a concrete hiring
> need: the Applied Computing AI-Challenge, solved in `co2-transformer/`. The
> other labs stay as they were; only the new subfolder is in active work.

Last updated: 2026-10-04 (co2-transformer: AI-Challenge solution + study guide)

## Why / What

**forecast-lab is an eval-first ML learning lab, not a sellable product.** One
through-line across forecasting and recommendation: the best method depends on
the data regime, so measure everything and never trust a model that doesn't
beat the dumb baseline on held-out data. The thesis was seeded when the
original event-forecast model **collapsed to a constant on real data** and only
the eval harness caught it — the harness, and the lessons it produces, are the
real asset.

**Users:** the repo owner (learner) and fleet agents. No external users.

**In scope:**
- Learning + eval harnesses: honest temporal/held-out benchmarks for
  forecasting (demand-forecast) and recommenders (recsys-lab).
- Distilled findings as docs: `TUTORIAL.md` guided path,
  `demand-forecast/docs/EXPLAINER.md`, per-lab `results.md` leaderboards,
  `docs/learning/new-things.md` study queue.
- The event-forecast Rust service as local/research code and origin story.

**Out of scope:**
- A production forecasting product, deployment, auth, or paying users.
- Shipping any model that doesn't beat its naive/popularity baseline.

Four sub-projects share the philosophy, not the code:

| Folder | What | Language |
|---|---|---|
| `demand-forecast/` | forecasting explainer: methods × dataset regimes (taxi · Olist · bike · Rossmann · M5) × eval gates | Python |
| `recsys-lab/` | recommender benchmark ladder: popularity → item-KNN → ALS → BPR → Markov → SASRec on MovieLens-1M | Python |
| `event-forecast/` | where it started — next-event forecaster whose model collapsed on real data (has its own `PROJECT_STATUS.md`) | Rust |
| `co2-transformer/` | AI-Challenge: from-scratch PyTorch Transformer forecasting CO₂ in a carbon-capture absorber, analysis notebook with root-cause analysis, Postgres + FastAPI + Docker Compose deployment, study guide | Python |

## Dependencies

### External
- **Python labs** (`demand-forecast/`, `recsys-lab/`): numpy, scipy, pandas,
  scikit-learn (demand-forecast); torch optional for SASRec (recsys-lab).
  Managed per-lab via `pyproject.toml` + `uv.lock`, pytest for tests.
- **event-forecast:** Rust + Rocket 0.5, sqlx, TimescaleDB/PostgreSQL via
  local Docker Compose (:54329), Leaflet single-file explorer UI. Full list
  in `event-forecast/PROJECT_STATUS.md`.
- **Datasets:** gitignored. The corrected bike runner fetches a pinned licensed
  UCI input; other historical experiments may need separately obtained datasets.

### Internal (fleet)
- None at runtime. Fleet standards (AGENTS.md at fleet root) govern process.
- No production Cloudflare surface; `package.json` `deploy` script fails
  closed by design (CI is the release gate for this lab repo).

## Timeline

- **2026-07-03** — Added CSV upload web UI (`web/`) wrapping the method-ladder report: upload a time-series CSV → naive → seasonal-naive → moving average → Holt-Winters → ensemble, with held-out backtest metrics (MAE/RMSE/wMAPE/bias), forecast chart, metrics table, and CSV/JSON download. Vite + React 19 + Tailwind v4. Consulting wedge: if forecast-lab ever becomes more than a learning lab, this is the upload-your-CSV entry point.
- **pre-2026-06-21** — sub-projects built in their own repos: event-forecast
  (Rust service, phases 1–5; 2026-06-20 eval found the baseline ties/loses to
  a majority-class guesser on real data), recsys-lab, demand-forecast.
- **2026-06-21** — consolidated the eval-first ML exploration into one repo;
  added shared GBT helper + guided learning tutorial (`TUTORIAL.md`).
- **2026-06-24** — event-forecast perf pass (hashset/hashmap) (#1).
- **2026-06-28** — added fleet learning track (`docs/learning/new-things.md`).
- **2026-07-02** — explicit deploy guard (no production deploy path).
- **2026-10-04** — reactivated; added `co2-transformer/` (AI-Challenge). Nested
  leave-one-run-out CV selected a log-scale target and a 3-step lookback; root-cause
  analysis found cold start (points not yet read in a run) dominated early error, leading
  to a CV-validated router (Transformer + linear cold-start soft sensor). Test RMSE at
  points 5–6: 0.295 routed / 0.312 Transformer alone vs 0.829 persistence. Honest limit:
  on points already read, persistence stays marginally better in CV. Study path in
  `co2-transformer/docs/learning/`.

## Products

- **No production deployments.** Everything runs locally.
- **GitHub repo:** `sarthakagrawal927/forecast-lab` (public), CI on GitHub
  Actions (`.github/workflows/ci.yml`): `cargo test` for event-forecast +
  pytest matrix for the Python labs.
- **Local surfaces:**
  - `demand-forecast/query.py` — pick area × product × time → count + odds +
    stocking quantity; `viz.py` → `report.html` (six-chart story).
  - `recsys-lab/recommend.py` — real top-10 recommendation lists;
    `results.md` leaderboard.
  - `event-forecast` — Rocket API on :8088 + Leaflet explorer + `evaluate` /
    `load_events` CLI binaries (see its `PROJECT_STATUS.md`).
  - `co2-transformer` — `docker compose up` → Postgres + loader + FastAPI on :8000
    (local only; no hosted deployment). See its `README.md`.
- **Docs as product:** `TUTORIAL.md` (run-it-in-order learning path),
  `demand-forecast/docs/EXPLAINER.md`, per-lab `docs/lessons.md`.

## Features (shipped)

### demand-forecast (Python)
- Baselines + models: naive, seasonal-naive, moving average, Holt/ETS,
  gradient boosting (lags + calendar + exogenous), pooled GBT + shape-based
  clustering, naive/model ensemble.
- Probabilistic layer: empirical / Poisson / Negative-Binomial quantiles;
  split + adaptive **online conformal** calibration; pinball-loss scoring.
- Decision layer: newsvendor cost-optimal stocking; online adaptive loop.
- Five dataset regimes with recorded findings (`results.md`, `docs/lessons.md`):
  NYC taxi (LastWeek lag-168h beats GBT — no exogenous signal), bike-sharing
  (weather signal → ML +50%), Rossmann (promo → +15%), M5 (intermittent —
  naive holds), Olist (sparsity/pooling); granularity dial: wMAPE 0.70 → 0.08
  by aggregating.
- Query surface (`query.py`), report generator (`viz.py` → `report.html`),
  pytest suite, per-run results JSON.

### recsys-lab (Python)
- Six-model ladder on MovieLens-1M: Popularity, ItemKNN, ALS, BPR,
  Markov (1st-order), SASRec (2-block causal transformer).
- Honest eval: leave-one-out temporal split, **full-ranking** (no sampled
  negatives), optimistic tie handling, Recall/NDCG/MRR/Coverage.
- Recorded leaderboard (`results.md`): SASRec NDCG@10 +388% vs popularity;
  sequence models dominate order-blind MF/KNN; undertrained SASRec came
  last — the eval catches under-training too.
- `recommend.py` top-10 surface, BPR sweep script, pytest suite, learning
  curriculum (`docs/learning/roadmap.md`, `metrics.md`).

### event-forecast (Rust)
- Full inventory lives in `event-forecast/PROJECT_STATUS.md` (kept as the
  canonical status for that sub-project): Rocket JSON API, transparent
  transition/median baseline, TimescaleDB ingest, heatmap/anomaly/decision/
  replay endpoints, Leaflet explorer, fixture tests.
- Its lasting contribution here is the **negative result**: the baseline
  ties/loses to a majority-class guesser on real data; the eval harness is
  what survived.

### Repo-level
- `TUTORIAL.md` guided learning path across all three labs.
- `docs/learning/new-things.md` study queue (7 stubs, `Why here:` left TBD
  for the learner per fleet learning-track standard).
- CI for both languages; deploy script that fails closed.

## Todo / Planned / Deferred / Blocked

### Planned
1. Fill `Why here:` stubs in `docs/learning/new-things.md` after studying
   each topic (owner: user — per the learning-track standard, agents must
   not pre-fill rationale).
2. recsys-lab curriculum growth per `docs/learning/roadmap.md`: fill
   remaining `Source:` links and add a short lessons entry per implemented
   tier.
3. event-forecast planned items tracked in its own `PROJECT_STATUS.md`
   (real-data-volume validation gate, `augurs` benchmark spike, first real
   event source, alert profile editor).

### Deferred
- Any production deployment or productization — this is a learning lab;
  the deploy guard enforces it.
- event-forecast deferrals (ML frameworks, persisted alerts, geocoding,
  ROADMAP phases 6+) — see `event-forecast/PROJECT_STATUS.md`.

### Blocked
- event-forecast accuracy validation blocked on a larger real stream
  (sample is 14 events / 4 entities — fixture/demo quality only).

## Reproducibility and task reconciliation — 2026-09-07

The existing hourly bike experiment now fetches checksum-pinned CC BY 4.0 input,
uses timestamp-aligned lags and a chronological held-out evaluation, and labels
observed target-hour weather as an oracle. Two local runs produced byte-identical
metrics and predictions. Seven protocol/metric tests passed. See the
[measured receipt](demand-forecast/docs/bike-reproducibility.md) and
[repair #4](https://github.com/sarthakagrawal927/forecast-lab/issues/4).

The planned/deferred/blocked items above are preserved in
[#5](https://github.com/sarthakagrawal927/forecast-lab/issues/5) and summarized in
the README. They are not complete and the portfolio hold has not been lifted.
Historical dataset scores above have not all been requalified by the bike run.
