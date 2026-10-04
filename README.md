# forecast-lab

One project, one through-line: **eval-first ML** — across forecasting and
recommendation, the best method depends on the data regime, so *measure
everything* and never trust a model that doesn't beat the dumb baseline on
held-out data.

**Start with a reproducible bounded experiment:**

```bash
cd demand-forecast
uv sync --frozen --extra dev
uv run --frozen python run_bike.py
uv run --frozen pytest -q
```

This downloads a checksum-pinned, CC BY 4.0 UCI Bike Sharing CSV and evaluates
three naive baselines and gradient boosting on a chronological held-out tail.
It writes metrics/provenance and per-hour predictions under the ignored
`demand-forecast/data/bike-evaluation/` directory. See
[the protocol and measured receipt](demand-forecast/docs/bike-reproducibility.md).

Apart from `co2-transformer/`, the project remains a paused learning lab. This experiment does not establish
commerce-demand accuracy, multi-step forecast performance, or B2B readiness.
The broader [TUTORIAL.md](TUTORIAL.md) describes historical experiments;
its other datasets and numeric claims have not been requalified by this check.

The documented bike commands also passed from a fresh, unauthenticated public
source download on 2026-09-07. Source-code licensing is still undecided; the UCI
dataset license does not license this repository's code.

It grew as one continuous exploration, in three parts (kept as clean parallel
subfolders — they share the *philosophy*, not the code: forecasting metrics ≠
ranking metrics ≠ a Rust service):

| Folder | What | Language |
|---|---|---|
| **[demand-forecast/](demand-forecast/)** | the culmination — a forecasting **explainer**: methods × dataset regimes (taxi · Olist · bike · Rossmann · M5) × eval gates. Quantile odds, newsvendor inventory, clustering, a query surface, and `report.html`. **Start at [docs/EXPLAINER.md](demand-forecast/docs/EXPLAINER.md).** | Python |
| **[recsys-lab/](recsys-lab/)** | the recommender benchmark ladder — popularity → item-KNN → ALS → BPR → Markov → SASRec on MovieLens-1M, honest full-ranking eval. | Python |
| **[event-forecast/](event-forecast/)** | where it started — a next-event forecaster whose model collapsed on real data, which seeded the eval-first thesis. | Rust |
| **[benchmark-lab/](benchmark-lab/)** | the challenge's benchmark datasets (ETT, Weather, Electricity) run through baselines → DLinear → two Transformers → hybrids under the published protocol; ETTh1 done. | Python |
| **[co2-transformer/](co2-transformer/)** | the Applied Computing AI-Challenge — a from-scratch PyTorch Transformer forecasting CO₂ in a carbon-capture absorber, with root-cause analysis, a Postgres + FastAPI + Docker deployment and a [study guide](co2-transformer/docs/learning/README.md). **Active since 2026-10-04.** | Python |

## The one-sentence synthesis

**Signal is always the lever; the regime decides whether it's enough — match the
method to the regime, and measure everything.**

(Datasets are gitignored. The bike entry point fetches its own input; other historical scripts may require separately obtained datasets.)

## Models to consider

- [google/timesfm-3.0-pytorch](https://huggingface.co/google/timesfm-3.0-pytorch) — considered for ML forecasting models (from issue #2)

## Remaining work after task reconciliation

The bounded bike repair is tracked in [#4](https://github.com/sarthakagrawal927/forecast-lab/issues/4).
The paused queue is retained in [#5](https://github.com/sarthakagrawal927/forecast-lab/issues/5):
owner-authored study rationales, recommender sources/lessons, a suitable larger
event stream and baseline evaluation before an augurs spike, the optional alert
profile editor, and further demand validation. The real-data and domain gates
are unresolved; the event fixture and this bike result do not complete them.
Production, persisted alerts, geocoding and later roadmap phases remain deferred
product decisions. No open pull requests were present at the 2026-09-07 review.
