# CO₂ absorber forecasting with a from-scratch Transformer

Solution to the [Applied Computing AI-Challenge](https://github.com/appliedcomputingtech/AI-Challenge). It
forecasts CO₂ concentration at the six sampling points of the absorber in Imperial College's
carbon-capture pilot plant, 0.7–8.6 minutes ahead.

| part | deliverable |
|---|---|
| 1 · Model | A Transformer encoder written in PyTorch without `nn.Transformer*` or `nn.MultiheadAttention` ([`co2tx/model.py`](co2tx/model.py)). It has a residual-on-persistence quantile head, trains with a masked loss on real analyser readings, and is selected by leave-one-run-out cross-validation over lookback windows. |
| 2 · Analysis | [`notebooks/analysis.ipynb`](notebooks/analysis.ipynb), executed with outputs: results, interpretability, root-cause analysis |
| 3 · Deployment | Postgres schema in historian format, a FastAPI service with monitoring, and `docker compose up` |

## Results (test run `140207_1`, held out from every decision)

RMSE in CO₂ percentage points at points 5–6, scored only where the analyser measured. Points 1–4 sit near 0%.
CV is leave-one-run-out over the 7 training runs, with nested early stopping.

| model | test RMSE | test skill vs persistence | CV RMSE (7 runs) |
|---|---|---|---|
| **Routed: Transformer + cold-start soft sensor (deployed)** | **0.295** | **64%** | **0.877** |
| Transformer alone (log target, 5-seed ensemble) | 0.312 | 62% | 0.953 |
| GRU with the same head (5-seed ensemble) | 0.350 | 58% | 0.983 |
| Ridge on the same inputs | 0.430 | 48% | 1.094 |
| Persistence (last reading) | 0.829 | 0% | 1.166 |
| *Reference: persistence + soft sensor* | *0.282* | *66%* | *0.835* |

- **Selected time window:** a lookback of 3 steps (about 2 minutes) with a log-scale target. It was chosen
  jointly by a paired one-standard-error rule over a target × lookback grid.
- **Horizons:** skill over persistence rises over the first ~2 minutes, then holds across the whole
  0.7–8.6 minute range.
- **Uncertainty:** the 80% intervals cover 74% of readings in CV and 89% on the test run.
- **Honest limit:**
  - On points the analyser has already read this run, plain persistence is still slightly more accurate
    than either network on unseen runs.
  - That's why the reference row edges ahead.
  - The Transformer's value is in cold start (as a single model) and in calibrated uncertainty.
- **Root causes:** see notebook §8 and [04-results-and-root-causes.md](docs/learning/04-results-and-root-causes.md).
  The biggest early error source was cold start (points not yet read in a run). The disclosed history
  of test-set contact is there too, along with an independent review that tightened the CV.

## Reproduce

Requires [uv](https://docs.astral.sh/uv/). Data downloads automatically from the
[original repository](https://github.com/tonyzyl/CO2-Soft-sensor-for-a-carbon-capture-pilot-plant/tree/main/data/withLabel).

```bash
uv sync                                       # Python 3.12, PyTorch, analysis tools
uv run python -m co2tx.experiments            # CV + selection + final fit (~360 small network fits, run in parallel)
uv run pytest                                 # 15 tests: causality, masking, padding, serving parity, routing
cd notebooks && uv run jupyter lab analysis.ipynb
```

- `experiments` writes `results/` (CV folds, test metrics, predictions) and `artifacts/co2_transformer.pt`.
  Those committed files are what the notebook and the API use.
- Runs are deterministic on CPU: re-running reproduces the committed numbers exactly.

## Deploy

```bash
docker compose up --build        # db → loader (one-shot) → api on :8000
open http://localhost:8000/docs  # interactive API docs
```

```bash
curl -s localhost:8000/predict -H 'content-type: application/json' \
  -d '{"run_id": "140207_1", "timestamp": "2014-02-07T12:30:00"}'
curl -s localhost:8000/predict/window -H 'content-type: application/json' \
  -d '{"run_id": "140207_1", "start": "2014-02-07T12:00:00", "end": "2014-02-07T13:20:00", "horizon_step": 3, "log": true}'
curl -s localhost:8000/monitoring/140207_1   # live RMSE / bias / interval coverage of logged forecasts
```

Each forecast row carries p10/p50/p90, the spread across ensemble members, and which estimator produced
it (`transformer` or `cold-start-linear`).

- **Ports:** the defaults are 5432 and 8000. Override them with `POSTGRES_PORT` and `API_PORT`.
- **Verified:** the live API reproduces the offline test predictions at all 117 test origins, to within its
  4-decimal rounding.
- **Monitoring:** `/monitoring` scores only the served model version (a content hash) and counts each
  logged forecast once.

## Layout

```
co2tx/
  data.py         load runs, causal observation state, windows, masks
  model.py        Transformer (from scratch), GRU twin, quantile residual head, pinball loss
  train.py        training loop, metrics (warm/cold split), baselines, router
  experiments.py  leave-one-run-out CV → selection → final ensemble → artifacts
  serving.py      inference wrapper shared by the API
  analysis.py     notebook helpers (error tables, permutation importance, attention)
deploy/           schema.sql, load_db.py, api.py
notebooks/        analysis.ipynb (Part 2)
docs/learning/    study guide: concepts → evaluation → results → production → interview prep
results/, artifacts/   committed outputs of co2tx.experiments
```

## Learn it

[`docs/learning/`](docs/learning/README.md) is a study path through everything here: the plant, the Transformer,
evaluation, root causes, production, the decision log, and questions a reviewer is likely to ask.
