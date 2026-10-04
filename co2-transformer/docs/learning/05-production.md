# 05 · From notebook to a running service

Part 3 of the challenge: Postgres, a FastAPI model server, and Docker Compose.
This page explains the design choices. The README has the commands.

```
docker compose up --build
   db      postgres:16       schema.sql applied by the loader
   loader  one-shot          download 8 runs → long-format rows → COPY → exit 0
   api     uvicorn :8000     waits for the loader to succeed, loads the 5-member ensemble + cold-start model
```

## Database: a plant-historian schema (`deploy/schema.sql`)

| table | grain | why |
|---|---|---|
| `runs` | one row per operating run | Run boundaries matter: windows never cross them |
| `tags` | one row per instrument (tag, instrument, unit, kind) | Metadata lives in data, not in column names |
| `sensor_readings` | (run, ts, tag) → value | **Long/narrow format**, like real historians (OSIsoft PI, AVEVA). A new instrument is an INSERT, not an ALTER TABLE |
| `analyzer_readings` | (run, ts) → sampling point, CO₂ | The CO₂ value only means something together with the point it came from |
| `predictions` | one row per served (origin, horizon, point) | Logs what the model said, so accuracy can be checked later |

Design notes worth saying out loud:

- **Primary key `(run_id, ts, tag)`.** It's unique, it doubles as the index the API
  uses (run plus time range), and re-loading is idempotent: the loader deletes a
  run and re-inserts it.
- **`COPY` instead of row-by-row INSERTs** for the ~80k sensor rows; this is Postgres's bulk-load path.
  Read: <https://www.postgresql.org/docs/current/populate.html>.
- **UTC everywhere.** The source timestamps have no time zone, so the loader stores
  them as UTC and the API reads naive request times as UTC. Without this, a laptop
  in IST and a container in UTC disagree by 5.5 hours.
- **At plant scale:** add TimescaleDB hypertables (the same SQL, with automatic time
  partitioning and compression) and continuous aggregates.
  Read: <https://docs.timescale.com/use-timescale/latest/hypertables/>.

## Model serving (`deploy/api.py`, `co2tx/serving.py`)

| endpoint | purpose |
|---|---|
| `GET /health` | model version plus a DB probe; used by the Docker healthcheck |
| `GET /runs` | what data is available |
| `POST /predict` | forecast from one origin timestamp: 12 horizons × 6 points × p10/p50/p90 |
| `POST /predict/window` | rolling forecasts across a time window, each paired with what the analyser later measured |
| `GET /monitoring/{run}` | joins logged forecasts to the readings that came later: RMSE, bias, 80% coverage by horizon |

The properties that make this production-shaped rather than a demo:

1. **No training/serving skew.** The API rebuilds history from Postgres and calls
   the *same* `Preprocessor.step_features` and `window_at` as training.
   - A unit test guards this, plus an end-to-end check: live API forecasts equal
     the offline test predictions to within the API's 4-decimal rounding.
   - Read: Google, *Rules of ML*, rule #29: <https://developers.google.com/machine-learning/guides/rules-of-ml>.
2. **The artifact is self-describing.** `artifacts/co2_transformer.pt` holds the
   config, ensemble weights, scaler statistics, feature list and cold-start model.
   You can't load weights with the wrong preprocessing.
3. **Routing by state.**
   - Each forecast row says which estimator produced it: `estimator` and
     `point_read_this_run`.
   - Points not yet read in this run use the linear soft sensor (see RC2 in
     [04](04-results-and-root-causes.md)).
   - Consumers can see which numbers are less certain.
4. **Uncertainty in every response.**
   - **p10/p90** cover data noise (aleatoric uncertainty).
   - **`ensemble_std`**, the spread between the 5 members, covers model uncertainty
     (epistemic).
   - A rise in `ensemble_std` is an early signal of inputs unlike the training data.
5. **Physical constraints.** Quantiles are sorted, then clipped at 0% CO₂.
6. **Monitoring that measures the right thing.**
   - Forecasts are logged and later joined to the actual readings, so you can
     watch rolling RMSE and **interval coverage** over time.
   - A coverage drop well below 0.8, or a sustained bias, is the trigger to retrain
     or recalibrate.
   - Read: Chip Huyen, *Designing ML Systems*, chapters 8–9 on data distribution
     shifts and monitoring.
7. **Startup order.**
   - Compose waits for Postgres to pass its healthcheck, then for the loader to
     **exit successfully**, and only then starts the API.
   - The API has its own healthcheck.
   - Read: <https://docs.docker.com/compose/how-tos/startup-order/>.

## The container (`Dockerfile`)

- **One image, two roles** (loader and API). It installs dependencies before
  copying the code, so code edits don't reinstall PyTorch.
- **`uv sync --frozen`** installs exactly what's in the lockfile.
  Read: <https://docs.astral.sh/uv/guides/integration/docker/>.
- **CPU-only PyTorch on Linux.** `pyproject.toml` routes `torch` to the
  `download.pytorch.org/whl/cpu` index for Linux only. That's about 650 MB instead
  of several GB of CUDA libraries a CPU server would never use. On a Mac,
  development still uses the normal wheel with Apple-GPU (MPS) support.
- **Analysis-only libraries stay out of the image.** scikit-learn, matplotlib and
  Jupyter are dev-only. The cold-start model is saved as plain NumPy arrays, so
  serving doesn't need scikit-learn.
- **Runs as a non-root user**, with a `HEALTHCHECK`.

## What I'd add before a real plant

| gap | approach |
|---|---|
| Streaming ingestion | An OPC UA or MQTT subscriber writing to `sensor_readings`; forecast on every analyser step |
| Auth and rate limits | API keys or mTLS behind a gateway; never expose Postgres |
| Model registry | Versioned artifacts (MLflow or a bucket), `model_version` already logged per prediction, shadow-deploy new versions and compare in `/monitoring` |
| Retraining | Scheduled, or triggered by coverage/bias alerts; gated on beating the current model in leave-one-run-out CV |
| Input validation | Range checks per tag (the P&ID valid ranges in [01](01-problem-and-data.md)); flag frozen or out-of-range sensors before they reach the model |
| Online calibration | Conformal recalibration of the interval width from the `predictions` table |
| Batch window endpoint | `/predict/window` runs one forward pass per origin; batch all origins into one tensor for long windows |

## Exercises

1. Kill the DB container while the API is running. What does `/health` report, and why is that the right behaviour for a liveness probe versus a readiness probe?
2. Write the SQL for "hourly RMSE at point 6 over the last day" against `predictions` joined to `analyzer_readings`.
3. Why is `predictions` indexed on `target_ts` and not only on `origin_ts`?
