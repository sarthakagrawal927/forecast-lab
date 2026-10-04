"""FastAPI service: CO2 forecasts for the absorber's six sampling points.

    uvicorn deploy.api:app --host 0.0.0.0 --port 8000      # docs at /docs

Endpoints
    GET  /health                       model + database status
    GET  /runs                         runs available in the database
    POST /predict                      forecast from one origin timestamp
    POST /predict/window               rolling forecasts for every origin in [start, end]
    GET  /monitoring/{run_id}          live accuracy + interval coverage of logged forecasts
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from datetime import datetime

import numpy as np
import pandas as pd
import psycopg
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from co2tx.serving import Ensemble

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://co2:co2@localhost:5432/co2")
MAX_WINDOW_ORIGINS = 500
state: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    state["model"] = Ensemble()
    yield


app = FastAPI(title="CO2 absorber forecaster", version="1.0", lifespan=lifespan)


class PredictRequest(BaseModel):
    run_id: str = Field(examples=["140207_1"])
    timestamp: datetime = Field(description="Forecast origin; the latest analyser step at or before it is used.",
                                examples=["2014-02-07T12:30:00"])
    log: bool = Field(True, description="Store the forecast in the predictions table for monitoring.")


class WindowRequest(BaseModel):
    run_id: str = Field(examples=["140207_1"])
    start: datetime = Field(examples=["2014-02-07T12:00:00"])
    end: datetime = Field(examples=["2014-02-07T12:30:00"])
    horizon_step: int = Field(1, ge=1, le=12, description="Which horizon to return per origin (1 step ≈ 43 s).")
    log: bool = False


def connect():
    return psycopg.connect(DATABASE_URL, options="-c TimeZone=UTC")


def utc(ts: datetime) -> pd.Timestamp:
    """Naive request times mean UTC, matching how the loader stores the data."""
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def run_history(conn, run_id: str, until: datetime | None = None) -> pd.DataFrame:
    """Wide frame (one row per analyser step) from the run's start up to ``until``."""
    until = utc(until) if until is not None else None
    tags = state["model"].required_tags
    with conn.cursor() as cur:
        cur.execute("SELECT ts, sampling_point, co2_pct FROM analyzer_readings "
                    "WHERE run_id = %s AND (%s::timestamptz IS NULL OR ts <= %s) ORDER BY ts",
                    (run_id, until, until))
        analyzer = cur.fetchall()
        if not analyzer:
            raise HTTPException(404, f"no analyser data for run {run_id!r} at or before {until}")
        cur.execute("SELECT ts, tag, value FROM sensor_readings "
                    "WHERE run_id = %s AND tag = ANY(%s) AND (%s::timestamptz IS NULL OR ts <= %s)",
                    (run_id, tags, until, until))
        sensors = cur.fetchall()
    wide = pd.DataFrame(sensors, columns=["ts", "tag", "value"]).pivot(index="ts", columns="tag", values="value")
    missing = [t for t in tags if t not in wide.columns]
    if missing:
        raise HTTPException(422, f"run {run_id!r} has no readings for model tags {missing[:5]}")
    base = pd.DataFrame(analyzer, columns=["ts", "point", "co2"]).set_index("ts")
    frame = base.join(wide[tags], how="left")
    # A real historian drops samples. Gaps are filled causally: carry the last
    # good value forward; before any value exists, use the training mean. Never
    # back-fill, which would copy a later reading into an earlier window.
    pre = state["model"].pre
    frame[tags] = frame[tags].ffill().fillna(pd.Series(pre.mean, index=pre.features))
    return frame


def log_forecast(conn, run_id: str, fc, version: str):
    rows = fc.records()
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO predictions (model_version, run_id, origin_ts, horizon_step, target_ts, sampling_point,"
            " p10, p50, p90, ensemble_std) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            [(version, run_id, fc.origin, r["horizon_step"], r["target_time"], r["sampling_point"],
              r["p10"], r["p50"], r["p90"], r["ensemble_std"]) for r in rows])
    conn.commit()


@app.get("/health")
def health():
    model = state.get("model")
    try:
        with connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM analyzer_readings")
            db = {"ok": True, "analyzer_rows": cur.fetchone()[0]}
    except Exception as exc:  # report, don't crash the probe
        db = {"ok": False, "error": type(exc).__name__}
    return {"model": model.version if model else None, "database": db}


@app.get("/runs")
def runs():
    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT r.run_id, r.started_at, r.ended_at, r.split, count(a.ts) "
                    "FROM runs r LEFT JOIN analyzer_readings a USING (run_id) GROUP BY r.run_id ORDER BY r.started_at")
        return [dict(zip(["run_id", "started_at", "ended_at", "split", "steps"], row)) for row in cur.fetchall()]


@app.post("/predict")
def predict(req: PredictRequest):
    model: Ensemble = state["model"]
    with connect() as conn:
        history = run_history(conn, req.run_id, req.timestamp)
        fc = model.forecast(history)
        if req.log:
            log_forecast(conn, req.run_id, fc, model.version)
    return {"model_version": model.version, "run_id": req.run_id, "origin": fc.origin.isoformat(),
            "units": "CO2 % (vol)", "interval": "p10-p90 (80%)", "forecasts": fc.records()}


@app.post("/predict/window")
def predict_window(req: WindowRequest):
    """Rolling forecasts: for each analyser step in [start, end] forecast
    ``horizon_step`` ahead, and attach what the analyser later measured."""
    model: Ensemble = state["model"]
    with connect() as conn:
        full = run_history(conn, req.run_id)
        idx = np.where((full.index >= utc(req.start)) & (full.index <= utc(req.end)))[0]
        if len(idx) == 0:
            raise HTTPException(404, "no analyser steps in that window")
        if len(idx) > MAX_WINDOW_ORIGINS:
            raise HTTPException(422, f"window spans {len(idx)} origins; max is {MAX_WINDOW_ORIGINS}")
        out = []
        for t in idx:
            fc = model.forecast(full.iloc[: t + 1])
            if req.log:
                log_forecast(conn, req.run_id, fc, model.version)
            h = req.horizon_step
            target = t + h
            measured = None
            if target < len(full):
                p = int(full["point"].iloc[target])
                measured = {"sampling_point": p, "co2_pct": float(full["co2"].iloc[target]),
                            "time": full.index[target].isoformat()}
            out.append({"origin": fc.origin.isoformat(),
                        "forecast": [r for r in fc.records() if r["horizon_step"] == h],
                        "measured": measured})
    return {"model_version": model.version, "run_id": req.run_id, "horizon_step": req.horizon_step, "rows": out}


@app.get("/monitoring/{run_id}")
def monitoring(run_id: str, model_version: str | None = None):
    """Join logged forecasts to the analyser reading taken nearest their target
    time at the forecast point: RMSE, MAE, bias and 80 % interval coverage.
    Coverage drifting well below 0.8 is the cue to retrain or recalibrate.
    Scores one model version (default: the one being served); a forecast
    logged more than once counts once (latest wins)."""
    version = model_version or state["model"].version
    sql = """
        WITH latest AS (
            SELECT DISTINCT ON (origin_ts, horizon_step, sampling_point) *
            FROM predictions
            WHERE run_id = %s AND model_version = %s
            ORDER BY origin_ts, horizon_step, sampling_point, created_at DESC, id DESC
        )
        SELECT p.horizon_step, p.sampling_point, p.p10, p.p50, p.p90, a.co2_pct
        FROM latest p
        JOIN LATERAL (
            SELECT co2_pct, ts FROM analyzer_readings a
            WHERE a.run_id = p.run_id AND a.sampling_point = p.sampling_point
              AND a.ts BETWEEN p.target_ts - interval '22 seconds' AND p.target_ts + interval '22 seconds'
            ORDER BY abs(extract(epoch FROM a.ts - p.target_ts)) LIMIT 1
        ) a ON true
    """
    with connect() as conn, conn.cursor() as cur:
        cur.execute(sql, (run_id, version))
        rows = cur.fetchall()
    if not rows:
        return {"run_id": run_id, "model_version": version, "matched": 0}
    df = pd.DataFrame(rows, columns=["h", "point", "p10", "p50", "p90", "y"])
    err = df["p50"] - df["y"]
    by_h = df.assign(se=err ** 2).groupby("h")["se"].mean().pow(0.5).round(4).to_dict()
    return {"run_id": run_id, "model_version": version, "matched": len(df), "rmse": round(float(np.sqrt((err ** 2).mean())), 4),
            "mae": round(float(err.abs().mean()), 4), "bias": round(float(err.mean()), 4),
            "coverage_80": round(float(((df["y"] >= df["p10"]) & (df["y"] <= df["p90"])).mean()), 4),
            "rmse_by_horizon": {int(k): v for k, v in by_h.items()}}
