-- Time-series sensor log for the carbon-capture pilot plant.
--
-- Shape follows a plant historian: one narrow row per (run, timestamp, tag)
-- instead of a 91-column wide table, so adding an instrument is an INSERT into
-- `tags`, not an ALTER TABLE. The gas analyser gets its own table because its
-- reading is only meaningful together with the sampling point it was switched to.

CREATE TABLE IF NOT EXISTS runs (
    run_id      TEXT PRIMARY KEY,               -- source file name, e.g. 140207_1
    started_at  TIMESTAMPTZ NOT NULL,
    ended_at    TIMESTAMPTZ NOT NULL,
    split       TEXT NOT NULL CHECK (split IN ('train', 'test'))
);

CREATE TABLE IF NOT EXISTS tags (
    tag         TEXT PRIMARY KEY,               -- column name as exported, e.g. FT103(kg/hr)
    instrument  TEXT NOT NULL,                  -- FT103
    unit        TEXT,                           -- kg/hr
    kind        TEXT NOT NULL                   -- flow | temperature | pressure | level | ph | other
);

CREATE TABLE IF NOT EXISTS sensor_readings (
    run_id  TEXT NOT NULL REFERENCES runs (run_id) ON DELETE CASCADE,
    ts      TIMESTAMPTZ NOT NULL,
    tag     TEXT NOT NULL REFERENCES tags (tag),
    value   DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (run_id, ts, tag)
);

CREATE TABLE IF NOT EXISTS analyzer_readings (
    run_id          TEXT NOT NULL REFERENCES runs (run_id) ON DELETE CASCADE,
    ts              TIMESTAMPTZ NOT NULL,
    sampling_point  SMALLINT NOT NULL CHECK (sampling_point BETWEEN 1 AND 6),
    co2_pct         DOUBLE PRECISION NOT NULL CHECK (co2_pct >= 0),
    PRIMARY KEY (run_id, ts)
);

-- Every served forecast is logged, so accuracy and calibration can be
-- monitored later by joining against analyzer_readings (see GET /monitoring).
CREATE TABLE IF NOT EXISTS predictions (
    id              BIGSERIAL PRIMARY KEY,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    model_version   TEXT NOT NULL,
    run_id          TEXT NOT NULL REFERENCES runs (run_id) ON DELETE CASCADE,
    origin_ts       TIMESTAMPTZ NOT NULL,
    horizon_step    SMALLINT NOT NULL,
    target_ts       TIMESTAMPTZ NOT NULL,
    sampling_point  SMALLINT NOT NULL,
    p10             DOUBLE PRECISION NOT NULL,
    p50             DOUBLE PRECISION NOT NULL,
    p90             DOUBLE PRECISION NOT NULL,
    ensemble_std    DOUBLE PRECISION NOT NULL
);
CREATE INDEX IF NOT EXISTS predictions_run_target ON predictions (run_id, target_ts);
