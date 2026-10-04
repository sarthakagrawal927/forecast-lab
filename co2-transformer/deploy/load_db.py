"""Load the pilot-plant runs into PostgreSQL (idempotent).

    DATABASE_URL=postgresql://co2:co2@localhost:5432/co2 python -m deploy.load_db [--test-only]

Downloads any missing source files, applies schema.sql, then upserts runs,
tags, sensor readings and analyser readings. Re-running replaces a run's rows.
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import psycopg

from co2tx.data import RUNS, TEST_RUN, load_runs

SCHEMA = Path(__file__).with_name("schema.sql")
KIND = {"FT": "flow", "TT": "temperature", "PT": "pressure", "PD": "pressure", "LT": "level", "AT": "ph"}


def parse_tag(tag: str) -> tuple[str, str | None, str]:
    # A trailing lowercase letter is part of the instrument only before "(":
    # TT110a(0C) -> TT110a, but FT303m3/hr -> FT303 + m3/hr.
    m = re.match(r"([A-Z]+\d+(?:[a-z](?=\())?)\s*\(?([^)]*)\)?", tag)
    instrument, unit = (m.group(1), m.group(2) or None) if m else (tag, None)
    if unit == "0C":
        unit = "degC"
    return instrument, unit, KIND.get(instrument[:2], "other")


def load(conn: psycopg.Connection, run_ids: list[str]) -> None:
    runs = load_runs()
    with conn.cursor() as cur:
        cur.execute(SCHEMA.read_text())
        for run_id in run_ids:
            df = runs[run_id].tz_localize("UTC")  # source stamps are naive; store them as UTC
            tags = [c for c in df.columns if c not in ("co2", "point")]
            cur.executemany(
                "INSERT INTO tags (tag, instrument, unit, kind) VALUES (%s, %s, %s, %s) ON CONFLICT (tag) DO NOTHING",
                [(t, *parse_tag(t)) for t in tags])
            cur.execute("DELETE FROM runs WHERE run_id = %s", (run_id,))
            cur.execute("INSERT INTO runs VALUES (%s, %s, %s, %s)",
                        (run_id, df.index[0], df.index[-1], "test" if run_id == TEST_RUN else "train"))
            long = df[tags].stack().reset_index()
            long.columns = ["ts", "tag", "value"]
            with cur.copy("COPY sensor_readings (run_id, ts, tag, value) FROM STDIN") as copy:
                for ts, tag, value in long.itertuples(index=False):
                    copy.write_row((run_id, ts, tag, float(value)))
            with cur.copy("COPY analyzer_readings (run_id, ts, sampling_point, co2_pct) FROM STDIN") as copy:
                for ts, row in df[["point", "co2"]].iterrows():
                    copy.write_row((run_id, ts, int(row["point"]), float(row["co2"])))
            print(f"loaded {run_id}: {len(df)} steps, {len(long)} sensor rows")
    conn.commit()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-only", action="store_true", help="load only the held-out test run")
    args = ap.parse_args()
    url = os.environ.get("DATABASE_URL", "postgresql://co2:co2@localhost:5432/co2")
    with psycopg.connect(url, options="-c TimeZone=UTC") as conn:
        load(conn, [TEST_RUN] if args.test_only else RUNS)


if __name__ == "__main__":
    main()
