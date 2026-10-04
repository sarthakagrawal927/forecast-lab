# Study guide: the CO₂ Transformer challenge

A path through everything in this project, written for someone who needs to
*explain* it, not just run it. Budget about 6–8 hours, including the two
external videos and papers that carry the theory.

| # | page | you'll be able to… | time |
|---|---|---|---|
| 1 | [01 · The plant, the analyser and the data](01-problem-and-data.md) | explain the absorber, the multiplexed analyser, and why runs are the unit of evaluation | 45 min |
| 2 | [02 · The Transformer, piece by piece](02-transformer-from-scratch.md) | derive attention, explain every layer in `model.py`, justify the residual quantile head | 3 h with Karpathy and the Annotated Transformer |
| 3 | [03 · Evaluation done right](03-evaluation-done-right.md) | name the leakage traps, explain grouped CV, the one-SE rule and calibration | 1 h |
| 4 | [04 · Results and root causes](04-results-and-root-causes.md) | tell the story of the result in the order it was found | 30 min |
| 5 | `notebooks/analysis.ipynb` | point at the figure that proves each claim | 45 min |
| 6 | [05 · Production](05-production.md) | defend the schema, API, monitoring and container choices | 45 min |
| 7 | [Interview prep](interview-prep.md) | answer the questions a reviewer is likely to ask, in your own words | 45 min |

Reference pages:

- [Decision log](decisions.md): what was decided, on what evidence, and what was rejected.
- [Lessons](lessons.md): the transferable ideas.

## Hands-on checkpoints

Do these with the code open. Each one makes a claim from the docs concrete.

1. `uv run pytest -v`: read each test name and explain *what bug it prevents*.
2. In a Python shell:
   ```python
   from co2tx.data import load_runs
   load_runs()["140207_1"][["point", "co2"]].head(20)
   ```
   Find the analyser cycle and the flushing carry-over.
3. Run the quick experiment:
   ```bash
   uv run python -m co2tx.experiments --quick
   ```
   This overwrites `results/` and `artifacts/`, so restore them afterwards with
   `git checkout -- results artifacts`. Compare its selection with the full run.
4. `docker compose up --build`, then call `/predict` before and after 12:06 on
   `140207_1`. Watch point 6 switch from `cold-start-linear` to `transformer`.
5. Pick one exercise at the bottom of pages 02, 03 or 05 and do it.

## The one-paragraph summary to memorise

A from-scratch Transformer forecasts CO₂ at six absorber points from causal
inputs: carried-forward readings with their age, plus 13 absorber instruments.
It's trained with a masked quantile loss on the real analyser readings. The
model, features, log-scale target and the 3-step window were selected by leave-one-run-out
cross-validation (nested early stopping) with a one-SE rule. Root-cause analysis showed most of the
error was cold start, before a point's first reading. Cross-validation confirmed
that a linear soft sensor handles that case better, so the deployed system
routes between the two. The final test RMSE is 0.30 at points 5–6, 64% better
than persistence. On points already read, persistence remains marginally better,
which the write-up reports next to the result. The system is served via FastAPI over a Postgres historian
schema, with logged predictions for monitoring, all in one `docker compose up`.
