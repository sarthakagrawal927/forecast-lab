# 04 · Results and root causes, in the order they were found

The notebook (`notebooks/analysis.ipynb`) holds the final numbers and figures.
This page tells the *story*, because the order of discovery is the lesson. The
numbers in Acts 1–4 come from earlier versions of the pipeline and are kept here
as history. If a final number here and the notebook ever disagree, the notebook wins.

## Act 1: a respectable first result

- **Inputs:** causal features, a masked loss, and leave-one-run-out cross-validation.
- **Cross-validation result:** the Transformer looks best (0.85 RMSE at points 5–6)
  against persistence at 1.17.
- **Test result:** the Transformer beats persistence by 35%, but **ridge regression
  beats the Transformer** (0.43 vs 0.54), even though ridge was the least stable
  model in cross-validation.

**The tempting moves**, all of them wrong:

- Tune against the test run.
- Switch to ridge, choosing on one run against seven runs of evidence.
- Hide the result.

## Act 2: hypotheses that failed

| hypothesis | test | verdict |
|---|---|---|
| The network is too big for about 800 samples | Smaller and regularised variants in CV | ✗ every variant was worse |
| The hard runs have unusual operating inputs | Correlate each run's distance in instrument space with its fold error | ✗ correlation ≈ 0 |
| The intervals are badly calibrated | CV coverage | ✗ close to the nominal 0.80 |

## Act 3: slice by state, and the picture flips

Split every error by **whether the target point had already been read in this
run** at the forecast origin (first version of the model):

| test RMSE (points 5–6) | all | warm (already read) | cold (not yet read) |
|---|---|---|---|
| Transformer v1 | 0.54 | **0.25** | 1.32 |
| Ridge | **0.43** | 0.43 | **0.40** |
| Persistence | 0.83 | 0.26 | 2.16 |

**Root cause RC2, cold start:**

- Before a point's first reading, its "last value" is a training-mean fill: 5.63%
  at point 6, while this run sits around 3.2%.
- Cold cells were **14% of cells but 81% of the v1 Transformer's squared error**.
- Ridge's whole test advantage came from cold cells.

**The action:** route cold points to a linear soft sensor. The idea came from the
test run. It was adopted because the 7-run CV showed the same pattern
independently. That disclosure is in the decision log and the notebook.

## Act 4: an independent review changes the numbers

A reviewer, an agent briefed to distrust the results, found that **CV early-stopped
each network on the same run it then scored**. The baselines get no such help,
so the networks looked better than they were. Other findings: a learning-rate
schedule mismatch in the final fit, a mixed-up average in stage 1, inconsistent
clipping between serving and evaluation, and an unequal GRU comparison.

After the fixes, with nested early stopping (the epoch is picked on an inner run):

- **The networks' CV error rose** (the Transformer went from 0.85 to 1.04).
- **On warm cells, persistence became the best model.** That's the most important
  and least comfortable finding in the project.

## Act 5: two principled attempts, judged in CV only

| attempt | rationale | CV verdict |
|---|---|---|
| Train only on warm cells | In the routed system the network only answers warm cells | Warm error improved, but still behind persistence (+0.026 ± 0.007); not adopted |
| **Log-scale target** | RC3: error scaled with the CO₂ level | **Adopted** by the pipeline's own target × lookback grid. The Transformer improved from 1.04 to **0.95** and became the best single model |

## Final results (from the notebook)

| | CV (7 runs) | test | test, warm | test, cold |
|---|---|---|---|---|
| **Routed: Transformer + soft sensor (delivered)** | **0.877** | **0.295** | 0.274 | 0.402 |
| Transformer alone (log target, L=3) | 0.953 | 0.312 | 0.274 | 0.486 |
| GRU, same head (5 seeds) | 0.983 | 0.350 | 0.253 | 0.703 |
| Ridge | 1.094 | 0.430 | 0.434 | 0.402 |
| Persistence | 1.166 | 0.829 | 0.258 | 2.155 |
| *Reference: persistence + soft sensor* | *0.835* | *0.282* | *0.258* | *0.402* |

**How to read it:**

- The Transformer is the best *single* model in CV and on test, and its intervals
  are well calibrated (80% coverage: 0.74 in CV, 0.89 on test).
- The router improves on it.
- On points already read, persistence is still slightly more accurate than any
  network. So the reference system, persistence plus the soft sensor, edges
  ahead in both CV and test. Say so plainly.

## Remaining root causes

**RC1/RC2: point 6 and cold start**
- **Evidence:** cold cells still cause about a third of the final Transformer's
  squared error, down from 81% in v1.
- **What would fix it:** warm-start each run from the previous run, gate forecasts,
  or train with randomly hidden points.

**RC3: the CO₂ level**
- **Evidence:** warm-state fold error correlates with the held-out run's point-6
  level (r ≈ 0.9).
- **What would fix it:** the log target is a partial fix; more high-CO₂ runs and
  physics features are the real one.

**RC4: analyser flushing**
- **Evidence:** later readings within a point-6 visit are harder, plus carry-over
  (point 1 reads 0.29% right after point 6).
- **What would fix it:** use the dwell position as an input, or model the sample-line dynamics.

**RC5: a calm test run**
- **Evidence:** test errors are well below the CV average, and a sister run was
  recorded the same afternoon.
- **What would fix it:** nothing is broken; validate on a new, harder run before
  claiming more.

**RC6: warm cells don't beat persistence**
- **Evidence:** nested CV, paired.
- **What would fix it:** more runs, so the network can learn plant dynamics that
  transfer between runs.

## What to say in one breath

> "I built the Transformer from scratch and selected everything by
> leave-one-run-out CV. It's the best single model I tested: 62% better than
> persistence on the held-out run, with calibrated intervals. Slicing errors by
> state showed most of the early error was cold start, which a linear soft
> sensor handles better, so the deployed system routes between them. An
> independent review then caught optimistic early stopping in my CV. With
> nested early stopping, plain persistence is still marginally better on points
> already read. I report that next to my result, because with 7 runs that's
> where the data runs out."
