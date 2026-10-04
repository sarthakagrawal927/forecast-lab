# 03 · Evaluation done right on 898 rows

With this little data the model is the easy part. The hard part is not fooling
yourself. This page covers the evaluation mechanisms, why each exists, and where
each one lives. The numbers are in `notebooks/analysis.ipynb` and `results/`.

**Background reading:**

- **Hyndman & Athanasopoulos, *Forecasting: Principles and Practice* (3rd ed.)**,
  chapter 5 on evaluation and time-series cross-validation: <https://otexts.com/fpp3/>.
- **Kapoor & Narayanan 2023, "Leakage and the reproducibility crisis in ML-based
  science"**: a taxonomy of leakage that you should be able to name in an
  interview. <https://reproducible.cs.princeton.edu/>

---

## 1. Leakage you must avoid here

**Interpolation look-ahead**
- **What it is:** filling a point's value between two readings uses the *later* reading.
- **Our guard:** carry-forward plus age. `test_observation_state_is_causal`
  changes the future data and asserts that past features don't move.

**Random row splits**
- **What it is:** neighbouring rows, 43 s apart, end up in both train and validation,
  so validation measures memorisation.
- **Our guard:** whole runs are held out, and windows never cross run boundaries.

**Scaler fit on test**
- **What it is:** normalising with statistics that include the test run.
- **Our guard:** `Preprocessor.fit` only ever sees the training runs, and API
  serving reuses those saved statistics.

**Selection on test**
- **What it is:** picking the lookback, features or epochs by looking at test scores.
- **Our guard:** every choice is made in leave-one-run-out cross-validation, and
  the test run is scored once at the end.

**Training/serving skew**
- **What it is:** the API builds features differently from training.
- **Our guard:** both paths call `window_at`. `test_serving_features_match_training_features`
  checks this, and live API forecasts match offline forecasts to 4 decimal places.

## 2. Leave-one-run-out cross-validation (grouped CV)

- **What:** each of the 7 training runs takes a turn as the validation set; the model
  trains on the other 6. That's 7 folds, times 3 seeds.
- **Why here:** runs are independent operating regimes. A forecaster that only
  works on runs it has seen is useless at a new operating point.
- **Source:** scikit-learn's `GroupKFold` and `LeaveOneGroupOut` docs explain grouped CV:
  <https://scikit-learn.org/stable/modules/cross_validation.html#group-k-fold>.
- **Code:** `co2tx/experiments.py:_fold_job`.

## 3. Baselines: the floor every model must clear

| baseline | what it says | why include it |
|---|---|---|
| Persistence | "the next reading equals the last reading of that point" | The honest floor for a slow process. Many published models fail to beat it. |
| Climatology | "each point is at its training-run average" | Shows how much the run-to-run level shift costs |
| Ridge on the same window | A linear model with exactly the same inputs | If a linear model matches the network, the network isn't earning its complexity |
| GRU with the same head | A recurrent backbone instead of attention | Isolates the Transformer as the only difference |

- **Source:** FPP3 §5.2, "Some simple forecasting methods".
- **Code:** `co2tx/train.py`, functions `persistence`, `climatology`, `ridge`.

## 4. Score only what was measured

- **What:** `metrics()` computes errors on measured cells only, and reports points
  5–6 separately (`rmse_p56`).
- **Why:** points 1–4 sit near 0%. Including them averages tiny errors into the
  score and flatters every model.

## 5. Warm vs cold cells

This was found during the root-cause analysis. Split every error by whether the
target point **had already been read in this run** at the forecast origin.

- **Warm cells** are real forecasting: we know where the point was and predict
  where it goes.
- **Cold cells** are really *soft sensing*: the point hasn't been read yet, so the
  model must infer its level from plant instruments alone.

On short isolated runs, cold cells are a small share of cells but most of the
squared error. A single blended number hid the fact that different models win
on the two tasks. The metrics now report both (`rmse_p56_warm`, `rmse_p56_cold`).

**Lesson:** always slice the error by the system's *state*, not just by horizon.

## 6. Choosing among nearly equal options: the paired one-SE rule

- **What:** for each fold and seed, compute the gap between each lookback and the
  best lookback. Then choose the **shortest** window whose mean gap is within one
  standard error of that paired gap.
- **Why paired:** folds differ far more than lookbacks do. Hard runs are hard for
  every window. Pairing removes that shared difficulty, so the comparison measures
  the lookback alone.
- **Why one-SE:** among models that are statistically indistinguishable, prefer
  the simplest. This is the classic rule from Breiman et al. (CART) and from
  *The Elements of Statistical Learning*, §7.10: <https://hastie.su.domains/ElemStatLearn/>.
- **Code:** `experiments.py`, stage 3. The rule is written into `results/test_metrics.json`.

## 7. Nested early stopping, and picking epochs without a validation set

Inside each CV fold:

1. The epoch count is chosen on an **inner** run, the next training run.
2. The model is refit on all 6 fitting runs.
3. Only then is it scored on the held-out run.

If you early-stop on the run you score, you are quietly selecting on that run.
The baselines get no such help, so the comparison flatters the network. An
independent review caught exactly this in the first version. Fixing it made the
networks look worse and the comparison honest (decision D15).

For the final model there's no data left to stop on:

The final model trains on all 7 runs, so there's nothing left to early-stop on.
Instead:

- Use the **median best epoch** from cross-validation.
- Average a 5-seed ensemble to smooth out how sensitive the model is to that number.

This is standard practice when the data is too small to hold anything back.

## 8. Calibration: are the intervals honest?

- **Coverage:** the share of readings that fall inside the 80% interval.
  - About 0.8 is honest.
  - Well below 0.8 is overconfident.
  - Well above 0.8 is too cautious.
- **Our result:** cross-validation coverage is about 0.74, slightly
  overconfident across unseen runs. On the calm test run it is about 0.89,
  slightly cautious.
- **In production:** track coverage live (the API's `/monitoring`) and recalibrate
  with conformal prediction if it drifts.
- **Source:** Angelopoulos & Bates, *A Gentle Introduction to Conformal
  Prediction*: <https://arxiv.org/abs/2107.07511>.

## Exercises

1. Re-run the pipeline with a random 80/20 row split instead of leaving runs out.
   How much better does the model *appear*?
2. Compute the persistence score with the reference notebook's interpolated
   labels. Why does it look much better, and why is that misleading?
3. Change the lookback rule to the raw minimum. Which window is chosen, and does
   the test result change by more than the spread between seeds?
