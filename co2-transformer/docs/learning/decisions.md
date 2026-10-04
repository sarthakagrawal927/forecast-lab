# Decision log

Each entry records a decision, the evidence behind it, and what was rejected.
Dates are 2026-10-04 unless noted.

### D1 · Task framing: multi-horizon forecasting with causal inputs
- **Decided:** forecast all 6 points at t+1…t+12 from history up to t, scored only
  on measured cells.
- **Why:** the challenge asks for forecasting. Interpolated inputs (as in the
  reference notebook) would leak future readings.
- **Rejected:** reproducing the reference soft-sensor setup with interpolated labels.

### D2 · Inputs: carry-forward + age, not interpolation
- **Why:** causal. Age tells the model how stale each value is.
- **Guard:** `test_observation_state_is_causal`.

### D3 · Masked pinball loss on real readings only
- **Why:** one label per step is the truth. Inventing the other five adds bias.
- **Rejected:** MSE on interpolated targets.

### D4 · Residual-on-persistence head, zero-initialised
- **Why:** start at the strongest naive forecaster and learn the departures from it.
- **Known weakness (found later):** the anchor is wrong when a point hasn't been
  read yet. D11 handles this.

### D5 · Leave-one-run-out cross-validation; the test run is used only for final scoring
- **Why:** runs are separate regimes, and a random row split leaks neighbouring rows.

### D6 · 13 absorber instruments instead of all 88 tags
- **Evidence:** CV RMSE 1.01 vs 1.12 (nested CV, Transformer rows only).

### D7 · Base capacity (d=64, 2 layers, 4 heads)
- **Evidence:** the smaller and more regularised variants were all worse in CV (1.05–1.06 vs 1.01).
- **Negative result kept on record:** the bottleneck is the shift between runs, not capacity.

### D8 · Log target + lookback of 3 steps, chosen with the paired one-SE rule
- **How:** a grid of target scale (linear, log) × lookback (3–36 steps), scored in
  nested CV. Among cells within one paired SE of the best, the rule takes the
  shortest window.
- **Result:** log/3 is the raw best, and the other cells are about one SE or more
  behind. In an earlier, linear-only run the raw best was 36 steps and the rule
  picked 3. Both times the rule guarded against choosing noise.

### D9 · Final model trained on all 7 runs
- **How:** 5-seed ensemble, epochs set to the median nested-CV best epoch (8). The
  learning-rate schedule matches CV (D16).
- **Why:** there's no data left to early-stop on, and the ensemble reduces sensitivity to that choice.

### D10 · Warm/cold error split added to all metrics
- **Trigger:** test-run root-cause analysis (ridge beat the Transformer).
- **Why:** the two kinds of cells are different tasks, and a blended number hid
  that different models win each one.

### D11 · Route cold points to a linear soft sensor
- **Hypothesis source:** the test-run RCA (disclosed).
- **Adoption evidence:** 7-run CV only.
  - Ridge is best on cold cells.
  - The router beats the Transformer alone overall. Final: 0.877 vs 0.953.
- **Cold-point intervals:** split-conformal style, from ridge's leave-one-run-out
  residuals, with the preprocessor refit per fold.
- **Test contact:** the pipeline prints test scores on every full run, and it ran
  about six times while the method evolved. No choice was made from a test number
  except that the router *idea* came from the test-run RCA. Final test: 0.295.
- **Open refinement:** per-point routing. At point 5 the Transformer is slightly
  better on cold cells.

### D12 · CPU-only PyTorch in Docker; analysis libraries kept out of the image
- **Why:** the server has no GPU. The cold-start model is stored as plain NumPy
  arrays, so serving doesn't need scikit-learn.

### D13 · Narrow historian schema, UTC everywhere, a predictions log
- **Why:** this is how plant data is actually stored, and the predictions log
  makes monitoring possible. See [05-production.md](05-production.md).

### D14 · Clip forecasts at 0%, identically everywhere
- **Why:** it's a physical constraint.
- **How:** `finalize()` sorts the quantiles, then clips. It's used by evaluation,
  the analysis and serving alike, so the parity check is exact. The review caught
  an earlier version that clipped only in serving.

### D15 · Nested early stopping in CV (from the independent review)
- **Problem:** CV used to early-stop each network on the same run it then scored,
  which flattered the networks against the baselines.
- **Fix:** the epoch is picked on an inner run, then the model is refit on the 6
  fitting runs and scored.
- **Effect:** the networks' CV error rose (Transformer 0.85 → 1.04), and
  persistence became the best warm-cell model. Reported, not hidden.

### D16 · Final-fit learning-rate schedule matches CV
- **Problem:** the cosine schedule spanned only the final epoch count, so the
  learning rate decayed to zero in about 5 epochs. CV used a 150-epoch span.
- **Fix:** the schedule always spans `max_epochs`, and training stops at the chosen epoch.

### D17 · Epoch-0 "do no harm" check
- **What:** early stopping also scores the untrained model, which is exactly persistence.
- **Kept as a safeguard.** In practice inner-run epochs rarely transfer as "no
  training", so it didn't change the results.

### D18 · Warm-only training: tried, not adopted
- **Result:** it improved warm-cell error, but stayed behind persistence in paired
  CV (+0.026 ± 0.007 at best), and it hurt the standalone model.

### D19 · Log-scale target (adopted via D8's grid)
- **Rationale:** RC3, error scaled with the CO₂ level.
- **Effect:** the Transformer improved from 1.04 to 0.95 in CV, the best single
  model, and its cold-cell error dropped sharply.

### D20 · GRU compared as a 5-seed ensemble with the same target
- **Why:** like-for-like with the Transformer (review finding).

### D21 · Model version = config + artifact hash
- **Why:** `/monitoring` filters by version and de-duplicates repeated forecasts,
  so a retrained model is never scored together with its predecessor.

## Things deliberately not done

| idea | why not (yet) |
|---|---|
| Tuning on the test run | Invalidates the only honest number |
| A pretrained or foundation model (TimesFM, Chronos) | The challenge requires a self-implemented model; it's still worth a comparison later |
| Kinetic-model fusion as in the reference paper | Needs MATLAB/Simulink outputs; listed as the top data improvement |
| A larger hyperparameter search | CV spread (±0.3–0.4) dwarfs the gaps between configs; more search would fit noise |
| Shipping persistence + soft sensor instead | It is marginally better in CV and on test, but the challenge asks for a Transformer service; the gap is reported prominently and choosing the warm-cell estimator by CV is next step #2 |
