# Interview prep: questions a reviewer is likely to ask

Answer from understanding, not from memory. If you can't explain an answer in
your own words, re-read the linked page. Every claim here comes from the code
or the notebook.

## About the problem

**Q: What exactly are you predicting?**
CO₂ (in %) at each of the six absorber sampling points, 1–12 analyser steps
ahead, which is 43 s to 8.6 min. The inputs are only data available at forecast
time. → [01](01-problem-and-data.md)

**Q: Why not use the interpolated columns like the reference notebook?**
Interpolating between two readings uses the later reading, which is up to about
9 minutes in the future. That's fine for a retrospective soft-sensor study but
not for forecasting. I carry the last reading forward and give the model its
age. About 70% of steps would otherwise carry future information. → notebook §2

**Q: Only one point is measured per step. How do you train on that?**
I use a masked loss: each training window contributes only the cells the
analyser actually measured. There are no invented labels. → `pinball_loss`

## About the model

**Q: Walk me through the Transformer.**
1. A linear embedding per step, plus sinusoidal positions.
2. Two pre-LayerNorm encoder blocks with hand-written multi-head attention and a padding mask.
3. Pool the last token together with the masked mean.
4. A head that predicts, per horizon and point, a correction to the last reading
   plus p10/p90 offsets.

→ [02](02-transformer-from-scratch.md)

**Q: Why predict a residual over persistence?**
Persistence is a strong baseline for a slow process. With a zero-initialised
head, the untrained model *is* persistence, so learning only has to capture the
departures from it. The weakness is cold start, and I found and fixed it. → [04](04-results-and-root-causes.md)

**Q: Why quantiles?**
Operators need a range. The pinball loss trains each quantile directly, and the
softplus offsets guarantee p10 ≤ p50 ≤ p90. Interval coverage then becomes a
monitoring metric.

**Q: Is the Transformer better than an LSTM/GRU here?**
Slightly, but not significantly. With the same inputs, head and ensemble size,
the CV scores are 0.95 vs 0.98 and the test scores are 0.31 vs 0.35. The gains
come from how the problem is framed (causal state, masked quantile loss,
residual head, log target), not from attention. Zeng et al. (2022) found the
same on many benchmarks.

**Q: Does your Transformer beat persistence?**
Overall, yes, clearly: 62% lower error on the test run, and 0.95 vs 1.17 in CV.
On points the analyser has already read, no. In nested CV, persistence is
slightly better there (0.75 vs 0.80). The Transformer's advantage comes from
points not yet read, and from calibrated uncertainty. With 7 runs it can't learn
plant dynamics that beat "it stays where it was" on an unseen run. I report a
"persistence + soft sensor" reference next to my system for that reason.

**Q: What does the attention learn?**
With a 3-step window, it learns close to uniform smoothing. The long memory is
in the carried-forward state features, which is why longer windows didn't help.
→ notebook §7

## About evaluation

**Q: How do you know you're not overfitting?**
- Leave-one-run-out cross-validation (7 folds × 3 seeds).
- The test run was scored only after every choice was made.
- A scaler fitted on training runs only.
- Tests for causality and for train/serve parity.
→ [03](03-evaluation-done-right.md)

**Q: An early version lost to ridge on the test run. Why didn't you switch to ridge?**
- On that run ridge won only on cold cells. On warm cells it was the worst model.
- In cross-validation ridge was the worst model on 2 of 7 runs.
- The fix was to route cold points to the soft sensor, which CV supported. The
  final Transformer beats ridge on its own (0.31 vs 0.43).

**Q: You looked at the test set and then changed the model. Isn't that leakage?**
- Partly yes, and I disclose it.
- The *idea* came from the test-run analysis. The *decision* rested on the 7-run
  CV, which shows the same pattern on its own.
- The pipeline also prints test scores each time it runs, and it ran about six
  times. No choice was made from a test number. I log that in the decision log.
- The clean way to confirm it is a fresh run that has never been seen.

**Q: What did the independent review change?**
It found early stopping on the scored CV run, a learning-rate schedule mismatch
in the final fit, a mixed-up average, inconsistent clipping and an unequal GRU
comparison. After the fixes the CV numbers got worse and more honest, and the
log target was selected. → [04](04-results-and-root-causes.md), Act 4

**Q: How did you pick the time window?**
- **Lookback and target scale:** a grid of {linear, log} × 3–36 steps, chosen with
  a paired one-SE rule. The log target with 3 steps (about 2 minutes) won clearly.
- **Horizons:** all 12 are reported. Skill over persistence rises over about 2
  minutes and then holds at about 60–65%.

**Q: Why RMSE on points 5–6 only?**
Points 1–4 are near 0%, so including them shrinks every model's error and
compresses the differences between models. I report both.

## About root causes

**Q: What's the biggest source of error?**
Cold start: points not yet read in a run. In the first version they were 14% of
cells and 81% of the squared error. With the log target and the router, they're
about a third of the final Transformer's error.

**Q: What else?**
- Error scales with the run's CO₂ level (a shift in the target level).
- The analyser's sample line keeps flushing for a while after each switch.
- There are only 7 training runs, so the plant instruments add little beyond the
  CO₂ history. The N₂ inlet flow (FT304) is the one that matters most, which
  makes physical sense: more N₂ dilutes the CO₂.

**Q: With more time, what would you do first?**
Get more data across CO₂ levels. Add physics features (inlet CO₂ fraction, the
kinetic-model output). Use a level-normalised target. Train on randomly hidden
points so one network covers cold start.

## About production

**Q: How would this run at a plant?**
1. Historian data streams into Postgres or TimescaleDB.
2. The API forecasts on every analyser step and logs each prediction.
3. `/monitoring` tracks rolling RMSE and coverage.
4. Alerts trigger retraining, which is gated on leave-one-run-out CV.
5. New versions are shadow-deployed first.

→ [05](05-production.md)

**Q: How do you avoid training/serving skew?**
The API rebuilds features from the database through the same functions used in
training. There's a unit test for it, and the live API matches the offline
predictions to 4 decimal places.

**Q: How do you stop monitoring mixing two model versions?**
The version string includes a hash of the artifact. `/monitoring` filters on
that version and counts a repeatedly logged forecast once.

**Q: Why is the image about 2.6 GB?**
CPU-only PyTorch (about 650 MB) plus NumPy and pandas, measured uncompressed.
Next steps would be a multi-stage build or exporting the model to ONNX Runtime.

## Questions you should ask them

- Which sampling points matter most operationally, and what's the decision a
  forecast would feed (alarm, setpoint, analyser scheduling)?
- How long are real campaigns? Short runs make cold start dominant; continuous
  operation makes it rare.
- Is the kinetic/Simulink model available online? Fusing it is the obvious next step.
