# 03 · Hybrids: when combining a simple model with a Transformer helps

Two hybrids are tested here. Both follow one idea from co2-transformer: **start
from a strong simple forecast and let the expensive model earn every change.**

## Residual hybrid (`Hybrid` in `bench/models.py`)

```
forecast = DLinear(x)            ← trained first, then frozen
         + PatchTransformer(x)   ← head initialised to zero: starts as "no correction"
```

- **Training:** only the Transformer learns, by minimising the error *left over*
  after DLinear.
- **Do no harm:** early stopping also scores epoch 0, which is exactly DLinear.
  So the hybrid can never be chosen worse than its anchor *on validation*.
- **Why it could help:** a linear model captures level, trend and the daily cycle
  cheaply. A Transformer could add the non-linear parts, such as regime-dependent
  shapes and interactions across the window, without having to relearn the easy
  parts.
- **Why it might not:** if the linear model already captures nearly all the
  predictable signal, the residual is mostly noise. Then the Transformer either
  learns nothing (stops near epoch 0) or overfits the validation period.
- **Same idea elsewhere:** boosting (each stage fits the residual of the last),
  and N-BEATS-style residual stacks. Oreshkin et al. 2020:
  <https://arxiv.org/abs/1905.10437>.

## Blend (`run.py`)

```
forecast = w · DLinear(x) + (1 − w) · PatchTransformer(x),   w ∈ {0, 0.1, …, 1} chosen on validation
```

- **Why it works so often:** if two models make *different* errors, averaging
  cancels part of them. This is the forecast-combination result that has held
  for 50 years: Bates & Granger 1969, and the M4 competition, where combinations
  dominated. Makridakis et al. 2020: <https://doi.org/10.1016/j.ijforecast.2019.04.014>.
- **The cost:** two models to train and serve, and one weight to tune per
  horizon (on validation, never on test).

## How this relates to the CO₂ router

| | CO₂ router | residual hybrid | blend |
|---|---|---|---|
| Decides between models by | system state (point read yet?) | — (always both, additive) | a fixed weight |
| Simple component | ridge soft sensor | DLinear | DLinear |
| When it wins | the two tasks favour different models | the residual has learnable structure | the models' errors are not strongly correlated |

## Verdict on ETTh1: neither hybrid helped

| horizon | DLinear | Patch Transformer | residual hybrid | blend |
|---|---|---|---|---|
| 96 | 0.373 | 0.381 | 0.373 | 0.373 |
| 192 | 0.406 | 0.420 | 0.406 | 0.406 |
| 336 | 0.445 | 0.448 | 0.445 | 0.445 |
| 720 | 0.507 | 0.484 | 0.507 | 0.507 |

(Test MSE, mean of 3 seeds; `results/ETTh1.md` has the full table.)

- **Residual hybrid:** the best validation epoch was 0 in 11 of 12 fits, and 1 in
  the other. The Transformer found no correction to DLinear that held up on the
  validation months, so the "do no harm" rule kept DLinear.
- **Blend:** validation chose w = 1.0 (all DLinear) in all 12 fits.
- **Why:** on ETTh1 the predictable part is level, trend and the daily cycle, and
  DLinear captures it. What's left looks like noise *on the validation period*.
- **The one place it would have paid:** at 720 steps the patch Transformer beats
  DLinear on test (0.484 vs 0.507). On the validation months the order was
  reversed and wide (DLinear 1.20 vs Transformer 1.48), and validation errors are
  2–3× test errors at every horizon: the two periods behave differently. Picking
  the Transformer there anyway would be choosing on test. Note also that
  DLinear is unstable across seeds at 720 (0.48–0.56), while the Transformer is
  not (±0.001).

**Lesson:** a hybrid needs the two models to make *different* errors on data like
the test period. The rule "only change what validation supports" works as
intended: it refused to spend complexity, and it cost almost nothing when the
Transformer would have helped.

## Verdict on ETTm1: blending helps; the residual correction mostly does not

The completed 2026-10-08 experiment contains four horizons × three training-seed
runs. Test MSE means (lookback 336, paper-size patch model):

| horizon | DLinear | Patch Transformer | residual hybrid | blend |
|---|---|---|---|---|
| 96 | 0.300 | 0.289 | 0.304 | 0.277 |
| 192 | 0.336 | 0.330 | 0.336 | 0.320 |
| 336 | 0.385 | 0.367 | 0.385 | 0.367 |
| 720 | 0.444 | 0.417 | 0.444 | 0.430 |

The [full result](../../results/ETTm1-full-20261008.md) includes MAE and run spread;
its [manifest](../../results/ETTm1-full-20261008.provenance.json) preserves the
twelve input hashes. Validation selected these linear blend weights, in training
seed order 0, 1, 2:

| horizon | linear weights | hybrid best epochs |
|---|---|---|
| 96 | 0.4, 0.5, 0.4 | 0, 0, 6 |
| 192 | 0.5, 0.6, 0.6 | 0, 0, 0 |
| 336 | 0.7, 0.7, 0.7 | 0, 0, 0 |
| 720 | 0.7, 1.0, 1.0 | 0, 0, 0 |

Unlike ETTh1, validation supports mixing the two models at every H=96, H=192
and H=336 run. The blend is clearly lower on mean test MSE at the first two
horizons; H=336 is only about 0.0004 below the patch model. At H=720 the patch
model is best on test (0.417), but validation chooses weights 0.7, 1.0 and 1.0.
We retain those choices. The blend's mean test MSE is 0.430; replacing it with
the test winner would violate the selection protocol.

The residual hybrid keeps its DLinear anchor in 11 of 12 fits. H=96 seed 2
selects epoch 6: its validation MSE improves from 0.380 to 0.378, yet test MSE
worsens from 0.300 to 0.314. The no-harm rule is a validation guarantee, not a
test guarantee. Training the correction still costs a second Transformer fit
even when epoch 0 wins.

At H=720 DLinear varies from 0.425 to 0.472 test MSE, while the patch model
ranges from 0.415 to 0.421. These are three-run ranges, not confidence intervals.
The existing runner seeds training after constructing models, so initial weights
are uncontrolled in these and the earlier ETTh1 runs. Treat the experiment as
exploratory; seed before model construction and rerun before claiming a fully
seeded reproduction. The original quick results remain separate.

**Lesson:** combining forecasts can help without a learned residual correction.
Keep validation's decision even when held-out test favors another model. This
comparison changes dataset granularity and model capacity together, so it does
not establish that sample count alone caused the improvement.
