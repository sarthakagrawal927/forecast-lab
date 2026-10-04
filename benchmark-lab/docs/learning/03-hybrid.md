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

## Verdict on ETTh1

Filled in from `results/ETTh1.md` once the run completes. See the README.
