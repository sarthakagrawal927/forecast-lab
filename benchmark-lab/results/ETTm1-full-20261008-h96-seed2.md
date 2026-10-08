# ETTm1 — test MSE / MAE (standardised, all channels; mean over seeds [2])

Lookback 336. Lower is better; **bold** = best per horizon.

| model | H=96 MSE | H=96 MAE |
|---|---|---|
| naive | 1.214 | 0.665 |
| seasonal_naive | 0.423 | 0.387 |
| dlinear | 0.300 | 0.345 |
| patch_transformer | 0.289 | 0.340 |
| hybrid | 0.314 | 0.359 |
| blend | **0.277** | 0.333 |
