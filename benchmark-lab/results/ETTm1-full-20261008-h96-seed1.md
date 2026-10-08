# ETTm1 — test MSE / MAE (standardised, all channels; mean over seeds [1])

Lookback 336. Lower is better; **bold** = best per horizon.

| model | H=96 MSE | H=96 MAE |
|---|---|---|
| naive | 1.214 | 0.665 |
| seasonal_naive | 0.423 | 0.387 |
| dlinear | 0.299 | 0.343 |
| patch_transformer | 0.293 | 0.343 |
| hybrid | 0.299 | 0.343 |
| blend | **0.277** | 0.332 |
