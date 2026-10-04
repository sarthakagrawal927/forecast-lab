# benchmark-lab: long-horizon forecasting benchmarks

The datasets referenced by the [AI-Challenge](https://github.com/appliedcomputingtech/AI-Challenge)'s
`src/utils` download scripts (ETT, Weather, Electricity), run through one ladder:

**naive → seasonal-naive → DLinear → Transformer (time-step tokens) → Transformer (patch tokens) → hybrids**

The protocol matches the published papers (same splits, train-only scaling, standardised MSE/MAE), so
results sit next to published tables. Every choice is made on the validation months; the test months are
only scored.

## ETTh1 results (test MSE, mean of 3 seeds, lookback 336)

| model | 96 | 192 | 336 | 720 |
|---|---|---|---|---|
| naive | 1.294 | 1.325 | 1.330 | 1.335 |
| seasonal-naive | 0.512 | 0.581 | 0.650 | 0.655 |
| DLinear | **0.373** | **0.406** | **0.445** | 0.507 |
| Transformer, time-step tokens (co2-transformer design) | 0.485 | 0.549 | 0.594 | 0.635 |
| Transformer, patch tokens (PatchTST-style) | 0.381 | 0.420 | 0.448 | **0.484** |
| Hybrid: DLinear + Transformer correction | 0.373 | 0.406 | 0.445 | 0.507 |
| Blend, weight chosen on validation | 0.373 | 0.406 | 0.445 | 0.507 |
| *Published DLinear* | *0.375* | *0.405* | *0.439* | *0.472* |
| *Published PatchTST/42* | *0.375* | *0.414* | *0.431* | *0.449* |

Full table with MAE and seed spread: [`results/ETTh1.md`](results/ETTh1.md). The published rows were
transcribed from the papers' PDFs: [`results/published_etth1.json`](results/published_etth1.json).

**What it shows:**
- **The protocol is right.** Naive matches the published "Repeat" (1.294 vs 1.295), and DLinear lands
  next to its published numbers.
- **Tokenisation decides whether a Transformer works here.** One token per hour is the worst trained
  model. Patches with channel independence are competitive, and best at 30 days.
- **Both hybrids fell back to DLinear.** Validation never supported adding the Transformer
  ([03-hybrid.md](docs/learning/03-hybrid.md)).
- **Training recipe matters as much as the model.** DLinear scores 0.454 with a constant learning rate
  and 0.373 with the paper's per-epoch halving.
- **Our patch Transformer trails PatchTST/42 slightly** (simplified training schedule; see
  [02](docs/learning/02-linear-vs-transformer.md)).

## Run

```bash
uv sync
uv run python -m bench.run --dataset ETTh1                         # all horizons, 3 seeds (~2.5 h on an M-series GPU)
uv run python -m bench.run --dataset ETTh1 --horizons 96 --seeds 0 --tag quick   # ~10 min
uv run pytest
```

ETT files download automatically. Weather and Electricity need the CSVs placed in `data/` (the
challenge's Google Drive links may not work from a script).

## Learn it

[`docs/learning/`](docs/learning/README.md): the benchmarks and protocol, linear vs Transformer, hybrids.
