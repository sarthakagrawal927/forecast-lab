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

## ETTm1 results (2026-10-08; exploratory)

All four horizons and training seeds 0, 1 and 2 completed with six models and
lookback 336. ETTm1 uses 15-minute samples: that lookback is 3.5 days, and the
horizons below cover 1, 2, 3.5 and 7.5 days. The patch model uses width 128,
16 heads and 3 layers. The time-step-token Transformer was not run here.

| model | 96 | 192 | 336 | 720 |
|---|---|---|---|---|
| naive | 1.214 | 1.261 | 1.287 | 1.322 |
| seasonal-naive | 0.423 | 0.463 | 0.496 | 0.574 |
| DLinear | 0.300 | 0.336 | 0.385 | 0.444 |
| Transformer, patch tokens (PatchTST-style) | 0.289 | 0.330 | 0.367 | **0.417** |
| Hybrid: DLinear + Transformer correction | 0.304 | 0.336 | 0.385 | 0.444 |
| Blend, weight chosen on validation | **0.277** | **0.320** | **0.367** | 0.430 |
| *Published DLinear* | *0.299* | *0.335* | *0.369* | *0.425* |
| *Published PatchTST/42* | *0.290* | *0.332* | *0.366* | *0.420* |

Our rows are mean test MSE across three training-seed runs; bold marks the best
local mean. Full MSE/MAE and population standard deviations are in
[`results/ETTm1-full-20261008.md`](results/ETTm1-full-20261008.md).
The [provenance manifest](results/ETTm1-full-20261008.provenance.json) retains
all twelve input hashes, reviewed source hashes, configuration and dataset hash.
Source/data digests describe the current reviewed files; the original input
artifacts do not embed historical execution digests. The
original per-horizon/per-seed JSON and Markdown files are retained beside it;
the earlier quick `ETTm1.json`/`ETTm1.md` are preserved separately.
Published rows were checked against [DLinear Table 2](https://arxiv.org/pdf/2205.13504)
and [PatchTST Table 3](https://arxiv.org/pdf/2211.14730).

**What it shows:**

- The patch Transformer has lower mean test MSE than DLinear at every horizon
  in this recipe. At H=96 it is close to the published PatchTST/42 value, and
  naive is close to the published Repeat baseline. This supports a coarse
  protocol sanity check, not an exact reproduction claim.
- Validation-selected blending helps most at H=96 and H=192. H=336 is a near
  tie with the patch model (the means differ by about 0.0004 MSE).
- The residual hybrid retained epoch 0 in 11 of 12 fits. Its one nonzero
  correction improved validation slightly but worsened test, raising H=96
  mean MSE from 0.300 to 0.304.
- At H=720 the patch model has the lowest mean test MSE, while validation
  selects all DLinear in two of three blends. The validation/test ordering
  differs; changing that choice after seeing test would leak test information.
- Both dataset granularity and patch-model capacity differ from ETTh1. These
  results do not isolate a causal effect of having four times as many samples.
  Three runs are descriptive evidence, not a significance test.

**Reproducibility limit:** in source `38f5b3d`, models are constructed before
`fit` seeds training. Initial weights are therefore uncontrolled, including in
the earlier ETTh1 runs. Seeds label training randomness rather than a fully
seeded initialization; reported spread is run variation. A confirmed reproduction
requires seeding before each model construction and rerunning the experiment.
No model, epoch or blend weight was selected on test for these retained runs.
Weather and Electricity remain optional and unrun.

## Run

```bash
uv sync
uv run python -m bench.run --dataset ETTh1                         # all horizons, 3 seeds (~2.5 h on an M-series GPU)
uv run python -m bench.run --dataset ETTh1 --horizons 96 --seeds 0 --tag quick   # ~10 min
uv run python -m bench.run --dataset ETTm1 --horizons 96 192 336 720 --seeds 0 1 2 \
  --models naive seasonal_naive dlinear patch_transformer hybrid blend --tag full
uv run pytest
```

ETT files download automatically. Weather and Electricity need the CSVs placed in `data/` (the
challenge's Google Drive links may not work from a script).

## Learn it

[`docs/learning/`](docs/learning/README.md): the benchmarks and protocol, linear vs Transformer, hybrids.
