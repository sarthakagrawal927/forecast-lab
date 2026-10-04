# 02 · Linear vs Transformer: what actually matters

The 2021–2023 literature on these datasets is a short, useful story. Read the
three papers in order:

1. **Informer (2021) and Autoformer (2021).** Transformers built for long
   sequences, with large gains reported over RNNs.
   <https://arxiv.org/abs/2012.07436> · <https://arxiv.org/abs/2106.13008>
2. **Zeng et al. 2022, *Are Transformers Effective for Time Series Forecasting?***
   One linear layer (DLinear) beats them on nearly every benchmark.
   <https://arxiv.org/abs/2205.13504>
3. **Nie et al. 2023, *A Time Series is Worth 64 Words* (PatchTST).** A
   Transformer becomes competitive again by changing *what a token is*.
   <https://arxiv.org/abs/2211.14730>

Each model in `bench/models.py` is one step of that story.

## The ladder

**Naive** (`naive`)
- **What it is:** the last value, repeated across the whole horizon.
- **Why it's there:** the floor. A model that can't beat it has learned nothing.

**Seasonal-naive** (`seasonal_naive`)
- **What it is:** yesterday, repeated hour by hour.
- **Why it's there:** ETT has a strong daily cycle, so this is a much harder floor.

**DLinear** (`DLinear`)
- **What it is:** a moving-average trend plus the remainder, each mapped to the
  horizon by one linear layer shared across channels.
- **Why it's there:** the "embarrassingly simple" bar from Zeng et al.

**Step-token Transformer** (`StepTransformer`)
- **What it is:** one token per time step, embedding all 7 channels together.
  This is the co2-transformer design.
- **Why it's there:** the Informer-era tokenisation. It shows why that design struggles here.

**Patch-token Transformer** (`PatchTransformer`)
- **What it is:** each channel on its own, cut into 16-step patches with stride 8.
  Each patch is a token.
- **Why it's there:** PatchTST's two ideas, patching and channel independence.

**Hybrid** (`Hybrid`)
- **What it is:** a trained DLinear, frozen, plus a patch Transformer that learns
  only the correction. Its head starts at zero.
- **Why it's there:** the anchor-plus-residual idea from co2-transformer.

**Blend** (in `run.py`)
- **What it is:** `w·DLinear + (1−w)·Patch`, with w chosen on validation.
- **Why it's there:** the simplest hybrid. Is the extra machinery worth more than
  just averaging?

## Ideas you should be able to explain

### Tokenisation is the architecture
- **What:** a token per time step gives attention 336 near-identical tokens. A
  single reading carries little meaning, and attention costs grow with L².
- **Patching:** 16-step patches give 42 tokens that each describe a local
  *shape*, at about 1/64 of the attention cost.
- **Source:** PatchTST §3.1.

### Channel independence
- **What:** forecast each channel with the same shared weights, as if it were a
  separate series.
- **Why it helps:** it multiplies the training examples by the number of
  channels, and stops the model fitting spurious links between channels that
  don't hold up in the test period.
- **Source:** PatchTST §3.1, plus the analysis in Han et al. 2023:
  <https://arxiv.org/abs/2304.05206>.

### Reversible instance normalisation (RevIN)
- **What:** subtract each window's own mean and divide by its std, per channel,
  before the model. Restore them afterwards.
- **Why it helps:** the test period's level differs from training (distribution
  shift). RevIN lets the model learn *shape* while the level comes from the
  input itself. That's the same job our residual-on-persistence head did for CO₂.
- **Source:** Kim et al., ICLR 2022, *Reversible Instance Normalization for
  Accurate Time-Series Forecasting against Distribution Shift*.

### Direct multi-horizon output
- **What:** every model here outputs all H steps at once from a single linear
  head. Nothing is fed back step by step.
- **Why it helps:** feeding predictions back in (autoregressive decoding) compounds
  errors over 720 steps. Informer's "generative decoder" and DLinear both avoid it.

### Training details are part of the result
- **What:** DLinear's published 0.375 needs the reference schedule, which halves
  the learning rate after every epoch.
- **Evidence:** with a constant learning rate we measured 0.454. That gap is
  larger than many of the "improvements" papers report.
- **Lesson:** match the training recipe before you compare architectures.

## What we changed from the papers, and why

| item | paper | here | effect |
|---|---|---|---|
| PatchTST learning-rate schedule | one-cycle, 100 epochs | constant 1e-4, early stop (60 epochs max, patience 8) | slightly worse than the paper; same for every Transformer here, so the comparison stays fair |
| PatchTST normalisation | BatchNorm in the encoder | LayerNorm (our co2 block) | minor |
| Seeds | usually 1, sometimes 5 | 3, mean ± std reported | lets you see which gaps are real |
| Test windows | `drop_last=True` in many codebases | every window | tiny effect on ETTh1; avoids a known pitfall |
