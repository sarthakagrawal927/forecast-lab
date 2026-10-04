# 02 · The Transformer, piece by piece

Every concept below is used in `co2tx/model.py`. Each entry gives what it is, why
it matters here, and where to learn it properly. Read the code next to this page.
For the theory, the linked sources are better than anything rewritten here.

**Study order:** watch Karpathy first, then read the Annotated Transformer, then
read `model.py` top to bottom. It is about 170 lines.

- **Karpathy, "Let's build GPT: from scratch, in code, spelled out"**: <https://www.youtube.com/watch?v=kCc8FmEb1nY>
  (about 2 hours, attention written live).
- **The Annotated Transformer** (Harvard NLP): <https://nlp.seas.harvard.edu/annotated-transformer/>.
- **Original paper**: Vaswani et al. 2017, *Attention Is All You Need*: <https://arxiv.org/abs/1706.03762>.

---

## Encoder backbone

### Scaled dot-product attention
- **What:** each step builds a query, compares it with every step's key (`QKᵀ/√d`),
  softmaxes the scores into weights, and averages the values `V` with those weights.
- **Why here:** it lets the forecast at "now" weigh any of the past steps in the
  window directly, with no recurrence.
- **Source:** Vaswani §3.2; the Annotated Transformer, "Attention".
- **Code:** `MultiHeadSelfAttention.forward`, where `scores = q @ k.transpose(-2, -1) / math.sqrt(self.dh)`.
- **Why √d:** without it, dot products grow with the width, the softmax saturates,
  and gradients vanish.

### Multi-head attention
- **What:** the same operation run on *h* smaller subspaces in parallel, then
  concatenated back together.
- **Why here:** different heads can track different things, such as the recent
  analyser readings or a slow flow drift. With 4 heads at width 64, each head is
  16 wide.
- **Code:** `.view(B, L, 3, self.h, self.dh).permute(...)`. One fused linear layer
  makes Q, K and V for every head at once.

### Padding mask
- **What:** set the scores at padded positions to `-inf` before the softmax, so
  those positions get exactly zero weight.
- **Why here:** at the start of a run the history is shorter than the window. We
  left-pad with zeros, and the mask keeps the model from treating those zeros as
  real plant data.
- **Test:** `test_padded_steps_do_not_change_the_forecast` fills the padded slots
  with 999 and checks that the output is unchanged.
- **Gotcha:** a row where every position is masked gives `NaN`. We avoid that
  because every window ends on a real step.

### Positional encoding (sinusoidal)
- **What:** fixed sine and cosine waves of different frequencies, added to each
  step's embedding.
- **Why here:** attention ignores order on its own. The encoding tells the model
  which step is "now" and which is 2 minutes ago.
- **Source:** Vaswani §3.5; Kazemnejad's explainer:
  <https://kazemnejad.com/blog/transformer_architecture_positional_encoding/>.
- **Code:** `sinusoidal_positions`.

### Pre-LayerNorm residual blocks
- **What:** each block computes `x + Attn(LN(x))` then `x + FFN(LN(x))`, normalising
  *before* each sublayer instead of after.
- **Why here:** pre-LN trains stably without a careful learning-rate warmup, which
  matters when each fit lasts only a few epochs.
- **Source:** Xiong et al. 2020, *On Layer Normalization in the Transformer Architecture*:
  <https://arxiv.org/abs/2002.04745>.
- **Code:** `EncoderBlock`.

### Feed-forward sublayer (GELU)
- **What:** a two-layer MLP applied to each step separately, with width
  `d_model → d_ff → d_model`.
- **Why:** attention mixes information *across* steps; the feed-forward layer
  transforms the features *within* each step.
- **Source:** Vaswani §3.3; Hendrycks & Gimpel 2016 for GELU: <https://arxiv.org/abs/1606.08415>.

### Pooling: last token + masked mean
- **What:** the head reads two summaries: the representation of the final step,
  and the average over the real (unpadded) steps.
- **Why here:** "what is happening now" plus "what the whole window looked like".
  This is a common choice for encoder-only forecasters.

---

## Head and loss: the parts designed for this problem

### Residual-on-persistence head
- **What:** the head predicts a *correction* to each point's last observed CO₂,
  not the CO₂ value itself. Its output layer starts at zero, so an untrained
  model forecasts exactly "nothing changes" (persistence).
- **Why here:** persistence is a strong baseline for a slow chemical process.
  Starting there means the network only has to learn the departures, and it can
  never do worse than persistence just because training started from a bad place.
- **Same idea elsewhere:** residual learning (He et al. 2015) and "naive-anchored"
  forecasting.
- **Code:** `QuantileResidualHead`. Test: `test_untrained_head_equals_persistence...`.
- **Weakness found later:** when a point has never been read in the run, the
  "last value" is only a fill value, and the anchor is wrong. See RC2 in
  [04-results-and-root-causes.md](04-results-and-root-causes.md).

### Quantile (pinball) loss and non-crossing quantiles
- **What:** to predict the q-th quantile, penalise under-prediction by a weight of
  `q` and over-prediction by `1−q`. We predict the 10th, 50th and 90th
  percentiles, which gives an 80% interval around the median.
- **Why here:** plant operators need a range, not just a single number, and the
  interval width doubles as a monitoring signal.
- **Non-crossing:** the bounds are `median ± softplus(·)`, so p10 ≤ p50 ≤ p90 holds
  by construction.
- **Source:** Koenker & Bassett 1978. In a forecasting context: Lim et al. 2019,
  *Temporal Fusion Transformers*, §4: <https://arxiv.org/abs/1912.09363>.
- **Code:** `pinball_loss`.

### Masked loss on sparse labels
- **What:** compute the loss only on the (horizon, point) cells the analyser
  actually measured, which is one cell per future step.
- **Why here:** the alternative is to train on interpolated labels, which invents
  data and leaks the future. See [03-evaluation-done-right.md](03-evaluation-done-right.md).
- **Code:** `(loss * mask).sum() / mask.sum()`. Test: `test_loss_ignores_unobserved_cells`.

---

## Training mechanics

All of these are in `co2tx/train.py`, in `fit`.

### AdamW
- **What:** Adam with weight decay applied separately from the gradient step.
- **Source:** Loshchilov & Hutter 2017: <https://arxiv.org/abs/1711.05101>.

### Cosine learning-rate schedule
- **What:** the learning rate decays smoothly to zero over the planned epochs.
- **Source:** SGDR: <https://arxiv.org/abs/1608.03983>.

### Gradient clipping (norm 1.0)
- **What:** caps the size of each update, so one odd batch can't throw the weights off.

### Early stopping on a held-out run
- **What:** keep the weights from the epoch with the best validation loss.
- **Here:** the best epoch is typically 5–15. It's chosen on an inner run (nested;
  see [03](03-evaluation-done-right.md)). With 7 training runs the model starts to
  memorise run-specific levels quickly, so this is the main defence against overfitting.

### Input jitter
- **What:** small Gaussian noise added to the sensor inputs during training.
- **Why:** a cheap regulariser when there are about 800 samples.

### Seed ensemble
- **What:** train 5 copies with different random seeds and average their quantiles.
- **Why:** this reduces variance. The spread between members (`ensemble_std` in the
  API) measures how unsure the *model* is, which is a different thing from noise
  in the data.
- **Source:** Lakshminarayanan et al. 2017, *Deep Ensembles*: <https://arxiv.org/abs/1612.01474>.

---

## Why also a GRU?

`CO2GRU` uses the same inputs, the same head and the same loss, so comparing it
with the Transformer isolates the backbone. In cross-validation the Transformer
is slightly ahead (0.95 vs 0.98), well within the fold-to-fold spread. That near
tie is a common result on small, low-dimensional time series.

### Log-scale target (`LogSpace` in `model.py`)
- **What:** the network sees `log(1 + CO₂)` as its anchor and is trained against
  `log(1 + y)`, then maps the output back with `expm1`.
- **Why here:** error grew with the CO₂ level across runs. In log space, a 10%
  relative change looks the same at 3% CO₂ as at 10%.
- **Quantiles:** they survive the round trip exactly, because the transform is
  monotone.
- **Selected by:** the CV grid (D8/D19 in the decision log).

- **Source:** Zeng et al. 2022, *Are Transformers Effective for Time Series
  Forecasting?*: <https://arxiv.org/abs/2205.13504>. It shows simple linear models
  beating many Transformer forecasters. Read it before claiming any
  architecture is a win.

## Exercises

1. Set `n_heads=1` and re-run one fold. Does the error change by more than the
   difference between seeds?
2. Replace the sinusoidal encoding with a learned `nn.Parameter(L, d)`. With a
   3-step window, why does it barely matter?
3. Remove the zero-initialisation in `QuantileResidualHead`. What happens to the
   best epoch and to the cross-validation error?
