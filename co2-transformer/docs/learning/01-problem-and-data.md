# 01 · The plant, the analyser and the data

## The process in five sentences

1. Flue gas (CO₂ mixed into N₂) enters the bottom of an **absorber column**.
2. A lean amine solvent (30% MEA, monoethanolamine) flows down the column from the top.
3. CO₂ reacts with the amine and dissolves, so the gas gets cleaner as it rises.
4. The rich amine goes to a **stripper**, which heats it to release pure CO₂ and
   regenerate the solvent.
5. Six **sampling points** up the column show the CO₂ profile. Point 6 has the
   most CO₂ (around 3–12%) and points 1–4 are near zero.

**Read:**

- Imperial College London's carbon-capture pilot plant, which is the source of this data (search "Imperial ChemEng carbon capture pilot plant").
- The amine scrubbing overview from Rochelle 2009, *Science*:
  <https://doi.org/10.1126/science.1176731>.
- The dataset's own repository and notebook:
  <https://github.com/tonyzyl/CO2-Soft-sensor-for-a-carbon-capture-pilot-plant>.
  The challenge team will also email the paper.

**Why Oil & Gas cares:**

- Capture efficiency sets both cost and emissions compliance.
- Analysers are expensive and slow.
- A model that estimates or forecasts CO₂ between analyser readings is a **soft
  sensor**. It lets operators react minutes earlier and lets one analyser serve
  many points.

## The analyser is the whole problem

One analyser (AT400) is switched between the six points: about 3 readings per
visit, one reading every ~43 s.

```
step:   1 1 1 2 2 2 3 3 3 4 4 5 5 5 6 6 6 1 1 1 2 ...
         <------------ one cycle ≈ 17 steps ≈ 12 min ----->
```

Three consequences shape every design choice:

| fact | consequence | where handled |
|---|---|---|
| Only 1 of 6 points is known at each step | Labels are sparse, so the loss is computed only on measured cells | `make_windows` mask, `pinball_loss` |
| A point is refreshed roughly every 9–12 min | "Last value" goes stale, so the model also needs its **age** | `observation_state` |
| The analyser flushes its sample line after each switch | The first reading at point 1 after point 6 can be contaminated (e.g. 0.29%), and readings drift within a visit | RC4 in the notebook |

## Dataset facts (measured, not assumed)

| | |
|---|---|
| Files (runs) | 8, from Jan–Mar 2014 |
| Rows per run | 50–228 steps |
| Total rows | 898 |
| Sampling | every 43 s |
| Missing values | none |
| Columns | 88 process tags + CO₂ + sampling-point label |
| Tags | 53 temperature, 19 flow, 11 pressure, 4 level, 2 pH |
| Dead tags (constant) | FT101, FT107, FT300 |
| Test run (challenge advice) | `140207_1`: 118 steps, ~84 min |
| CO₂ level at point 6 by run | ~3% (most runs) up to ~10% (`140120_1`) |

Each file is a **separate operating run** with its own flows and CO₂ level. For
example, N₂ flow is about 24 m³/h in the two January–February runs and 50–60
m³/h later. So the real question is "does the model generalise to a run it has
never seen?", which is why every model choice is made by leaving one whole run out.

## The 13 instruments we feed the model

These are the absorber instruments the reference notebook names. In
cross-validation they beat all 88 tags (1.01 vs 1.12 RMSE), because the extra
tags mostly add noise across only 7 training runs.

| tag | meaning |
|---|---|
| FT103 / FT104 | lean amine flow (two meters in series) |
| TT210 / TT211 | lean amine temperature |
| FT301 / FT302 | CO₂ inlet flow |
| FT303 / FT304 | N₂ inlet flow |
| PT402 / PT403 | CO₂ / N₂ inlet pressure (PT403 is also the column top) |
| TT104 | mixed gas inlet temperature |
| TT304 | N₂ inlet temperature |
| PT111 | column bottom pressure |

The physics you should be able to explain:

- **More amine per unit of gas** (FT103/104 up, or gas flow down) means more CO₂
  is absorbed, so concentrations up the column fall.
- **A higher CO₂ fraction in the feed** (FT302 relative to FT304) means more CO₂ at
  every point.
- **Temperature** works both ways: it speeds up the reaction but lowers solubility.

That is why the N₂ inlet flow (FT304/FT303) showing up as the most important
*instruments* (notebook §7) is a good sign: more N₂ dilutes the feed gas.

## What the reference notebook did, and what we changed

| reference | this solution | why |
|---|---|---|
| Linear interpolation fills each point between readings | Carry forward the last reading and add its age | Interpolation uses the *next* reading, which is future data |
| Replaces point-1 readings with point-2 averages | Keep the raw readings; analyse contamination in the RCA | Don't hide a measurement issue inside preprocessing |
| PCA/POD/autoencoder to 16–32 dimensions, then an LSTM | 13 named instruments with a Transformer and a GRU | Named instruments stay interpretable for root-cause analysis; latent components do not |
| Random 1-in-5 validation split across time | Leave-one-run-out cross-validation | A random split of a time series leaks neighbouring rows |
| Keras | PyTorch, written from scratch | Challenge requirement |

The reference is a *soft-sensor / estimation* study, so look-ahead was a smaller
concern for its goal. Forecasting is a different task and needs causal inputs.
Saying this precisely in the interview, rather than calling the reference
"wrong", shows you understand both tasks.
