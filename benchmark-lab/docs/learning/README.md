# Study guide: forecasting benchmarks

| # | page | you'll be able to… |
|---|---|---|
| 1 | [01 · Benchmarks and protocol](01-benchmarks-and-protocol.md) | explain the splits and scaling, and why matching them is the only way to compare with papers |
| 2 | [02 · Linear vs Transformer](02-linear-vs-transformer.md) | explain DLinear, patching, channel independence and RevIN, and why tokenisation decides the result |
| 3 | [03 · Hybrids](03-hybrid.md) | explain residual and blended hybrids, and why neither helped on ETTh1 |

Read alongside the papers: Informer → DLinear → PatchTST (links in page 02). Then compare
`results/ETTh1.md` with `results/published_etth1.json`.

**How this connects to the CO₂ challenge:** the CO₂ lab's step-token Transformer is the weakest trained
model here, and its anchor-plus-correction hybrid has nothing to correct. Both outcomes are the same lesson
as the CO₂ write-up: framing and baselines decide more than architecture.
