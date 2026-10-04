# Lessons from building this

These are the transferable ideas, the ones worth carrying into the next
forecasting problem. Each one comes from something that actually happened here.

1. **Model the measurement system before the process.**
   The single most important fact was not about CO₂ chemistry. It was that one
   analyser is switched between six points. Sparse labels, staleness, cold start
   and flushing artefacts all follow from it.

2. **A baseline you can't beat is information, not embarrassment.**
   Persistence and ridge framed every result. When ridge won on test, asking
   *where* it won produced the main finding of the project.

3. **Slice errors by system state, not only by horizon or entity.**
   The warm/cold split turned one confusing number (ridge beats the Transformer)
   into two clear facts: the Transformer forecasts well, and ridge soft-senses well.

4. **Choose among near-ties with a rule, decided in advance.**
   Lookbacks from 3 to 36 steps were within noise of each other. The paired
   one-SE rule makes the choice reproducible and defensible, where "it scored
   best" is neither.

5. **Write negative results down.**
   Smaller networks, regularisation and "regime shift in inputs" were all tested
   and all failed. Recording them stopped me retrying them, and they're good
   interview material.

6. **Disclose test-set contact.**
   The router idea came from looking at the test run. Saying so, and showing the
   cross-validation evidence that independently supports it, is stronger than
   pretending the idea arrived from nowhere.

7. **Make serving parity a test, not a hope.**
   A unit test plus a live check (API forecast = offline forecast) caught the
   kinds of bugs that only show up in production: time zones and feature order.
   The tag-parsing test caught a real bug (`FT303m3/hr` parsed as instrument `FT303m`).

8. **Have someone try to break your evaluation.**
   An independent review found early stopping on the scored CV run. That one
   finding moved every network's CV score and reversed a headline claim
   ("the Transformer is the best warm-state forecaster"). Budget for a
   red-team pass before you trust a number.

9. **Architecture is rarely the bottleneck on small data.**
   The Transformer and GRU were close. Everything that moved the numbers was
   framing: causal inputs, the masked loss, the residual head, the log target
   and routing. The next gains are data: more runs and physics features.

## Where this connects to forecast-lab

The repository's thesis, from `PROJECT_STATUS.md`, is that **the evaluation
harness is the asset**. This challenge is the third confirmation:

| lab | what the harness caught |
|---|---|
| `event-forecast` | a model that had collapsed to a constant on real data |
| `demand-forecast` | naive forecasts winning on dense, stationary data; the best method depends on the data regime |
| `co2-transformer` | a headline metric dominated by cold start |
