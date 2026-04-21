# GroundedChart

GroundedChart is a prototype framework for **faithful language-to-chart generation**. The core idea is not to build another generic chart agent, but to verify whether the data actually plotted by generated code is consistent with the user's natural-language intent and the source table.

## Research framing

**Problem / scene:** LLMs can generate executable, plausible-looking chart code, but the plotted data can silently violate the requested aggregation, filter, join, sort, or type constraints.

**Framework:**

```text
NL query + schema
  -> ChartIntentPlan
  -> expected plotted data
  -> actual plotted data trace
  -> operator-level verification
  -> localized repair hint / repair prompt
```

## Current MVP

This scaffold implements the deterministic core first:

- `ChartIntentPlan`: explicit operator-level intent representation.
- `CanonicalExecutor`: computes expected plotted data from a plan and rows.
- `PlotTrace`: normalized representation of what was actually plotted.
- `OperatorLevelVerifier`: detects length, data point, aggregation, sorting, and chart-type mismatches.
- `RuleBasedRepairer`: produces structured repair guidance.
- `GroundedChartPipeline`: wires the components together.

The LLM parser and LLM repairer are intentionally interfaces/placeholders for now. The first research milestone should be proving that the verifier catches and localizes fidelity errors before adding agent complexity.

## Quick smoke test

```powershell
$env:PYTHONPATH = "src"
python examples/simple_mvp.py
python -m unittest discover -s tests -v
```
