# MVP Result Diagnosis

Date: 2026-04-26

This note diagnoses the first policy-clean MVP comparison after the evidence-grounded repair framework was connected end to end. It is intentionally conservative: the results are useful evidence for the direction, but they are not yet a publishable evaluation.

## Run Setup

Configuration:

- LLM config: `configs/llm_ablation.deepseek.yaml`
- Repair rounds: `1`
- Ablation flag: `--llm-repair-ablation`
- Variants: `verify_only`, `rule_repair`, `vanilla_llm_repair`, `evidence_guided_llm_repair`
- Main outputs:
  - `outputs/mvp_compare_small_error_bench_deepseek_latest/compare_summary.json`
  - `outputs/mvp_compare_matplotbench_failed_tasks_deepseek_latest/compare_summary.json`

Important interpretation rule:

- `pass_rate` is based on our current verifier and requirement bindings, not an official benchmark score.
- `matplotbench_failed_tasks` is a curated failed subset, not the full MatPlotBench distribution.
- Evidence-guided repair uses richer failure atoms and artifact previews, so its prompt is more informative but also more expensive.

## Aggregate Results

### small_error_bench

| Variant | Passed | Pass Rate | Failed Reqs | Hard Failed Reqs | LLM Calls | Tokens |
|---|---:|---:|---:|---:|---:|---:|
| verify_only | 2/9 | 0.2222 | 11 | 11 | 0 | 0 |
| rule_repair | 2/9 | 0.2222 | 12 | 12 | 0 | 0 |
| vanilla_llm_repair | 4/9 | 0.4444 | 8 | 8 | 4 | 5918 |
| evidence_guided_llm_repair | 4/9 | 0.4444 | 7 | 7 | 4 | 9032 |

Diagnosis:

- Evidence-guided repair does not improve case-level pass rate over vanilla on this small bench.
- It does reduce one additional failed requirement compared with vanilla.
- It is not uniformly better: it fixes `small-fail-missing-legend-annotation`, while vanilla fixes `small-fail-wrong-projection-annotation`.
- Token cost increases from `5918` to `9032`, about `+52.6%`.

### matplotbench_failed_tasks

| Variant | Passed | Pass Rate | Hard Pass Rate | Failed Reqs | Hard Failed Reqs | LLM Calls | Tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| verify_only | 1/10 | 0.1000 | 0.2000 | 13 | 12 | 0 | 0 |
| rule_repair | 3/10 | 0.3000 | 0.4000 | 8 | 7 | 0 | 0 |
| vanilla_llm_repair | 6/10 | 0.6000 | 0.7000 | 4 | 3 | 7 | 17696 |
| evidence_guided_llm_repair | 7/10 | 0.7000 | 0.8000 | 3 | 2 | 7 | 24462 |

Diagnosis:

- Evidence-guided repair improves over vanilla by `+1` passed case and `-1` failed requirement at the same number of LLM calls.
- The additional solved case is `matplotbench-35-wrong-log-function-titles`, where the remaining failure was a title/text requirement.
- No case regressed from vanilla to evidence-guided on this subset.
- Token cost increases from `17696` to `24462`, about `+38.2%`.

## Case-Level Takeaways

### Evidence-Guided Wins

`matplotbench-35-wrong-log-function-titles`:

- Vanilla remains hard failed with `wrong_axis_title`.
- Evidence-guided passes.
- The failure atom binds `panel_0.axis_0.title` to `expected.figure_requirements -> actual.panel_0.text`.
- Interpretation: evidence-guided prompting appears useful when the repair target is a concrete presentation/text mismatch and the artifact chain points to the exact requirement.

`small-fail-missing-legend-annotation`:

- Vanilla remains hard failed.
- Evidence-guided passes.
- Resolved requirements: `panel_0.axis_0.legend_labels`, `panel_0.axis_0.text_contains`.
- Interpretation: failure atoms help the LLM focus on missing semantic annotations rather than generic plotting edits.

### Shared Wins

Both vanilla and evidence-guided repair solve:

- `small-fail-wrong-sort`
- `matplotbench-20-wrong-3d-bar-yticks`
- `matplotbench-41-missing-success-label`
- `matplotbench-48-wrong-histogram-order-and-extra-colorbars`
- `matplotbench-59-wrong-bar-cardinality`
- `matplotbench-90-plotly-title-mismatch`

Interpretation:

- Some gains are from giving any LLM repair opportunity, not specifically from evidence grounding.
- The current evidence-guided advantage is incremental, not dominant.

### Evidence-Guided Regression / Tradeoff

`small-fail-wrong-projection-annotation`:

- Vanilla passes.
- Evidence-guided remains hard failed.
- New failed requirements under evidence-guided: `panel_0.axis_0.projection`, `panel_0.axis_0.text_contains`.
- Interpretation: evidence guidance can over-focus the patch or preserve constraints in a way that prevents a broader successful repair. This is a real risk for the claim.

## Remaining Failure Modes

`matplotbench-58-wrong-eventplot-orientation-counts`:

- All variants fail with `wrong_artist_count`.
- Failure atom: `panel_0.axis_0.artist_counts`, `expected.figure_requirements -> actual.figure_trace`.
- Current interpretation: this is visual-structure repair, not simple local text/data repair. The framework can diagnose it, but the current repair policy does not yet handle it.

`matplotbench-71-wrong-zoom-layout`:

- All variants remain `warning_only_failed` with `wrong_axis_layout`.
- Failure atom: `panel_0.axis_0.bounds`, `expected.figure_requirements -> actual.figure.layout`.
- Current interpretation: this should not be a core automatic-repair claim yet. It is more of a layout fidelity warning.

`matplotbench-81-sankey-structure-mismatch`:

- Verify-only has an execution error; repair variants still fail.
- Failure atoms include runtime/chart-type failure and missing figure artifact binding.
- Current interpretation: this should likely be routed as unsupported or structural regeneration, not local repair.

`small-fail-missing-groupby`:

- All variants fail, even though the artifact chain is clear: `expected.aggregated_table -> actual.aggregated_table`, `expected.sorted_table -> actual.sorted_table`.
- Current interpretation: diagnosis is strong, but one-round repair did not rewrite the data preparation block. This is a framework capability gap, not a verifier gap.

## What We Can Claim Now

Safe claim:

- Evidence-grounded failure atoms and artifact chains can improve LLM repair quality on selected chart-fidelity failures, especially when the verifier can bind a failed requirement to concrete expected/actual artifacts.
- On the 10-case MatPlotBench failed subset, evidence-guided repair improves over vanilla repair by `+1` passed case and `-1` failed requirement at the same LLM call count.
- On the 9-case small bench, evidence-guided repair matches vanilla case-level pass rate but leaves fewer failed requirements.

Unsafe claim:

- Do not claim broad SOTA improvement.
- Do not claim full MatPlotBench improvement.
- Do not claim the framework solves plotting generation generally.
- Do not claim evidence guidance is always better than vanilla prompting.
- Do not claim cost-neutral improvement, because evidence-guided repair uses substantially more tokens.

## EMNLP-Relevant Interpretation

The current result supports the framework direction, but only as a preliminary signal. The most defensible research angle is not simply "we get higher pass rate". A stronger framing is:

> Evidence-grounded chart repair converts opaque verifier failures into requirement-level, artifact-linked failure atoms, enabling more targeted LLM repair and more interpretable failure analysis.

This is closer to the B+C direction: better repair plus better evidence-grounded diagnosis. The numbers are promising enough to continue, but not enough for a final EMNLP claim.

## Required Next Experiments

1. Run a larger but still controlled MatPlotBench subset, ideally 30 to 50 cases, with both failed and originally-correct cases.
2. Add a token-budget-matched control. Otherwise, reviewers can argue evidence-guided repair wins because it receives more context, not because the evidence structure matters.
3. Add repeated runs or deterministic replay checks for LLM repair outputs, even at temperature 0.
4. Report requirement-family metrics: data transformation, text/annotation, layout, visual structure, runtime/unsupported.
5. Separate `auto_repairable`, `diagnose_only`, and `unsupported` cases in the headline metrics.
6. Add qualitative before/after images for the cases where evidence-guided differs from vanilla.
7. Decide whether structural visual repair should be implemented or explicitly routed to regeneration.

## Immediate Engineering Priorities

1. Add an automated diagnosis exporter so this document is not manually maintained.
2. Implement token-budget-matched ablations: vanilla+same-token context, evidence without previews, evidence with compressed previews.
3. Improve policy routing for visual-structure failures such as artist count and eventplot orientation.
4. Keep Sankey/unsupported backend cases out of automatic local repair unless structural regeneration is explicitly enabled.
5. Add positive-case preservation tests so repair does not overfit failed subsets.
## Parser Stress Follow-Up

After the initial MVP diagnosis, we ran `parser_stress_bench` with `--parse-source both` and found a serious false-pass risk: before parser optimization, `verify_only__predicted` passed `4/5` cases while `verify_only__oracle` passed `0/5`. This meant the heuristic parser was missing constraints, so the verifier did not check them.

A small schema-aware parser improvement was added for common requirement synonyms:

- `trend`, `over time`, `time series` -> line chart intent
- `typical` -> mean aggregation
- `smallest to largest`, `lowest to highest` -> ascending sort
- value-before-column categorical filters such as `for East region`
- conservative column aliases such as `market` -> `region`

After this parser optimization, `parser_stress_bench` changed as follows:

| Variant | Before Parser Opt | After Parser Opt | Interpretation |
|---|---:|---:|---|
| verify_only__predicted | 4/5 | 0/5 | False passes were removed; predicted requirements now expose the intended failures. |
| verify_only__oracle | 0/5 | 0/5 | Oracle baseline unchanged. |
| vanilla_llm_repair__predicted | 5/5 | 5/5 | Vanilla can repair all parser-stress predicted failures after constraints are exposed. |
| evidence_guided_llm_repair__predicted | 4/5 | 4/5 | Evidence-guided remains one case behind vanilla on this stress bench. |
| vanilla_llm_repair__oracle | 3/5 | 3/5 | Oracle-side vanilla unchanged. |
| evidence_guided_llm_repair__oracle | 4/5 | 4/5 | Oracle-side evidence-guided unchanged and still better than vanilla. |

Important conclusion:

- Parser quality is part of the framework, not a peripheral detail. A weak parser can inflate pass rate by failing to produce verifiable requirements.
- The parser optimization makes the benchmark stricter and more honest, even though it lowers predicted verify-only pass rate.
- Evidence-guided repair still needs improvement for data-transformation rewrites such as `mean` aggregation. In the failing case, the failure atom is correct (`expected.aggregated_table -> actual.aggregated_table`), but the LLM returned a non-applicable natural-language patch rather than changed code or patch operations.

This strengthens the evidence-grounded framing: the framework should be evaluated not only by final pass rate, but also by whether it exposes hidden requirement violations instead of silently passing under-specified parser outputs.
