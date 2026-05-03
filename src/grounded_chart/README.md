# GroundedChart Package Layout

`grounded_chart` is intentionally a namespace package. The package root should
not contain Python modules; code belongs in explicit subpackages.

## Subpackages

- `api/`: aggregated public imports for scripts, tests, and lightweight users.
- `agents/`: model-backed or agent-facing components such as PlanAgent,
  CodeGen, FigureReader, LayoutAgent, ProtocolAgent, and Executor validation.
- `core/`: stable data models and deterministic core planning/execution
  primitives.
- `data/`: source-data loading and expected/actual artifact extraction.
- `orchestration/`: end-to-end pipelines that wire core, agents, verification,
  repair, and rendering together.
- `repair/`: repair policy, patch operations, repair loops, and repairer
  implementations.
- `runtime/`: LLM client, rendering, trace running, config, and code-structure
  extraction.
- `verification/`: intent parsing, evidence graph construction, diagnostics,
  and requirement-level verification.
- `workspace/`: filesystem artifact/workspace construction.

## Import Rules

- Do not add `.py` files directly under `src/grounded_chart`.
- Use explicit subpackage imports inside framework code.
- Use `grounded_chart.api` only for high-level scripts and compatibility-style
  imports, not for internal module dependencies.
- Keep benchmark adapters outside the core package in `grounded_chart_adapters`.
