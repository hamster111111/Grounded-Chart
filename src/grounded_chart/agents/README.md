# GroundedChart Agent Layer

This package contains agent-facing implementations. Agents may call LLM/VLM
clients, write their own workspace artifacts, and produce planning or feedback
signals for orchestration code. They should not own core schemas, deterministic
verification, benchmark adapters, or end-to-end orchestration.

## Modules

- `planning.py`: PlanAgent interfaces and LLM/heuristic planning adapters.
- `codegen.py`: chart code generation agents.
- `executor.py`: static ExecutorAgent fidelity validation.
- `layout.py`: LayoutAgent / layout critic interfaces and implementations.
- `figure_reader.py`: FigureReaderAgent visual-semantic audit.
- `feedback.py`: normalized feedback objects passed between reader/layout agents and PlanAgent.
- `plan_revision.py`: legacy bounded plan revision implementation.
- `protocol.py`: ChartProtocolAgent and rendering protocol generation.

## Boundary Rules

- New agent implementations should live here, not in the `grounded_chart` package root.
- Root modules such as `grounded_chart.plan_agent` are compatibility wrappers only.
- Agents may depend on core data structures such as `construction_plan`, `schema`, `source_data`, and `llm`.
- Core deterministic modules should not depend on agent implementations.
- Orchestration modules may wire agents together, but should not hide agent behavior inside core verifiers.
