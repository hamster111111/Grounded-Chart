# GroundedChart Framework Design

## 1. Research Goal

GroundedChart is not intended to be another generic chart-generation agent. The research target is narrower:

> Verify and repair whether a generated chart faithfully satisfies the user's explicit and semantically inferable chart requirements.

This is broader than only checking data values, but narrower than judging generic chart aesthetics. The target is **chart requirement fidelity**: whether the generated chart satisfies the concrete requirements expressed or implied by the natural-language request.

The framework should make the following relation explicit and testable:

```text
natural-language chart request
  -> intended chart requirements
  -> expected requirement state
  -> actual plot trace
  -> requirement-level mismatch report
  -> localized code repair
```

The core claim is about **faithful language-to-chart generation**, not about generic multi-agent orchestration.

## 1.3 Design Inspirations from EMNLP-Style Work

This framework is intentionally shaped more by EMNLP-style methodological patterns than by generic chart-agent systems.

The most relevant design inspirations are:

### Structured Prediction over Free-Form Generation

EMNLP work on constrained and structured generation suggests that the model should not freely improvise the internal representation when the task ultimately requires verifiable structure.

Design implication for GroundedChart:

- requirement extraction should be schema-constrained
- requirement nodes should follow a DSL
- `figure / panel / shared` scopes should be explicit
- output validity should be enforced at parse time, not repaired after the fact

This is why the framework uses `RequirementNode` and `ChartRequirementPlan` instead of raw prompt text as the main internal representation.

### Abstention Is a First-Class Capability

EMNLP work on semantic parsing and task-oriented understanding repeatedly shows that forcing a fully specified structure under ambiguity degrades reliability.

Design implication for GroundedChart:

- abstention should operate per requirement, not only per query
- core requirements must be allowed to remain `ambiguous` or `unsupported`
- the framework should separate `cannot justify` from `incorrect`

This is why ambiguity handling and abstention are explicit framework policies rather than fallback heuristics.

### Attribution Should Include Intermediate Reasoning Artifacts

Work on attribution in QA and reasoning-heavy tasks shows that final outputs are not enough; intermediate evidence must also be attributable.

Design implication for GroundedChart:

- chart fidelity should not be judged only from the final rendered figure
- intermediate deterministic artifacts such as grouped tables, sorted outputs, top-k results, and filtered subsets should be explicit evidence objects
- verifier outputs should identify which requirement failed and which artifact supports the verdict

This is why the framework centers on `ExpectedArtifacts`, `ActualArtifacts`, and `EvidenceGraph`.

### Deterministic Programs Should Carry the Burden of Numerical Truth

Program-grounded EMNLP work suggests that numerical and structural correctness should be checked through deterministic execution whenever possible.

Design implication for GroundedChart:

- LLMs may interpret or repair, but should not serve as the authority for computed chart values
- aggregation, filtering, joins, sorting, percentages, and similar operations should be implemented through SQL, scripts, or deterministic Python execution
- numerical correctness should be evidence-backed, not judge-backed

This is why deterministic computation is treated as a core principle rather than an implementation convenience.

### Repair Should Be Localized, Not Blindly Regenerative

A recurring lesson from EMNLP-style system design is that unconstrained self-revision often makes analysis difficult and weakens causal claims.

Design implication for GroundedChart:

- repair should target failed requirements, not rewrite everything by default
- repair scope should be evidence-guided
- large regeneration should be explicitly justified
- no-regression checks should be mandatory

This is why the framework includes `RepairPlan`, repair levels, and repair acceptance rules.

### Representation, Verification, and Error Analysis Must Line Up

Strong EMNLP papers usually align:

- the representation used by the model
- the unit of verification
- the unit of error analysis

Design implication for GroundedChart:

- the framework should evaluate at the requirement level because requirements are also the unit of parsing, verification, attribution, and repair
- benchmark results should be decomposed by requirement family, not only by aggregate pass rate

This is why the document emphasizes requirement-level metrics such as coverage, satisfaction, and family-wise repair success.

## 1.1 Requirement Fidelity Scope

The framework should eventually cover four levels of chart requirements:

1. **Data Operation Fidelity**
   - correct columns
   - correct filter
   - correct group-by
   - correct aggregation
   - correct join
   - correct sort / limit

2. **Encoding Fidelity**
   - correct chart type
   - correct x/y/color/size/facet bindings
   - correct axis orientation
   - correct categorical vs quantitative role

3. **Semantic Annotation Fidelity**
   - correct title
   - correct axis labels
   - correct legend labels
   - correct units
   - correct annotation / highlight target when explicitly requested

4. **Explicit Presentation Constraint Fidelity**
   - requested order
   - requested top-k / bottom-k
   - requested log scale
   - requested percentage format
   - requested horizontal / vertical orientation
   - requested colors or markers when explicitly specified

5. **Figure Composition Fidelity**
   - panel count
   - panel layout
   - panel assignment
   - shared axis / shared legend requirements
   - panel-specific titles
   - figure-level vs panel-level constraints

The boundary is important:

- In scope: explicit or semantically inferable chart requirements.
- Out of scope for the main claim: generic aesthetics, publication style, subjective layout quality, or open-ended visual preference.

The unit of generation should be a **single figure**, not necessarily a single panel. A figure may contain:

- one chart panel
- multiple chart panels / subplots
- shared constraints across panels

The MVP should still start with the hardest and most evidence-supported subset: **data operation fidelity plus basic encoding/order fidelity**, initially for single-panel figures, while keeping the representation compatible with multi-panel figures.

## 1.2 Traceable Requirement Grounding

The stronger version of the framework should be **traceable**, not only verifiable.

Every requirement should carry an evidence chain:

```text
query span
  -> parsed requirement node
  -> deterministic expected artifact
  -> actual plot element / trace
  -> verifier verdict
  -> localized repair action
  -> post-repair verification result
```

This traceability is a core research advantage:

- It prevents the verifier from becoming a black-box LLM judge.
- It makes each failure attributable to a specific requirement.
- It supports paper-level error analysis.
- It makes repair success objectively re-checkable.

### Requirement Provenance

Each extracted requirement should retain where it came from in the user query.

Example:

```json
{
  "requirement_id": "r_limit_top5",
  "type": "presentation_constraint",
  "name": "top_k",
  "value": 5,
  "source_span": "top 5",
  "confidence": 0.97,
  "status": "explicit"
}
```

For inferred requirements, the plan should record the assumption:

```json
{
  "requirement_id": "r_metric",
  "type": "data_operation",
  "name": "measure",
  "value": "accuracy",
  "source_span": "performance",
  "status": "assumed",
  "assumption": "Use accuracy because it is the only performance-like numeric column."
}
```

This distinction is important. The framework should not silently convert ambiguous language into ground truth.

### Deterministic Computation Principle

Anything that can be computed deterministically should not be computed by an LLM.

Use scripts, SQL, DuckDB, pandas, or controlled Python execution for:

- aggregation
- filtering
- joins
- sorting
- top-k / bottom-k
- percentages and ratios
- normalization
- unit conversion when rules are explicit
- axis value extraction
- requirement-level metric computation

LLMs may be used for:

- interpreting natural language into requirement candidates
- resolving or documenting ambiguity
- generating or repairing chart code

But LLMs should not be trusted as the source of numerical truth. A reviewer-safe claim is:

> We use LLMs for requirement interpretation and code repair, but rely on deterministic execution for numerical and structural verification.

### Evidence Chain

Each verifier result should be able to produce a structured evidence chain.

Example:

```json
{
  "requirement_id": "r_agg_sum_sales",
  "source_span": "total sales",
  "parsed_requirement": {"agg": "sum", "measure": "sales"},
  "expected_artifact": {
    "executor": "duckdb",
    "program": "SELECT category, SUM(sales) FROM table GROUP BY category",
    "result_preview": [["A", 25], ["B", 7]],
    "artifact_hash": "expected:abc123"
  },
  "actual_artifact": {
    "trace_source": "matplotlib.ax.bar",
    "values": [["A", 10], ["A", 15], ["B", 7]],
    "artifact_hash": "actual:def456"
  },
  "verdict": "fail",
  "error_code": "missing_groupby"
}
```

The paper should report not only whether a chart passes, but which requirement failed and what evidence supports the verdict.

### Failure Attribution

The framework should distinguish at least these failure locations:

- `parse_failure`: the requirement was extracted incorrectly
- `ambiguous_requirement`: the query does not specify enough information
- `canonical_execution_failure`: the deterministic executor cannot compute the expected artifact
- `code_execution_failure`: generated chart code fails to run
- `trace_extraction_failure`: the plot was generated but cannot be traced reliably
- `requirement_violation`: expected and actual artifacts are both available and disagree
- `repair_failure`: the failed requirement is known, but repair does not fix it

This is better than reporting only `pass` or `fail`, because it separates framework limitations from model limitations.

### Requirement Coverage Metrics

Besides full pass rate, report requirement-level metrics:

```text
requirement coverage = verifiable requirements / extracted requirements
requirement satisfaction = satisfied requirements / verifiable requirements
repair success = repaired failed requirements / attempted failed requirements
```

These metrics are useful because a chart can partially satisfy a complex query. Full pass rate hides that structure.

## 2. Problem Scene

A user asks for a chart from structured data:

```text
Show total sales by product category in ascending order.
```

A language model may produce code that:

- runs successfully
- renders a plausible bar chart
- uses the right chart type
- has readable labels

but still violates the user's requirements because it:

- forgets `GROUP BY`
- uses `COUNT` instead of `SUM`
- filters the wrong column
- misses a required join
- sorts by the wrong axis
- plots raw rows instead of aggregated results
- binds the requested measure to the wrong axis
- uses a line chart when the user requested a bar chart
- omits a requested top-k constraint
- labels an axis with the wrong unit
- highlights the wrong entity

These failures are dangerous because they are often silent. The figure looks reasonable, but the chart does not actually satisfy the user's requirements.

## 3. Why This Is Not Just an Agent Framework

An agent framework usually emphasizes:

- planner / executor / critic roles
- multi-round critique and revision
- tool orchestration
- prompt decomposition

GroundedChart instead emphasizes:

- explicit chart requirement intent
- execution-grounded plot tracing
- deterministic verification where possible
- requirement-level error localization
- targeted code repair

Agents may be used later as an implementation detail, but the main contribution should be the verification target and the intermediate representations.

## 4. Core Intermediate Objects

### 4.0 `RequirementNode`

Represents one atomic chart requirement with provenance.

Each node should include:

- stable requirement id
- scope (`figure`, `panel`, or `shared`)
- requirement type
- normalized value
- source span from the query
- explicit / inferred / assumed status
- confidence
- dependency ids, if the requirement depends on other requirements

Example:

```json
{
  "requirement_id": "r_sort_desc",
  "scope": "shared",
  "type": "presentation_constraint",
  "name": "sort",
  "value": {"by": "measure", "direction": "desc"},
  "source_span": "from high to low",
  "status": "explicit",
  "depends_on": ["r_agg_sum_sales"],
  "confidence": 0.95
}
```

This node-level representation is what makes requirement verification traceable.

### 4.0.1 Requirement DSL

The framework should define a strict requirement schema rather than relying on loosely structured prompts.

The purpose of the DSL is to make the parser, executor, verifier, and repairer operate on the same objects.

Minimum fields for each requirement node:

- `requirement_id`
- `scope`
- `type`
- `name`
- `value`
- `source_span`
- `status`
- `confidence`
- `depends_on`
- `priority`

Recommended controlled vocabularies:

- `scope`:
  - `figure`
  - `panel`
  - `shared`

- `type`:
  - `data_operation`
  - `encoding`
  - `annotation`
  - `presentation_constraint`

- `status`:
  - `explicit`
  - `inferred`
  - `assumed`
  - `ambiguous`
  - `unsupported`

- `priority`:
  - `core`
  - `secondary`

Core requirements should include:

- chart type
- dimension / measure selection
- aggregation
- filter
- join
- sort / limit
- x/y binding
- orientation
- panel assignment, when multi-panel is requested

Secondary requirements may include:

- title
- axis labels
- legend labels
- units
- formatting
- annotation or highlighting
- panel title
- shared legend presentation

This distinction matters for both metrics and repair acceptance. Failing a core requirement should be treated as more severe than failing a secondary one.

### 4.1 `ChartRequirementPlan`

Represents what the user asked for at the chart-requirement level.

Current code uses the narrower name `ChartIntentPlan`. The research framing should move toward `ChartRequirementPlan`; the implementation can rename it later once the schema stabilizes.

Example:

```json
{
  "figure_requirements": {
    "panel_count": 1,
    "layout": "1x1"
  },
  "panels": [
    {
      "panel_id": "p1",
      "chart_type": "bar",
      "data_ops": {
        "dimensions": ["category"],
        "measure": {"column": "sales", "agg": "sum"},
        "filters": [],
        "sort": {"by": "measure", "direction": "asc"},
        "limit": null
      },
      "encodings": {
        "x": "category",
        "y": "sum(sales)",
        "color": null,
        "facet": null
      },
      "annotations": {
        "title": "Total sales by category",
        "x_label": "Category",
        "y_label": "Total sales"
      },
      "presentation_constraints": {
        "orientation": "vertical",
        "scale": "linear",
        "format": null
      }
    }
  ],
  "shared_requirements": []
}
```

This object is important because it turns natural language into a verifiable representation. It should separate data requirements, encoding requirements, annotation requirements, and presentation constraints instead of collapsing everything into a single free-form prompt.

Longer term, `ChartRequirementPlan` should contain a small dependency graph of `RequirementNode`s rather than only a flat JSON object. This matters for operations such as top-k, sorting, percentage computation, and ranking, which depend on earlier filtering or aggregation requirements.

Minimum supported requirements for the first prototype:

- chart type
- dimension column
- measure column
- aggregation
- filter
- sort
- limit
- x/y binding
- orientation
- simple join, later

Figure-level extensions for later versions:

- `panel_count`
- `panel_layout`
- `panel_assignment`
- `shared_axis`
- `shared_legend`
- panel-specific titles

### 4.2 `Expected Requirement State`

The expected state that should be satisfied if the requirement plan is correctly executed over the source data.

For the MVP, this mainly means the expected plotted data table for a single panel. Later, it should also include:

- expected panel count
- expected panel-to-requirement assignment
- expected encodings and labels per panel
- explicit shared constraints across panels

For the example above:

```text
category | sales_sum
B        | 7
A        | 25
```

This should be computed by a canonical executor, preferably deterministic.

Expected artifacts should record:

- executor type
- deterministic program or query
- input data hash
- output artifact hash
- short result preview
- requirement ids supported by the artifact

MVP implementation:

- in-memory Python executor for simple rows

Later implementation:

- DuckDB / SQL executor
- pandas executor
- benchmark-specific adapters

### 4.3 `PlotTrace`

Represents what the generated code actually plotted and encoded.

For multi-panel figures, `PlotTrace` should eventually become figure-aware:

- figure-level trace
- one trace per axes / panel
- shared elements such as legends or shared axes

Example:

```json
{
  "chart_type": "bar",
  "points": [
    {"x": "A", "y": 10},
    {"x": "A", "y": 15},
    {"x": "B", "y": 7}
  ],
  "encodings": {"x": "category", "y": "sales"},
  "labels": {"x_label": "Category", "y_label": "Sales"},
  "source": "matplotlib_trace"
}
```

This is the most important engineering component. It prevents the method from relying only on screenshots or LLM judging.

Actual trace artifacts should record:

- plotting API intercepted
- code location, when available
- plotted values
- labels and encodings, when available
- artifact hash
- requirement ids the trace is aligned to

MVP implementation:

- manual trace construction for unit tests

Next implementation:

- monkey-patch Matplotlib calls such as `bar`, `plot`, `scatter`, `pie`
- capture x/y values, chart type, order, labels, orientation, and basic encoding metadata
- later, record subplot index / axes identity for panel alignment

### 4.4 `VerificationReport`

Compares expected and actual chart requirements.

The MVP compares expected and actual plotted data for a single panel. The target framework should compare:

- data operations
- encodings
- labels
- explicit constraints
- figure-level composition constraints

Example errors:

```json
[
  {
    "code": "length_mismatch_extra_points",
    "operator": "groupby",
    "message": "Actual plot has more points than expected."
  },
  {
    "code": "wrong_aggregation_value",
    "operator": "aggregation",
    "message": "A plotted value does not match the expected aggregate."
  },
  {
    "code": "wrong_encoding_binding",
    "requirement": "encoding",
    "message": "The requested measure is bound to the wrong axis."
  },
  {
    "code": "wrong_axis_label",
    "requirement": "annotation",
    "message": "The axis label does not match the requested measure or unit."
  }
]
```

The report should be structured enough to support:

- error statistics
- ablations
- repair prompting
- paper tables
- evidence-chain visualization
- failure attribution

### 4.5 `RepairPatch`

Represents the repair action or repair instruction.

The MVP can output structured instructions:

```text
Aggregate rows before plotting; add the missing group-by over category.
Use SUM(sales) rather than raw sales values.
Sort by the aggregated measure in ascending order.
Bind category to the x-axis and total sales to the y-axis.
```

Later versions can generate patched code through an LLM repairer.

Repair patches should also be traceable:

- target requirement ids
- target error codes
- repair strategy
- modified code region, if available
- post-repair verifier result
- whether the same requirement passed after repair

### 4.5.1 `RepairPlan`

Before generating a patch, the framework should decide the repair scope.

Example:

```json
{
  "repair_level": 2,
  "scope": "data_transformation",
  "target_requirements": ["r_groupby_category", "r_agg_sum_sales"],
  "allowed_edits": ["data preparation block", "x/y binding if needed"],
  "forbidden_edits": ["data loading", "chart style", "title", "legend"],
  "reason": "Expected artifact differs from actual plot due to missing group-by aggregation."
}
```

This object prevents the repair step from becoming an unconstrained LLM rewrite.

### 4.6 `EvidenceGraph`

The long-term framework should maintain an evidence graph:

```text
RequirementNode
  -> ExpectedArtifact
  -> ActualArtifact
  -> VerificationError / Pass
  -> RepairPatch
  -> PostRepairVerification
```

The graph does not need a complex database in the first version. A JSON-serializable structure is enough.

This is the object that supports:

- reproducibility
- debugging
- error analysis
- paper case studies
- reviewer-facing evidence

## 5. End-to-End Pipeline

The intended framework is:

```text
Input:
  query
  source data
  schema
  generated plotting code
  single target figure (possibly multi-panel)

Step 1. Parse requirements
  query + schema -> RequirementNodes + ChartRequirementPlan

Step 2. Compute expected requirement state
  ChartRequirementPlan + source data -> ExpectedRequirementState + ExpectedArtifacts

Step 3. Execute and trace generated code
  generated plotting code + source data -> Actual PlotTrace + ActualArtifacts

Step 4. Verify
  ExpectedArtifacts + ActualArtifacts -> VerificationReport + EvidenceGraph

Step 5. Repair
  code + requirement plan + verification report + evidence graph -> repaired code or repair instruction

Step 6. Re-run
  repaired code -> new Actual PlotTrace -> post-repair EvidenceGraph
```

The loop should be bounded. For the first real prototype, use at most two repair rounds.

For the first implementation:

- generation unit = one figure
- supported execution target = one figure with one panel
- future extension = one figure with multiple panels / subplots

## 5.0 Ambiguity and Abstain Policy

The framework should not force a deterministic verification target when the query is under-specified.

Core rule:

> If a core requirement cannot be extracted with sufficient confidence, the framework should abstain rather than silently invent ground truth.

### When to Abstain Before Verification

Abstain if any of the following is true:

- the measure is ambiguous
- the grouping key is ambiguous
- multiple incompatible chart types are equally plausible
- the required column does not exist in the schema
- the query conflicts with the available data
- the requirement depends on an unsupported operation

Examples:

- "show best performance" when multiple performance metrics exist
- "compare groups" without a group field
- "top 5" together with "show all"
- "2024" when only 2023 data exists

### Verification Gating Rule

Only requirements with status `explicit`, `inferred`, or well-justified `assumed` may enter deterministic verification.

Requirements marked as:

- `ambiguous`
- `unsupported`

should be excluded from hard correctness claims.

This implies:

- `requirement coverage` should count only verifiable requirements
- `full pass` should not punish the system for unsupported requirements if they are explicitly marked unsupported
- the framework must distinguish abstention from failure

### Parser Confidence Is Not Enough

The system should not rely on raw parser confidence alone. Abstention should consider:

- parser confidence
- schema evidence
- deterministic computability
- contradiction between extracted requirements

The right question is not \"Can the parser produce something?\" but \"Can the framework justify this requirement as a verification target?\"

## 5.1 Scope-Controlled Repair Policy

Repair should be **evidence-guided and scope-controlled**. The framework should not always ask an LLM to regenerate the whole script.

Core principle:

> The repair scope should match the failure scope.

Small, well-localized failures should trigger constrained local edits. Large structural failures may justify regeneration. Ambiguity or insufficient evidence should trigger abstention rather than blind repair.

### Repair Levels

#### Level 0: No Repair / Verify Only

Use when:

- all verifiable requirements pass
- the query is ambiguous
- verifier evidence is insufficient
- expected artifacts cannot be computed
- trace extraction is unreliable

Action:

- do not modify code
- return `pass`, `abstain`, or failure attribution

This is important because not every failure is safely repairable.

#### Level 1: Constrained Local Patch

Use when:

- one or a small number of requirements fail
- the failed requirement is localized
- the existing code structure is mostly correct

Examples:

- wrong order
- wrong axis label
- missing top-k
- wrong chart orientation
- wrong chart type when data preparation is otherwise correct
- wrong percentage formatting

Action:

- only modify the smallest relevant code region
- preserve data loading, data preparation, visual style, labels, and chart type unless they are the failed requirement
- forbid full-script rewrites

Example repair instruction:

```text
Only modify the sorting step so the plotted values are ordered by the aggregated measure in ascending order.
Do not change data loading, aggregation, chart type, labels, or styling.
```

#### Level 2: Data Transformation Patch

Use when:

- data operations are wrong but the overall chart structure is useful
- expected artifact and actual artifact disagree because of data preparation logic

Examples:

- missing group-by
- wrong aggregation
- wrong filter
- missing join
- wrong measure column
- wrong dimension column

Action:

- allow edits to the data preparation block
- allow x/y binding changes only if required by the failed requirement
- preserve rendering block and visual formatting as much as possible

Example repair instruction:

```text
Modify only the data preparation section so that the plotted dataframe matches the expected artifact.
Preserve the plotting API call and visual formatting unless x/y binding must change.
```

#### Level 3: Structural Regeneration

Use when:

- multiple core requirement families fail
- generated code is unrelated to the query
- code execution succeeds but the plot trace is semantically far from the requirement plan
- local patch would be more fragile than regeneration

Examples:

- wrong chart type + wrong data + wrong encoding
- generated chart answers a different query
- multi-panel structure is inconsistent with a single-chart request
- data flow is too tangled to patch safely

Action:

- allow regeneration of the chart code
- constrain regeneration with the requirement plan and deterministic expected artifacts
- prohibit invented requirements
- require post-regeneration tracing and verification

Example repair instruction:

```text
Regenerate the chart code from the requirement plan and expected artifact.
Use the provided computed table as the only source of plotted values.
Do not invent additional requirements.
```

#### Level 4: Abstain / Ask Clarification

Use when:

- the user query is under-specified
- required columns are missing
- requirements conflict
- expected artifacts cannot be computed
- trace evidence is incomplete

Examples:

- "show best performance" with multiple possible performance metrics
- "compare groups" without a group column
- "show 2024" when data only contains 2023
- "top 5" and "show all" in the same request

Action:

- do not repair
- return clarification need or failure attribution

### Repair Planner vs Repair Executor

The repair component should be split into:

```text
VerificationReport + EvidenceGraph
  -> RepairPlanner
  -> RepairPlan
  -> RepairExecutor
  -> repaired code / repair instruction
  -> TraceRunner
  -> Verifier
```

The `RepairPlanner` decides the scope. The `RepairExecutor` applies the patch or calls an LLM under the scope constraints.

This distinction is important for the paper:

- It shows repairs are not arbitrary LLM rewrites.
- It enables ablation of scope control.
- It makes large regenerations explicitly justified.
- It allows the framework to abstain when evidence is insufficient.

### Repair Acceptance and No-Regression Rule

Repair should not be accepted only because one failed requirement becomes correct.

A repair is acceptable only if:

1. at least one previously failed target requirement now passes
2. no previously satisfied core requirement regresses
3. executability does not regress
4. if regeneration is used, the new code remains aligned with the original requirement plan

Suggested acceptance checks:

```text
net requirement gain > 0
and no core regression
and executable_after_repair = true
```

Where:

- `net requirement gain` = newly satisfied requirements - newly broken requirements
- `core regression` = any previously passed core requirement now failing

This rule is important because a repair can otherwise look successful while silently breaking chart type, ordering, or another core data requirement.

### Oracle vs Predicted Requirement Plan

To understand whether the bottleneck is the parser or the repair framework, evaluation should separate:

- `oracle requirement plan`
- `predicted requirement plan`

This gives two useful settings:

1. **Oracle-plan verification / repair**
   - isolates verifier and repair quality
   - answers whether the framework can repair errors given a correct requirement plan

2. **Predicted-plan verification / repair**
   - measures full end-to-end performance
   - includes parser errors and ambiguity handling

Without this split, it will be hard to tell whether failures come from interpretation or repair.

## 6. Module Responsibilities

### 6.1 Intent Parser

File:

```text
src/grounded_chart/intent_parser.py
```

Responsibilities:

- convert natural-language requests into a requirement plan
- preserve source spans and assumptions for each requirement
- use schema-aware parsing
- expose confidence when possible
- support deterministic tests

MVP:

- `HeuristicIntentParser`

Target:

- LLM parser constrained by JSON schema
- optional self-check for ambiguous queries

Risks:

- If the parser is wrong, the verifier will compare against the wrong target.
- The paper must distinguish parser failures from plotting-code failures.
- If the parser over-extracts subjective preferences, the framework will become too broad. Only explicit or semantically inferable requirements should be extracted.

### 6.2 Canonical Executor

File:

```text
src/grounded_chart/canonical_executor.py
```

Responsibilities:

- execute the requirement plan over source data
- produce expected plotted data and expected requirement state
- support groupby, filter, aggregation, sort, and limit
- record deterministic programs, input hashes, and output hashes for traceability

MVP:

- in-memory rows

Target:

- DuckDB executor for CSV / SQLite / benchmark tables

Risks:

- If this becomes benchmark-specific, the method will look overfit.
- Keep the executor generic and operator-driven.

### 6.3 Trace Runner

File:

```text
src/grounded_chart/trace_runner.py
```

Responsibilities:

- execute generated plotting code in a controlled environment
- intercept plotting calls
- extract actual plotted data and basic requirement evidence
- record plot API calls and actual artifacts for evidence alignment

MVP:

- `ManualTraceRunner` for tests

Target:

- `MatplotlibTraceRunner`

Initial Matplotlib calls to support:

- `Axes.bar`
- `Axes.barh`
- `Axes.plot`
- `Axes.scatter`
- `Axes.pie`
- `pyplot.bar`
- `pyplot.plot`
- `pyplot.scatter`
- `pyplot.pie`

Future multi-panel support should additionally track:

- `plt.subplots`
- axes creation order
- panel-to-axes alignment
- shared legends / shared axes where possible

Risks:

- Generated code may transform data before plotting.
- Code may use pandas plotting wrappers.
- Code may create multiple axes or subplots.
- Some data may be encoded visually rather than directly passed as x/y arrays.
- Some requirements, especially labels and annotations, may require lightweight text extraction from Matplotlib objects.

### 6.4 Verifier

File:

```text
src/grounded_chart/verifier.py
```

Responsibilities:

- compare expected and actual traces
- produce requirement-level mismatch types
- remain deterministic as much as possible
- produce evidence chains for pass/fail decisions

Initial error families:

- `wrong_chart_type`
- `length_mismatch_extra_points`
- `length_mismatch_missing_points`
- `data_point_not_found`
- `unexpected_data_point`
- `wrong_aggregation_value`
- `wrong_order`
- `wrong_data_type`
- `wrong_encoding_binding`
- `wrong_axis_label`
- `missing_requested_constraint`

Target paper analysis should aggregate these into:

- groupby failures
- filter failures
- aggregation failures
- join failures
- sort failures
- type failures
- encoding failures
- annotation failures
- presentation-constraint failures

### 6.5 Repairer

File:

```text
src/grounded_chart/repairer.py
```

Responsibilities:

- decide repair scope from verifier evidence
- convert verifier errors into localized repair instructions
- optionally call an LLM to patch code
- avoid full rewrites unless necessary
- target failed requirement ids rather than rewriting the whole chart blindly
- abstain when repair is unsafe or under-specified

MVP:

- rule-based repair planner
- rule-based repair instruction

Target:

- `RepairPlanner` that assigns repair level and allowed edit scope
- LLM repairer with a constrained prompt:
  - preserve chart type
  - preserve visual intent
  - only repair specified requirement mismatches
  - obey allowed / forbidden edit regions
  - return executable code only

Risks:

- A full-code rewrite may accidentally change unrelated behavior.
- Too many repair rounds will look like a generic agent loop.
- Unscoped repair may make improvements look like lucky regeneration instead of evidence-guided correction.

## 7. Evaluation Plan

### 7.1 Main Benchmark

Use VisEval as the first main benchmark because it directly exposes several requirement checks:

- chart type
- data values
- order
- readability, if later enabled

Its strongest support is still the strict `data_check`, so the paper should not pretend VisEval covers all chart requirements.

For the MVP, treat single-panel chart generation as the primary empirical setting. Multi-panel support should be presented as a planned extension unless a reliable panel trace implementation is ready.

Primary metrics:

- full pass rate
- data_check pass rate
- chart_type pass rate
- order_check pass rate
- requirement-level pass rate, after our verifier is implemented
- operator-level error reduction
- requirement-family error reduction
- repair success rate
- requirement coverage
- requirement satisfaction

Important reporting discipline:

- do not mix metrics with unrelated SOTA pass rates without explaining protocol differences
- separate execution success from requirement fidelity
- separate data-operation fidelity from visual aesthetics
- avoid black-box VLM/LLM judging for core fidelity claims
- report failure attribution, not just pass/fail

### 7.2 Transfer / Secondary Evidence

A single benchmark is not enough for an EMNLP-level claim.

Possible secondary evidence:

- PlotCraft, if data and code paths are accessible
- Text2Vis, only as end-to-end transfer evidence
- a custom operator-level diagnostic set

The diagnostic set may be the most useful because it can isolate:

- groupby
- filter
- aggregation
- join
- sort
- type
- chart type
- x/y binding
- axis label
- top-k / order / scale constraints

### 7.3 Ablations

Minimum ablations:

- without intent parser, direct code repair only
- without execution tracing, LLM critique only
- without localized repair, full rewrite repair
- one repair round vs two repair rounds

These ablations are important to prove the framework is not just prompt engineering.

Additional traceability ablations:

- without source-span provenance
- without deterministic expected artifacts
- without evidence-chain-guided repair
- LLM-only judge vs deterministic verifier
- scoped repair vs always-regenerate repair

## 8. MVP Build Order

### Phase 1: Deterministic Core

Already started:

- schema objects
- in-memory canonical executor
- manual trace runner
- operator-level verifier
- repair planner policy in the design
- rule-based repair instructions
- unit tests

Goal:

- verify the data-operation subset of requirement fidelity without involving an LLM
- add requirement ids and evidence chains before adding more model calls
- establish scoped repair policy before allowing regeneration
- keep the internal representation compatible with future multi-panel figures

### Phase 2: Matplotlib Tracing

Implement:

- `MatplotlibTraceRunner`
- safe execution wrapper
- monkey-patching for common plotting calls
- support for bar, line, scatter, pie
- extraction of chart type, x/y values, order, labels, and orientation where possible
- artifact hashes for traced plot data

Defer to later versions:

- multi-panel axes alignment
- shared legends
- figure-level panel assignment

Goal:

- extract actual plotted data and basic requirement evidence from generated code

### Phase 3: Benchmark Adapter

Implement:

- VisEval adapter
- load source data and query
- run existing generated code
- generate requirement plans
- compare against traces
- export evidence chains for sampled failures

Goal:

- reproduce a subset of known VisEval failures using our verifier

### Phase 4: LLM Parser and Repairer

Implement:

- JSON-schema constrained parser
- source-span requirement extraction
- ambiguity / assumption recording
- scoped repair planner
- localized repair prompt
- retry / validation logic

Goal:

- repair failed cases and measure improvement

### Phase 5: Evidence for Paper

Collect:

- per-error-family repair success
- before/after data_check pass rate
- before/after chart_type and order_check pass rate
- requirement-family success rates
- requirement coverage / satisfaction
- failure attribution distribution
- repair-level distribution
- local-patch vs regeneration outcomes
- evidence-chain case studies
- examples with code diffs
- failure cases where the framework cannot repair
- cost and latency

Goal:

- determine whether this is strong enough for EMNLP main, Findings, or should be pivoted

## 8.5 Current Benchmark Reading

The current benchmark runs already support a useful framework-level reading, even before introducing a stronger LLM repairer.

The main lesson is that GroundedChart should not be framed as "a chart agent that fixes benchmark cases." The evidence so far supports a narrower and more defensible claim:

> GroundedChart is strongest when requirement failures are localizable, verifiable, and repairable within a bounded scope. It is weaker on structural subplot reorganization and other global figure rewrites.

### Current Empirical Signals

On the current MatPlotBench failed-task mini-bench:

- the original failed subset starts at `0/10` pass, with `1` execution error
- the current bounded-repair run reaches `8/10` pass, with `0` remaining execution errors

The remaining failures are not random. They concentrate on structural figure-level issues:

- subplot semantic ordering / role assignment
- precise multi-panel layout alignment

This is useful because it shows that the framework is not just improving by chance. It improves strongly on some requirement families and remains weak on others.

The Plotly smoke run shows a related pattern:

- the framework already handles a meaningful portion of Plotly cases through backend-aware verification
- the main unresolved Plotly errors are still concentrated in data/value mismatches rather than superficial labeling only

This suggests that backend awareness is necessary, but backend-specific handling alone is not the main research story.

### What The Benchmarks Already Support

The current benchmark evidence supports the following framework claims:

1. **Requirement-level failure decomposition is working**

The system distinguishes between:

- axis label / title failures
- annotation failures
- axes-count and composition failures
- artist-count failures
- layout failures
- runtime execution failures

This is important because it means the framework is producing structured failure attribution rather than a single opaque pass/fail decision.

2. **Localized fidelity violations are a real strength**

The current runs show consistent recovery on cases whose failures are local and requirement-grounded, such as:

- wrong labels and titles
- wrong tick labels
- missing required text
- certain runtime compatibility failures
- some backend-specific title/subtitle mismatches

This is exactly the region where bounded repair is scientifically meaningful: the failed requirement is explicit, the verifier localizes it, and the repair can be limited to a small code region.

3. **Execution failures can be brought into the same evidence pipeline**

The current framework no longer needs to treat execution errors as an entirely separate class of benchmark failures. At least some runtime failures can be:

- captured as explicit failure signals
- mapped into repair scope
- rerun under the same verification loop

This is a framework contribution, not just a benchmark convenience, because it extends the evidence chain to runtime compatibility errors.

4. **Backend differences are already visible and should remain explicit**

The current runs show that backend support tiers matter:

- Matplotlib 2D behaves like a native hard-verification backend
- Plotly behaves more like a spec-accessible soft-verification backend
- 3D plots currently remain partially supported

This means the framework should continue modeling backend support explicitly instead of pretending all plotting backends are equally verifiable.

### What The Benchmarks Do Not Yet Support

The current evidence does **not** support the following stronger claims:

- that GroundedChart is generally strong on arbitrary multi-panel figure restructuring
- that bounded rule repair can solve complex subplot reordering
- that the current framework is already strong on data-transformation-heavy failures
- that the main research contribution is "repair" by itself

These unsupported claims are important to exclude, because the remaining failures already show the limits of the current approach.

### Current Capability Boundary

A practical summary of the framework boundary is:

- **Strong region**
  - requirement-grounded local verification
  - bounded local repair
  - runtime compatibility recovery for a subset of cases
  - backend-aware verification behavior

- **Weak region**
  - structural subplot regeneration
  - multi-panel semantic reordering
  - precise layout reconstruction
  - large global rewrites that exceed localized evidence

This boundary is not a weakness in itself. It is useful because it tells us what the framework is actually about.

### Research Implication

The benchmark evidence suggests that the strongest paper framing is not:

> a powerful chart-repair system

but instead:

> an evidence-grounded framework for chart requirement verification and bounded local recovery

where:

- verification is requirement-level
- repair is scope-bounded
- deterministic execution carries numerical and structural truth where possible
- benchmark analysis is reported by error family rather than only aggregate pass rate

### Immediate Design Consequence

Because of the current benchmark evidence, future work should prioritize:

- better error-family analysis
- cleaner separation between core repair and backend adapter logic
- stronger reporting by requirement family
- a principled handoff from local repair to higher-level structural regeneration

Future work should **not** prioritize:

- chasing every remaining failed case with task-specific rule patches
- presenting benchmark-specific repairs as if they were framework-level mechanisms
- overstating current support for multi-panel structural correction

## 8.6 Priority Checklist

The design document is broader than the current implementation. To keep the project aligned with a credible EMNLP-style framing, the missing pieces should be prioritized as follows.

### P0: Must-Have Before Strong Framework Claims

These are the highest-priority gaps. Without them, the framework can run, but the paper-level claim remains weaker than the document suggests.

1. **Requirement provenance must become parser-native**

Current limitation:

- the parser mainly outputs `ChartIntentPlan`
- `RequirementNode` is mostly reconstructed later from that plan
- source spans, assumptions, and requirement status are not first-class parser outputs

Needed:

- parser output should directly produce requirement candidates with:
  - `source_span`
  - `status` in `{explicit, inferred, assumed, ambiguous, unsupported}`
  - optional `assumption`
  - confidence at the requirement level

Reason:

- this is necessary for traceable grounding
- this is also necessary to distinguish parser mistakes from verifier or repair mistakes

2. **Ambiguity and abstention must be real, not only described**

Current limitation:

- the system almost always forces a plan
- ambiguity-aware gating is described in the document but not implemented as a true policy

Needed:

- requirement-level abstention
- pre-verification gating for unsupported / ambiguous requirements
- explicit reporting of abstained requirements instead of silently forcing defaults

Reason:

- otherwise the framework overclaims certainty under ambiguity
- this weakens both evidence quality and paper credibility

3. **Intermediate deterministic artifacts must be added**

Current limitation:

- evidence mostly binds expected and actual plot traces
- the framework does not yet expose intermediate grouped / filtered / sorted artifacts as first-class evidence

Needed:

- expected grouped table
- expected filtered subset
- expected sorted result
- optional program string / artifact hash for deterministic execution

Reason:

- this is central to the design's evidence-grounded claim
- without intermediate artifacts, evidence remains thinner than the document promises

4. **Requirement-level metrics must replace only aggregate reporting**

Current limitation:

- the current metrics and reports are still dominated by case-level pass/fail and error counts

Needed:

- requirement coverage
- requirement satisfaction
- per-family success
- repair success by failed requirement family

Reason:

- aggregate case pass rate hides partial success
- the paper framing depends on requirement-level evaluation

5. **Oracle vs predicted requirement-plan evaluation must be separated**

Current limitation:

- parser quality and verifier/repair quality are still entangled in most runs

Needed:

- oracle-plan runs
- predicted-plan runs
- explicit attribution of failures to parser vs verifier/repair

Reason:

- without this split, it will be hard to argue where the framework actually helps

### P1: Important For Bench-Driven Analysis

These are the next layer. They are not as foundational as P0, but they are necessary if the benchmark analysis is supposed to guide framework evolution instead of just producing numbers.

1. **Second benchmark or diagnostic set**

Current limitation:

- the strongest current evidence still comes from MatPlotBench-derived cases and a Plotly smoke set

Needed:

- at least one additional benchmark or diagnostic set with different failure structure

Reason:

- one benchmark is not enough to support a strong method claim

2. **Error-family-oriented reporting**

Needed:

- reports grouped by:
  - data operation
  - encoding
  - annotation
  - figure composition
  - runtime compatibility
  - backend-specific limitations

Reason:

- this is the right unit for framework analysis
- it also stops development from turning into task-by-task patching

3. **Cleaner boundary between core framework and backend adapters**

Current limitation:

- some current repairs mix core local repair with backend-specific compatibility logic

Needed:

- separate:
  - core requirement-grounded local repair
  - backend adapter / compatibility repair
  - experimental or benchmark-specific rules

Reason:

- this makes the method easier to explain and evaluate honestly

4. **Failure taxonomy should become explicit in reports**

Needed:

- standardized reporting for:
  - parse failure
  - ambiguous requirement
  - canonical execution failure
  - code execution failure
  - trace extraction failure
  - requirement violation
  - repair failure

Reason:

- this is one of the strongest potential framework contributions

### P2: Valuable But Not On The Critical Path

These are useful, but they should not be allowed to distract from the core evidence-grounded story.

1. **Stronger LLM parser**

- useful after oracle/predicted separation is in place
- not the current bottleneck for defining the framework

2. **Tiered LLM repair**

- useful later for structural and semantic failures
- should come after the deterministic and reporting boundary is clean

3. **Richer multi-panel / shared-requirement support**

- important eventually
- but too much investment here too early risks over-expanding the scope before the evidence pipeline is mature

4. **More sophisticated structural regeneration**

- likely needed for hard layout / subplot-role failures
- but this should come after the framework can clearly explain why local repair stops being appropriate

### Deprioritize For Now

The following should not be a near-term focus:

- chasing remaining benchmark failures one by one
- adding more task-specific local rules just to improve mini-bench pass rate
- presenting backend compatibility patches as if they were the core research contribution
- optimizing for arbitrary multi-panel figure reconstruction before provenance and evidence are fully solid

### Recommended Next Execution Order

If development is guided by the current framework goal, the recommended order is:

1. requirement-provenance parser output
2. ambiguity / abstain policy
3. intermediate deterministic artifacts
4. requirement-level metrics and failure taxonomy
5. oracle vs predicted evaluation split
6. second benchmark / diagnostic evidence
7. only then revisit stronger LLM repair or structural regeneration

## 9. Non-Goals

For the first serious prototype, do not focus on:

- generating prettier charts
- scientific illustration
- image-to-code chart mimicking
- open-ended multi-agent planning
- screenshot-only visual judging
- arbitrary Python program verification
- subjective aesthetics not grounded in explicit user requirements
- LLM-generated numerical ground truth for core verification
- full arbitrary figure composition before panel tracing is reliable

These may be useful later, but they will weaken the core paper if introduced too early.

## 10. EMNLP-Level Bar

The project is not EMNLP-ready just because the pipeline runs.

A strong submission needs to show:

- significant improvement on requirement-fidelity metrics
- gains across more than one error family
- at least one secondary benchmark or diagnostic set
- robustness on a stronger backbone
- minimal degradation in executability and visual quality
- clear distinction from ordinary agent frameworks
- traceable evidence chains for representative successes and failures
- deterministic verification for numerical and structural requirements

The most realistic EMNLP-level path is:

- main claim: chart requirement fidelity
- main empirical strength: data operation and encoding/order fidelity
- secondary evidence: labels, presentation constraints, or diagnostic cases where feasible

If the final system only fixes `missing_groupby` on one benchmark, it should be treated as a useful prototype but not a strong EMNLP main paper.
