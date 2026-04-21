from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from grounded_chart_adapters.reporting import BatchReport


def write_batch_report_html(report: BatchReport, path: str | Path, title: str = "GroundedChart Batch Report") -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _compact_payload(report.to_dict(), title=title)
    output_path.write_text(_render_html(payload), encoding="utf-8")


def _compact_payload(report_dict: dict[str, Any], title: str) -> dict[str, Any]:
    summary = dict(report_dict.get("summary", {}))
    cases = [_compact_case(case) for case in report_dict.get("cases", [])]
    return {
        "title": title,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "summary": summary,
        "cases": cases,
    }


def _compact_case(case: dict[str, Any]) -> dict[str, Any]:
    backend_profile = dict(case.get("backend_profile") or {})
    case_metadata = dict(case.get("case_metadata") or {})
    figure_requirements = dict(case.get("figure_requirements") or {})
    actual_figure = dict(case.get("actual_figure") or {})
    expected_axes = list(figure_requirements.get("axes") or [])
    actual_axes = list(actual_figure.get("axes") or [])

    return {
        "case_id": case.get("case_id"),
        "status": case.get("status"),
        "query": case.get("query"),
        "expected_chart_type": case.get("expected_chart_type"),
        "actual_chart_type": case.get("actual_chart_type"),
        "backend_profile": backend_profile,
        "error_codes": list(case.get("error_codes") or []),
        "errors": list(case.get("errors") or []),
        "repair_level": case.get("repair_level"),
        "repair_scope": case.get("repair_scope"),
        "repair_strategy": case.get("repair_strategy"),
        "repair_instruction": case.get("repair_instruction"),
        "exception_type": case.get("exception_type"),
        "exception_message": case.get("exception_message"),
        "query_preview": _truncate(str(case.get("query") or ""), 180),
        "reason": case_metadata.get("reason"),
        "source_code": case_metadata.get("source_code"),
        "native_id": case_metadata.get("native_id"),
        "gpt4v_score": case_metadata.get("gpt4v_score"),
        "expected_failure_family": case_metadata.get("expected_failure_family"),
        "expected_figure": {
            "axes_count": figure_requirements.get("axes_count"),
            "figure_title": figure_requirements.get("figure_title"),
            "axes": [_axis_preview(axis) for axis in expected_axes[:3]],
        },
        "actual_figure": {
            "title": actual_figure.get("title"),
            "axes_count": actual_figure.get("axes_count"),
            "size_inches": actual_figure.get("size_inches"),
            "axes": [_axis_preview(axis) for axis in actual_axes[:3]],
        },
    }


def _axis_preview(axis: dict[str, Any]) -> dict[str, Any]:
    artists = list(axis.get("artists") or [])
    artist_types = [artist.get("artist_type") for artist in artists[:6] if artist.get("artist_type")]
    return {
        "index": axis.get("index", axis.get("axis_index")),
        "title": axis.get("title"),
        "projection": axis.get("projection"),
        "xlabel": axis.get("xlabel"),
        "ylabel": axis.get("ylabel"),
        "zlabel": axis.get("zlabel"),
        "bounds": axis.get("bounds"),
        "artist_types": artist_types,
    }


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _render_html(payload: dict[str, Any]) -> str:
    json_payload = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{_escape_html(str(payload["title"]))}</title>
  <style>
    :root {{
      --paper: #f6f2ea;
      --panel: #fffdf8;
      --ink: #1f2933;
      --muted: #66727f;
      --line: #d8d3c7;
      --accent: #1c7c7d;
      --accent-soft: #d9f0ef;
      --error: #b64c4c;
      --error-soft: #f7d9d7;
      --warn: #a2641a;
      --warn-soft: #f5e2c8;
      --ok: #2f7d4a;
      --ok-soft: #d7eddd;
      --shadow: 0 10px 30px rgba(31, 41, 51, 0.08);
      --radius: 16px;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", "Helvetica Neue", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(28, 124, 125, 0.07), transparent 28rem),
        linear-gradient(180deg, #fbf8f3 0%, var(--paper) 100%);
    }}
    .shell {{
      width: min(1400px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 24px 0 40px;
    }}
    .hero {{
      background: linear-gradient(135deg, rgba(28, 124, 125, 0.10), rgba(255, 253, 248, 0.98));
      border: 1px solid rgba(28, 124, 125, 0.18);
      border-radius: 24px;
      box-shadow: var(--shadow);
      padding: 24px;
    }}
    .eyebrow {{
      text-transform: uppercase;
      letter-spacing: 0.12em;
      font-size: 12px;
      color: var(--accent);
      font-weight: 700;
      margin-bottom: 10px;
    }}
    h1 {{
      margin: 0;
      font-size: clamp(28px, 4vw, 42px);
      line-height: 1.05;
    }}
    .sub {{
      margin-top: 12px;
      color: var(--muted);
      font-size: 15px;
      max-width: 900px;
      line-height: 1.5;
    }}
    .meta {{
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      margin-top: 16px;
      color: var(--muted);
      font-size: 13px;
    }}
    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 14px;
      margin-top: 20px;
    }}
    .metric {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      padding: 16px;
      box-shadow: var(--shadow);
    }}
    .metric-label {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 8px;
    }}
    .metric-value {{
      font-size: 30px;
      font-weight: 700;
      line-height: 1;
    }}
    .bands {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 14px;
      margin-top: 20px;
    }}
    .band {{
      background: rgba(255, 253, 248, 0.9);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      padding: 16px;
      box-shadow: var(--shadow);
    }}
    .band h2 {{
      margin: 0 0 12px;
      font-size: 15px;
    }}
    .chips {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
    }}
    .chip {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 8px 12px;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: var(--panel);
      font-size: 13px;
    }}
    .chip strong {{
      font-size: 12px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}
    .toolbar {{
      position: sticky;
      top: 0;
      z-index: 10;
      margin-top: 22px;
      background: rgba(246, 242, 234, 0.92);
      border: 1px solid rgba(216, 211, 199, 0.9);
      border-radius: 18px;
      backdrop-filter: blur(14px);
      padding: 14px;
      box-shadow: var(--shadow);
    }}
    .toolbar-grid {{
      display: grid;
      grid-template-columns: minmax(220px, 2fr) repeat(3, minmax(160px, 1fr));
      gap: 10px;
    }}
    .toolbar input,
    .toolbar select {{
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 10px 12px;
      font: inherit;
      color: var(--ink);
      background: var(--panel);
    }}
    .toolbar-note {{
      margin-top: 10px;
      color: var(--muted);
      font-size: 13px;
    }}
    .cases {{
      margin-top: 20px;
      display: grid;
      gap: 14px;
    }}
    .case {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 18px;
      box-shadow: var(--shadow);
    }}
    .case-top {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: flex-start;
      flex-wrap: wrap;
    }}
    .case-id {{
      font-size: 20px;
      font-weight: 700;
      margin: 0;
      line-height: 1.2;
    }}
    .case-query {{
      margin-top: 8px;
      color: var(--muted);
      line-height: 1.45;
      font-size: 14px;
      max-width: 860px;
    }}
    .badge-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 12px;
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 6px 10px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 700;
      letter-spacing: 0.03em;
      border: 1px solid transparent;
    }}
    .status-passed {{ background: var(--ok-soft); color: var(--ok); border-color: rgba(47,125,74,0.18); }}
    .status-failed {{ background: var(--error-soft); color: var(--error); border-color: rgba(182,76,76,0.18); }}
    .status-error {{ background: var(--warn-soft); color: var(--warn); border-color: rgba(162,100,26,0.18); }}
    .badge-neutral {{ background: #edf1f4; color: #36414c; border-color: rgba(102,114,127,0.18); }}
    .case-grid {{
      margin-top: 16px;
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 12px;
    }}
    .case-panel {{
      background: rgba(248, 246, 240, 0.82);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 12px;
    }}
    .case-panel h3 {{
      margin: 0 0 10px;
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
    }}
    .kv {{
      display: grid;
      grid-template-columns: 110px 1fr;
      gap: 6px 10px;
      font-size: 13px;
      line-height: 1.4;
    }}
    .kv div:nth-child(odd) {{
      color: var(--muted);
    }}
    .code-list {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }}
    .code-pill {{
      padding: 6px 10px;
      border-radius: 10px;
      background: #eef4f4;
      border: 1px solid rgba(28, 124, 125, 0.16);
      font-size: 12px;
      font-family: Consolas, "SFMono-Regular", monospace;
    }}
    .details {{
      margin-top: 12px;
      border-top: 1px dashed var(--line);
      padding-top: 12px;
    }}
    .details summary {{
      cursor: pointer;
      font-weight: 700;
      color: var(--accent);
    }}
    .details-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 12px;
      margin-top: 12px;
    }}
    .mono {{
      font-family: Consolas, "SFMono-Regular", monospace;
      font-size: 12px;
      overflow-wrap: anywhere;
    }}
    .path-link {{
      color: var(--accent);
      text-decoration: none;
    }}
    .path-link:hover {{
      text-decoration: underline;
    }}
    .empty {{
      padding: 24px;
      border: 1px dashed var(--line);
      border-radius: 16px;
      color: var(--muted);
      text-align: center;
      background: rgba(255, 253, 248, 0.7);
    }}
    @media (max-width: 960px) {{
      .toolbar-grid {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <section class="hero">
      <div class="eyebrow">GroundedChart Smoke Browser</div>
      <h1>{_escape_html(str(payload["title"]))}</h1>
      <div class="sub">Backend-aware smoke report for chart-generation failures. The page separates Matplotlib hard/partial cases from Plotly soft cases and surfaces the current repair routing directly in the browser.</div>
      <div class="meta">
        <div>Generated: <strong>{_escape_html(str(payload["generated_at"]))}</strong></div>
        <div>Total Cases: <strong id="meta-total"></strong></div>
        <div>Visible Cases: <strong id="meta-visible"></strong></div>
      </div>
      <div id="summary-grid" class="summary-grid"></div>
      <div class="bands">
        <section class="band">
          <h2>Backend Mix</h2>
          <div id="backend-counts" class="chips"></div>
        </section>
        <section class="band">
          <h2>Verification Modes</h2>
          <div id="verification-counts" class="chips"></div>
        </section>
        <section class="band">
          <h2>Repair Scopes</h2>
          <div id="repair-counts" class="chips"></div>
        </section>
        <section class="band">
          <h2>Error Families</h2>
          <div id="error-counts" class="chips"></div>
        </section>
      </div>
    </section>

    <section class="toolbar">
      <div class="toolbar-grid">
        <input id="search" type="search" placeholder="Search case id, reason, query, error code...">
        <select id="status-filter"></select>
        <select id="backend-filter"></select>
        <select id="repair-filter"></select>
      </div>
      <div class="toolbar-note">Use the filters to isolate Plotly soft cases, 3D partial cases, or only backend-specific regeneration cases.</div>
    </section>

    <section id="cases" class="cases"></section>
  </div>

  <script id="report-data" type="application/json">{json_payload}</script>
  <script>
    const payload = JSON.parse(document.getElementById('report-data').textContent);
    const cases = payload.cases || [];
    const summary = payload.summary || {{}};
    const searchInput = document.getElementById('search');
    const statusFilter = document.getElementById('status-filter');
    const backendFilter = document.getElementById('backend-filter');
    const repairFilter = document.getElementById('repair-filter');
    const casesContainer = document.getElementById('cases');

    document.getElementById('meta-total').textContent = String(cases.length);

    renderSummary();
    initFilters();
    renderCases();

    searchInput.addEventListener('input', renderCases);
    statusFilter.addEventListener('change', renderCases);
    backendFilter.addEventListener('change', renderCases);
    repairFilter.addEventListener('change', renderCases);

    function renderSummary() {{
      const metricSpecs = [
        ['Total Cases', summary.total_cases],
        ['Completed', summary.completed_cases],
        ['Failed', summary.failed_cases],
        ['Errored', summary.errored_cases],
        ['Completion Rate', formatRatio(summary.completion_rate)],
        ['Overall Pass Rate', formatRatio(summary.overall_pass_rate)],
      ];
      document.getElementById('summary-grid').innerHTML = metricSpecs.map(([label, value]) => `
        <article class="metric">
          <div class="metric-label">${{escapeHtml(String(label))}}</div>
          <div class="metric-value">${{escapeHtml(String(value ?? '0'))}}</div>
        </article>
      `).join('');

      renderCountChips('backend-counts', summary.backend_name_counts || {{}});
      renderCountChips('verification-counts', summary.backend_verification_mode_counts || {{}});
      renderCountChips('repair-counts', summary.repair_scope_counts || {{}});
      renderCountChips('error-counts', summary.error_counts || {{}});
    }}

    function renderCountChips(containerId, counts) {{
      const entries = Object.entries(counts || {{}}).sort((a, b) => String(a[0]).localeCompare(String(b[0])));
      const container = document.getElementById(containerId);
      container.innerHTML = entries.length
        ? entries.map(([key, value]) => `<span class="chip"><strong>${{escapeHtml(key)}}</strong> ${{escapeHtml(String(value))}}</span>`).join('')
        : '<span class="chip">none</span>';
    }}

    function initFilters() {{
      setSelectOptions(statusFilter, ['all', ...uniqueValues(cases.map((item) => item.status))], 'Status');
      setSelectOptions(backendFilter, ['all', ...uniqueValues(cases.map((item) => item.backend_profile?.backend_name))], 'Backend');
      setSelectOptions(repairFilter, ['all', ...uniqueValues(cases.map((item) => item.repair_scope))], 'Repair Scope');
    }}

    function setSelectOptions(select, values, label) {{
      select.innerHTML = values.map((value) => {{
        const normalized = String(value ?? '').trim();
        const text = normalized === 'all' ? `All ${{label}}` : normalized || `No ${{label}}`;
        return `<option value="${{escapeHtml(normalized)}}">${{escapeHtml(text)}}</option>`;
      }}).join('');
    }}

    function uniqueValues(values) {{
      return [...new Set(values.filter((value) => value !== undefined && value !== null && String(value).trim() !== ''))].sort((a, b) => String(a).localeCompare(String(b)));
    }}

    function renderCases() {{
      const query = searchInput.value.trim().toLowerCase();
      const status = statusFilter.value;
      const backend = backendFilter.value;
      const repair = repairFilter.value;

      const filtered = cases.filter((item) => matches(item, query, status, backend, repair));
      document.getElementById('meta-visible').textContent = String(filtered.length);

      if (!filtered.length) {{
        casesContainer.innerHTML = '<div class="empty">No cases match the current filters.</div>';
        return;
      }}

      casesContainer.innerHTML = filtered.map(renderCaseCard).join('');
    }}

    function matches(item, query, status, backend, repair) {{
      if (status !== 'all' && item.status !== status) return false;
      if (backend !== 'all' && item.backend_profile?.backend_name !== backend) return false;
      if (repair !== 'all' && (item.repair_scope || '') !== repair) return false;

      if (!query) return true;
      const haystack = [
        item.case_id,
        item.query,
        item.reason,
        item.backend_profile?.backend_name,
        item.backend_profile?.verification_mode,
        item.repair_scope,
        ...(item.error_codes || []),
      ].filter(Boolean).join(' ').toLowerCase();
      return haystack.includes(query);
    }}

    function renderCaseCard(item) {{
      const statusClass = `status-${{item.status || 'failed'}}`;
      const backend = item.backend_profile || {{}};
      const sourcePath = item.source_code ? `<a class="path-link mono" href="${{toFileHref(item.source_code)}}">${{escapeHtml(item.source_code)}}</a>` : '<span class="mono">n/a</span>';
      const expectedFigure = item.expected_figure || {{}};
      const actualFigure = item.actual_figure || {{}};
      const errors = (item.errors || []).slice(0, 6);
      const expectedAxes = (expectedFigure.axes || []).map(renderAxisLine).join('') || '<div class="mono">n/a</div>';
      const actualAxes = (actualFigure.axes || []).map(renderAxisLine).join('') || '<div class="mono">n/a</div>';
      const errorPills = (item.error_codes || []).length
        ? item.error_codes.map((code) => `<span class="code-pill">${{escapeHtml(code)}}</span>`).join('')
        : '<span class="code-pill">none</span>';

      return `
        <article class="case">
          <div class="case-top">
            <div>
              <h2 class="case-id">${{escapeHtml(item.case_id || 'unknown-case')}}</h2>
              <div class="case-query">${{escapeHtml(item.query_preview || item.query || '')}}</div>
              <div class="badge-row">
                <span class="badge ${{statusClass}}">${{escapeHtml(item.status || 'unknown')}}</span>
                <span class="badge badge-neutral">${{escapeHtml(backend.backend_name || 'unknown')}}</span>
                <span class="badge badge-neutral">${{escapeHtml(backend.verification_mode || 'none')}}</span>
                <span class="badge badge-neutral">${{escapeHtml(item.repair_scope || 'none')}}</span>
                <span class="badge badge-neutral">native id: ${{escapeHtml(String(item.native_id ?? 'n/a'))}}</span>
              </div>
            </div>
            <div class="badge-row">
              <span class="badge badge-neutral">expected: ${{escapeHtml(item.expected_chart_type || 'n/a')}}</span>
              <span class="badge badge-neutral">actual: ${{escapeHtml(item.actual_chart_type || 'n/a')}}</span>
              <span class="badge badge-neutral">gpt4v: ${{escapeHtml(String(item.gpt4v_score ?? 'n/a'))}}</span>
            </div>
          </div>

          <div class="case-grid">
            <section class="case-panel">
              <h3>Backend</h3>
              <div class="kv">
                <div>Name</div><div>${{escapeHtml(backend.backend_name || 'unknown')}}</div>
                <div>Support Tier</div><div>${{escapeHtml(backend.support_tier || 'unknown')}}</div>
                <div>Verification</div><div>${{escapeHtml(backend.verification_mode || 'unknown')}}</div>
                <div>Repair</div><div>${{escapeHtml(item.repair_scope || 'none')}}</div>
              </div>
            </section>
            <section class="case-panel">
              <h3>Reason</h3>
              <div>${{escapeHtml(item.reason || 'No benchmark-side reason recorded.')}}</div>
            </section>
            <section class="case-panel">
              <h3>Error Codes</h3>
              <div class="code-list">${{errorPills}}</div>
            </section>
            <section class="case-panel">
              <h3>Source</h3>
              <div class="kv">
                <div>File</div><div>${{sourcePath}}</div>
                <div>Failure Family</div><div>${{escapeHtml(item.expected_failure_family || 'n/a')}}</div>
                <div>Exception</div><div>${{escapeHtml(item.exception_type || 'n/a')}}</div>
              </div>
            </section>
          </div>

          <details class="details">
            <summary>Details</summary>
            <div class="details-grid">
              <section class="case-panel">
                <h3>Expected Figure</h3>
                <div class="kv">
                  <div>Axes Count</div><div>${{escapeHtml(String(expectedFigure.axes_count ?? 'n/a'))}}</div>
                  <div>Figure Title</div><div>${{escapeHtml(expectedFigure.figure_title || 'n/a')}}</div>
                </div>
                <div class="mono" style="margin-top:10px;">${{expectedAxes}}</div>
              </section>
              <section class="case-panel">
                <h3>Actual Figure</h3>
                <div class="kv">
                  <div>Axes Count</div><div>${{escapeHtml(String(actualFigure.axes_count ?? 'n/a'))}}</div>
                  <div>Figure Title</div><div>${{escapeHtml(actualFigure.title || 'n/a')}}</div>
                  <div>Size</div><div>${{escapeHtml(JSON.stringify(actualFigure.size_inches ?? 'n/a'))}}</div>
                </div>
                <div class="mono" style="margin-top:10px;">${{actualAxes}}</div>
              </section>
              <section class="case-panel">
                <h3>Error Detail</h3>
                <div class="mono">${{errors.length ? errors.map(renderErrorLine).join('<br>') : 'n/a'}}</div>
              </section>
              <section class="case-panel">
                <h3>Repair</h3>
                <div class="kv">
                  <div>Level</div><div>${{escapeHtml(String(item.repair_level ?? 'n/a'))}}</div>
                  <div>Scope</div><div>${{escapeHtml(item.repair_scope || 'n/a')}}</div>
                  <div>Strategy</div><div>${{escapeHtml(item.repair_strategy || 'n/a')}}</div>
                </div>
                <div style="margin-top:10px;">${{escapeHtml(item.repair_instruction || 'n/a')}}</div>
                <div style="margin-top:10px;">${{escapeHtml(item.exception_message || '')}}</div>
              </section>
            </div>
          </details>
        </article>
      `;
    }}

    function renderAxisLine(axis) {{
      const parts = [
        `axis=${{axis.index ?? 'n/a'}}`,
        axis.projection ? `projection=${{axis.projection}}` : null,
        axis.title ? `title=${{axis.title}}` : null,
        axis.xlabel ? `xlabel=${{axis.xlabel}}` : null,
        axis.ylabel ? `ylabel=${{axis.ylabel}}` : null,
        axis.zlabel ? `zlabel=${{axis.zlabel}}` : null,
        axis.artist_types?.length ? `artists=${{axis.artist_types.join(',')}}` : null,
      ].filter(Boolean);
      return `<div>${{escapeHtml(parts.join(' | '))}}</div>`;
    }}

    function renderErrorLine(error) {{
      const bits = [
        error.code || 'unknown',
        error.operator ? `operator=${{error.operator}}` : null,
        error.expected !== undefined ? `expected=${{compactJson(error.expected)}}` : null,
        error.actual !== undefined ? `actual=${{compactJson(error.actual)}}` : null,
      ].filter(Boolean);
      return escapeHtml(bits.join(' | '));
    }}

    function compactJson(value) {{
      try {{
        const text = typeof value === 'string' ? value : JSON.stringify(value);
        return text.length > 120 ? text.slice(0, 117) + '...' : text;
      }} catch (error) {{
        return String(value);
      }}
    }}

    function formatRatio(value) {{
      if (typeof value !== 'number' || Number.isNaN(value)) return '0%';
      return `${{(value * 100).toFixed(1)}}%`;
    }}

    function escapeHtml(value) {{
      return String(value)
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#39;');
    }}

    function toFileHref(path) {{
      const normalized = String(path || '').replaceAll('\\\\', '/');
      if (/^[A-Za-z]:\\//.test(normalized)) {{
        return 'file:///' + normalized;
      }}
      return normalized;
    }}
  </script>
</body>
</html>
"""


def _escape_html(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
