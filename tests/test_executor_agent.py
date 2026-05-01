from __future__ import annotations

import unittest

from grounded_chart.executor_agent import validate_executor_fidelity


CONTEXT = {
    "source_data_plan": {
        "files": [
            {"name": "Imports.csv"},
            {"name": "Consumption.csv"},
        ],
        "mentioned_files": ["Imports.csv", "Consumption.csv"],
    },
    "construction_plan": {
        "panels": [
            {
                "panel_id": "panel.main",
                "layers": [
                    {"layer_id": "layer.import_waterfall", "chart_type": "waterfall", "status": "explicit"},
                    {"layer_id": "layer.consumption_area", "chart_type": "area", "status": "explicit", "axis": "secondary"},
                ],
                "axes": {"secondary_y": "secondary quantitative axis"},
            },
            {
                "panel_id": "panel.pie_2002",
                "layers": [
                    {"layer_id": "layer.pie_2002", "chart_type": "pie", "status": "explicit"},
                ],
            },
        ],
        "global_elements": [
            {"type": "legend", "status": "explicit"},
            {"type": "title", "text": "Grain Import and Consumption Trends", "status": "explicit"},
        ],
    },
}


class ExecutorAgentTest(unittest.TestCase):
    def test_validator_accepts_code_that_follows_plan_contracts(self) -> None:
        code = """
import pandas as pd
import matplotlib.pyplot as plt
imports = pd.read_csv('Imports.csv')
consumption = pd.read_csv('Consumption.csv')
fig, ax = plt.subplots()
ax2 = ax.twinx()
ax.bar(imports['Year'], imports['Urban'])
ax2.stackplot(consumption['Year'], consumption['Urban'], consumption['Rural'])
pie_ax = fig.add_axes([0.2, 0.7, 0.1, 0.1])
pie_ax.pie([1, 2, 3])
ax.set_title('Grain Import and Consumption Trends')
ax.legend()
fig.savefig(OUTPUT_PATH)
"""
        report = validate_executor_fidelity(code, context=CONTEXT)

        self.assertTrue(report.ok)
        self.assertEqual((), report.issues)

    def test_validator_rejects_synthetic_data_and_missing_planned_layers(self) -> None:
        code = """
import numpy as np
import matplotlib.pyplot as plt
values = np.random.rand(10)
fig, ax = plt.subplots()
ax.plot(values)
fig.savefig(OUTPUT_PATH)
"""
        report = validate_executor_fidelity(code, context=CONTEXT)
        codes = {issue.code for issue in report.issues}

        self.assertFalse(report.ok)
        self.assertIn("missing_required_source_file", codes)
        self.assertIn("synthetic_data_with_required_source", codes)
        self.assertIn("missing_planned_visual_layer", codes)
        self.assertIn("missing_secondary_axis", codes)
        self.assertIn("missing_explicit_legend", codes)
        self.assertIn("missing_explicit_title_text", codes)

    def test_validator_rejects_mixed_x_basis_for_overlaid_layers(self) -> None:
        code = """
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
imports = pd.read_csv('Imports.csv')
consumption = pd.read_csv('Consumption.csv')
x_pos = np.arange(len(imports))
fig, ax = plt.subplots()
ax2 = ax.twinx()
ax.bar(x_pos, imports['Urban'])
ax2.fill_between(consumption['Year'], 0, consumption['Urban'])
ax.set_xticks(x_pos)
ax.set_xticklabels(imports['Year'])
ax.legend()
ax.set_title('Grain Import and Consumption Trends')
fig.savefig(OUTPUT_PATH)
"""
        report = validate_executor_fidelity(code, context=CONTEXT)
        codes = {issue.code for issue in report.issues}

        self.assertFalse(report.ok)
        self.assertIn("mixed_overlay_x_coordinate_basis", codes)

    def test_validator_rejects_overwriting_prepared_artifacts(self) -> None:
        context = {
            **CONTEXT,
            "artifact_workspace": {
                "artifacts": [
                    {"name": "step_02_imports_waterfall_values.csv"},
                    {"name": "step_03_consumption_area_values.csv"},
                ]
            },
        }
        code = """
import pandas as pd
import matplotlib.pyplot as plt
imports = pd.read_csv('Imports.csv')
imports.to_csv('execution/round_1/step_02_imports_waterfall_values.csv', index=False)
area = pd.read_csv('execution/round_1/step_03_consumption_area_values.csv')
fig, ax = plt.subplots()
ax.bar(imports['Year'], imports['Urban'])
ax2 = ax.twinx()
ax2.stackplot(area['Year'], area['Urban_area'], area['Rural_area'])
ax.legend()
ax.set_title('Grain Import and Consumption Trends')
fig.savefig(OUTPUT_PATH)
"""
        report = validate_executor_fidelity(code, context=context)
        codes = {issue.code for issue in report.issues}

        self.assertFalse(report.ok)
        self.assertIn("prepared_artifact_overwrite", codes)

    def test_validator_rejects_not_reading_prepared_artifacts(self) -> None:
        context = {
            **CONTEXT,
            "artifact_workspace": {
                "artifacts": [
                    {"name": "step_02_imports_waterfall_values.csv"},
                ]
            },
        }
        code = """
import pandas as pd
import matplotlib.pyplot as plt
imports = pd.read_csv('Imports.csv')
fig, ax = plt.subplots()
ax.bar(imports['Year'], imports['Urban'])
ax.legend()
ax.set_title('Grain Import and Consumption Trends')
fig.savefig(OUTPUT_PATH)
"""
        report = validate_executor_fidelity(code, context=context)
        codes = {issue.code for issue in report.issues}

        self.assertFalse(report.ok)
        self.assertIn("prepared_artifact_not_read", codes)

    def test_validator_accepts_prepared_artifacts_as_source_grounded_inputs(self) -> None:
        context = {
            **CONTEXT,
            "artifact_workspace": {
                "artifacts": [
                    {"name": "step_02_imports_waterfall_values.csv"},
                    {"name": "step_03_consumption_area_values.csv"},
                    {"name": "step_04_pie_values.csv"},
                ]
            },
        }
        code = """
import pandas as pd
import matplotlib.pyplot as plt
imports = pd.read_csv('execution/round_1/step_02_imports_waterfall_values.csv')
area = pd.read_csv('execution/round_1/step_03_consumption_area_values.csv')
pies = pd.read_csv('execution/round_1/step_04_pie_values.csv')
fig, ax = plt.subplots()
ax.bar(imports['Year'], imports['Urban'])
ax2 = ax.twinx()
ax2.stackplot(area['Year'], area['Urban'], area['Rural'])
pie_ax = fig.add_axes([0.2, 0.7, 0.1, 0.1])
pie_ax.pie(pies[pies['Year'] == 2002]['Consumption Ratio'])
ax.legend()
ax.set_title('Grain Import and Consumption Trends')
fig.savefig(OUTPUT_PATH)
"""
        report = validate_executor_fidelity(code, context=context)
        codes = {issue.code for issue in report.issues}

        self.assertNotIn("missing_required_source_file", codes)
        self.assertNotIn("unplanned_source_file_read", codes)

    def test_validator_rejects_bare_filename_reads_for_prepared_artifacts_with_relative_path(self) -> None:
        context = {
            **CONTEXT,
            "artifact_workspace": {
                "artifacts": [
                    {
                        "name": "step_02_imports_waterfall_render_table.csv",
                        "relative_path": "execution/round_1/step_02_imports_waterfall_render_table.csv",
                    },
                    {
                        "name": "step_03_consumption_area_values.csv",
                        "relative_path": "execution/round_1/step_03_consumption_area_values.csv",
                    },
                    {
                        "name": "step_04_pie_values.csv",
                        "relative_path": "execution/round_1/step_04_pie_values.csv",
                    },
                ],
                "metadata": {"chart_protocols": [{"chart_type": "waterfall"}]},
            },
        }
        code = """
import pandas as pd
import matplotlib.pyplot as plt
imports = pd.read_csv('step_02_imports_waterfall_render_table.csv')
area = pd.read_csv('execution/round_1/step_03_consumption_area_values.csv')
pies = pd.read_csv('execution/round_1/step_04_pie_values.csv')
fig, ax = plt.subplots()
ax.bar(imports['x_position'], imports['bar_height'], bottom=imports['bar_bottom'], width=imports['bar_width'])
ax2 = ax.twinx()
ax2.fill_between(area['x_index'], area['Urban_fill_bottom'], area['Urban_fill_top'], alpha=0.35)
pie_ax = fig.add_axes([0.2, 0.7, 0.1, 0.1])
pie_ax.pie(pies[pies['Year'] == 2002]['Consumption Ratio'])
ax.legend()
ax.set_title('Grain Import and Consumption Trends')
fig.savefig(OUTPUT_PATH)
"""
        report = validate_executor_fidelity(code, context=context)
        codes = {issue.code for issue in report.issues}

        self.assertFalse(report.ok)
        self.assertIn("prepared_artifact_bare_filename_read", codes)

    def test_validator_accepts_os_path_join_for_prepared_artifact_relative_path(self) -> None:
        context = {
            **CONTEXT,
            "artifact_workspace": {
                "artifacts": [
                    {
                        "name": "step_02_imports_waterfall_render_table.csv",
                        "relative_path": "execution/round_1/step_02_imports_waterfall_render_table.csv",
                    },
                    {
                        "name": "step_03_consumption_area_values.csv",
                        "relative_path": "execution/round_1/step_03_consumption_area_values.csv",
                    },
                    {
                        "name": "step_04_pie_values.csv",
                        "relative_path": "execution/round_1/step_04_pie_values.csv",
                    },
                ],
                "metadata": {"chart_protocols": [{"chart_type": "waterfall"}, {"chart_type": "area"}]},
            },
        }
        code = """
import os
import pandas as pd
import matplotlib.pyplot as plt
execution_dir = os.path.join(os.path.dirname(__file__), 'execution', 'round_1')
imports = pd.read_csv(os.path.join(execution_dir, 'step_02_imports_waterfall_render_table.csv'))
area = pd.read_csv(os.path.join(execution_dir, 'step_03_consumption_area_values.csv'))
pies = pd.read_csv(os.path.join(execution_dir, 'step_04_pie_values.csv'))
fig, ax = plt.subplots()
for _, row in imports.iterrows():
    colors = {'Urban': 'green', 'Rural': 'blue'}
    ax.bar(row['x_position'], row['bar_height'], bottom=row['bar_bottom'], width=row['bar_width'], color=colors[row['fill_color_role']])
ax2 = ax.twinx()
ax2.fill_between(area['x_index'], area['Urban_fill_bottom'], area['Urban_fill_top'], alpha=0.35)
ax2.fill_between(area['x_index'], area['Rural_fill_bottom'], area['Rural_fill_top'], alpha=0.35)
pie_ax = fig.add_axes([0.2, 0.7, 0.1, 0.1])
pie_ax.pie(pies[pies['Year'] == 2002]['Consumption Ratio'])
ax.legend()
ax.set_title('Grain Import and Consumption Trends')
fig.savefig(OUTPUT_PATH)
"""
        report = validate_executor_fidelity(code, context=context)
        codes = {issue.code for issue in report.issues}

        self.assertNotIn("prepared_artifact_bare_filename_read", codes)

    def test_validator_accepts_variable_execution_dir_for_prepared_artifact_relative_path(self) -> None:
        context = {
            **CONTEXT,
            "artifact_workspace": {
                "artifacts": [
                    {
                        "name": "step_02_imports_waterfall_render_table.csv",
                        "relative_path": "execution/round_1/step_02_imports_waterfall_render_table.csv",
                    },
                    {
                        "name": "step_03_consumption_area_values.csv",
                        "relative_path": "execution/round_1/step_03_consumption_area_values.csv",
                    },
                    {
                        "name": "step_04_pie_values.csv",
                        "relative_path": "execution/round_1/step_04_pie_values.csv",
                    },
                ],
                "metadata": {"chart_protocols": [{"chart_type": "waterfall"}, {"chart_type": "area"}]},
            },
        }
        code = """
import os
import pandas as pd
import matplotlib.pyplot as plt
execution_dir = 'execution/round_1'
imports = pd.read_csv(os.path.join(execution_dir, 'step_02_imports_waterfall_render_table.csv'))
area = pd.read_csv(os.path.join(execution_dir, 'step_03_consumption_area_values.csv'))
pies = pd.read_csv(os.path.join(execution_dir, 'step_04_pie_values.csv'))
fig, ax = plt.subplots()
for _, row in imports.iterrows():
    colors = {'Urban': 'green', 'Rural': 'blue'}
    ax.bar(row['x_position'], row['bar_height'], bottom=row['bar_bottom'], width=row['bar_width'], color=colors[row['fill_color_role']])
ax2 = ax.twinx()
ax2.fill_between(area['x_index'], area['Urban_fill_bottom'], area['Urban_fill_top'], alpha=0.35)
ax2.fill_between(area['x_index'], area['Rural_fill_bottom'], area['Rural_fill_top'], alpha=0.35)
pie_ax = fig.add_axes([0.2, 0.7, 0.1, 0.1])
pie_ax.pie(pies[pies['Year'] == 2002]['Consumption Ratio'])
ax.legend()
ax.set_title('Grain Import and Consumption Trends')
fig.savefig(OUTPUT_PATH)
"""
        report = validate_executor_fidelity(code, context=context)
        codes = {issue.code for issue in report.issues}

        self.assertNotIn("prepared_artifact_bare_filename_read", codes)

    def test_validator_rejects_waterfall_protocol_bypass(self) -> None:
        context = {
            **CONTEXT,
            "artifact_workspace": {
                "artifacts": [
                    {"name": "waterfall_protocol.json"},
                    {"name": "step_02_imports_waterfall_values.csv"},
                    {"name": "step_02_imports_waterfall_render_table.csv"},
                ],
                "metadata": {"chart_protocols": [{"chart_type": "waterfall"}]},
            },
        }
        code = """
import pandas as pd
import matplotlib.pyplot as plt
imports = pd.read_csv('execution/round_1/step_02_imports_waterfall_values.csv')
fig, ax = plt.subplots()
ax.bar(imports['x_index'], imports['Urban_plot'])
ax.legend()
ax.set_title('Grain Import and Consumption Trends')
fig.savefig(OUTPUT_PATH)
"""
        report = validate_executor_fidelity(code, context=context)
        codes = {issue.code for issue in report.issues}

        self.assertFalse(report.ok)
        self.assertIn("waterfall_render_table_not_read", codes)
        self.assertIn("waterfall_render_protocol_not_used", codes)
        self.assertIn("waterfall_values_table_used_as_ordinary_bars", codes)

    def test_validator_accepts_waterfall_render_table_protocol_use(self) -> None:
        context = {
            **CONTEXT,
            "artifact_workspace": {
                "artifacts": [
                    {"name": "waterfall_protocol.json"},
                    {"name": "step_02_imports_waterfall_render_table.csv"},
                    {"name": "step_03_consumption_area_values.csv"},
                    {"name": "step_04_pie_values.csv"},
                ],
                "metadata": {"chart_protocols": [{"chart_type": "waterfall"}]},
            },
        }
        code = """
import pandas as pd
import matplotlib.pyplot as plt
imports = pd.read_csv('execution/round_1/step_02_imports_waterfall_render_table.csv')
area = pd.read_csv('execution/round_1/step_03_consumption_area_values.csv')
pies = pd.read_csv('execution/round_1/step_04_pie_values.csv')
fig, ax = plt.subplots()
ax2 = ax.twinx()
ax.bar(imports['x_position'], imports['bar_height'], bottom=imports['bar_bottom'], width=imports['bar_width'])
ax2.stackplot(area['x_index'], area['Urban_area'], area['Rural_area'])
pie_ax = fig.add_axes([0.2, 0.7, 0.1, 0.1])
pie_ax.pie(pies[pies['Year'] == 2002]['Consumption Ratio'])
ax.legend()
ax.set_title('Grain Import and Consumption Trends')
fig.savefig(OUTPUT_PATH)
"""
        report = validate_executor_fidelity(code, context=context)
        codes = {issue.code for issue in report.issues}

        self.assertTrue(report.ok, [issue.to_dict() for issue in report.issues])
        self.assertNotIn("waterfall_render_protocol_not_used", codes)

    def test_validator_warns_on_legacy_waterfall_change_color_mapping(self) -> None:
        context = {
            **CONTEXT,
            "artifact_workspace": {
                "artifacts": [
                    {"name": "waterfall_protocol.json"},
                    {"name": "step_02_imports_waterfall_render_table.csv"},
                    {"name": "step_03_consumption_area_values.csv"},
                    {"name": "step_04_pie_values.csv"},
                ],
                "metadata": {"chart_protocols": [{"chart_type": "waterfall"}]},
            },
        }
        code = """
import pandas as pd
import matplotlib.pyplot as plt
imports = pd.read_csv('execution/round_1/step_02_imports_waterfall_render_table.csv')
area = pd.read_csv('execution/round_1/step_03_consumption_area_values.csv')
pies = pd.read_csv('execution/round_1/step_04_pie_values.csv')
fig, ax = plt.subplots()
for _, row in imports.iterrows():
    if row['color_role'] == 'positive':
        color = 'green'
    elif row['color_role'] == 'negative':
        color = 'red'
    else:
        color = 'blue'
    ax.bar(row['x_position'], row['bar_height'], bottom=row['bar_bottom'], width=row['bar_width'], color=color)
ax2 = ax.twinx()
ax2.stackplot(area['Year'], area['Urban'], area['Rural'])
pie_ax = fig.add_axes([0.2, 0.7, 0.1, 0.1])
pie_ax.pie(pies[pies['Year'] == 2002]['Consumption Ratio'])
ax.legend()
ax.set_title('Grain Import and Consumption Trends')
fig.savefig(OUTPUT_PATH)
"""
        report = validate_executor_fidelity(code, context=context)
        codes = {issue.code for issue in report.issues}

        self.assertTrue(report.ok)
        self.assertIn("waterfall_visual_channel_policy_bypass", codes)

    def test_warns_when_hard_visual_channel_contract_mapping_is_not_recorded(self) -> None:
        context = {
            "artifact_workspace": {
                "artifacts": [
                    {
                        "name": "artifact_waterfall_geometry.csv",
                        "relative_path": "execution/round_1/artifact_waterfall_geometry.csv",
                        "artifact_role": "waterfall_geometry",
                        "chart_type": "waterfall",
                        "schema": {"columns": ["x_position", "bar_height", "bar_bottom", "bar_width", "fill_color_role"]},
                    }
                ],
                "metadata": {
                    "chart_protocols": [
                        {
                            "chart_type": "waterfall",
                            "visual_channel_contracts": [
                                {
                                    "layer_id": "layer.waterfall",
                                    "channel": "fill_color",
                                    "semantic_dimension": "series_identity",
                                    "field": "fill_color_role",
                                    "strength": "hard",
                                    "executor_freedom": "choose_palette",
                                }
                            ],
                        }
                    ]
                },
            },
        }
        code = """
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('execution/round_1/artifact_waterfall_geometry.csv')
fig, ax = plt.subplots()
colors = {'A': 'blue', 'B': 'orange'}
for _, row in df.iterrows():
    ax.bar(row['x_position'], row['bar_height'], bottom=row['bar_bottom'], width=row['bar_width'], color=colors[row['fill_color_role']])
fig.savefig(OUTPUT_PATH)
"""
        report = validate_executor_fidelity(code, context=context)
        codes = {issue.code for issue in report.issues}

        self.assertTrue(report.ok)
        self.assertIn("missing_visual_channel_decisions_record", codes)

    def test_accepts_recorded_hard_visual_channel_contract_mapping(self) -> None:
        context = {
            "artifact_workspace": {
                "artifacts": [
                    {
                        "name": "artifact_waterfall_geometry.csv",
                        "relative_path": "execution/round_1/artifact_waterfall_geometry.csv",
                        "artifact_role": "waterfall_geometry",
                        "chart_type": "waterfall",
                        "schema": {"columns": ["x_position", "bar_height", "bar_bottom", "bar_width", "fill_color_role"]},
                    }
                ],
                "metadata": {
                    "chart_protocols": [
                        {
                            "chart_type": "waterfall",
                            "visual_channel_contracts": [
                                {
                                    "layer_id": "layer.waterfall",
                                    "channel": "fill_color",
                                    "semantic_dimension": "series_identity",
                                    "field": "fill_color_role",
                                    "strength": "hard",
                                    "executor_freedom": "choose_palette",
                                }
                            ],
                        }
                    ]
                },
            },
        }
        code = """
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('execution/round_1/artifact_waterfall_geometry.csv')
fig, ax = plt.subplots()
colors = {'A': 'blue', 'B': 'orange'}
for _, row in df.iterrows():
    ax.bar(row['x_position'], row['bar_height'], bottom=row['bar_bottom'], width=row['bar_width'], color=colors[row['fill_color_role']])
Path(OUTPUT_PATH).with_name('visual_channel_decisions.json').write_text(json.dumps({'layer.waterfall.fill_color': {'field': 'fill_color_role', 'mapping': colors}}))
fig.savefig(OUTPUT_PATH)
"""
        report = validate_executor_fidelity(code, context=context)
        codes = {issue.code for issue in report.issues}

        self.assertTrue(report.ok)
        self.assertNotIn("missing_visual_channel_decisions_record", codes)

    def test_soft_artifact_contract_does_not_force_prepared_artifact_read(self) -> None:
        context = {
            "artifact_workspace": {
                "artifacts": [
                    {
                        "name": "readability_hint.csv",
                        "relative_path": "execution/round_1/readability_hint.csv",
                        "artifact_role": "layout_hint",
                        "chart_type": "bar",
                        "required_for_plotting": True,
                        "contract_tier": "soft_guidance",
                    }
                ],
            },
        }
        code = """
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.bar([1], [2])
fig.savefig(OUTPUT_PATH)
"""
        report = validate_executor_fidelity(code, context=context)
        codes = {issue.code for issue in report.issues}

        self.assertTrue(report.ok)
        self.assertNotIn("prepared_artifact_not_read", codes)

    def test_soft_visual_channel_contract_does_not_require_decision_record(self) -> None:
        context = {
            "artifact_workspace": {
                "metadata": {
                    "chart_protocols": [
                        {
                            "chart_type": "bar",
                            "visual_channel_contracts": [
                                {
                                    "layer_id": "layer.bar",
                                    "channel": "fill_color",
                                    "semantic_dimension": "readability",
                                    "field": "category",
                                    "strength": "soft",
                                    "contract_tier": "soft_guidance",
                                }
                            ],
                        }
                    ]
                },
            },
        }
        code = """
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.bar([1], [2], color='steelblue')
fig.savefig(OUTPUT_PATH)
"""
        report = validate_executor_fidelity(code, context=context)
        codes = {issue.code for issue in report.issues}

        self.assertTrue(report.ok)
        self.assertNotIn("missing_visual_channel_decisions_record", codes)

    def test_warns_when_executor_computed_layout_is_not_recorded(self) -> None:
        context = {
            "construction_plan": {
                "layout_strategy": "main_axes_with_overlaid_insets",
                "panels": [
                    {
                        "panel_id": "panel.main",
                        "role": "primary_composite_chart",
                        "bounds": None,
                        "placement_policy": {"region": "main"},
                        "layers": [{"layer_id": "layer.bar", "chart_type": "bar", "role": "main"}],
                    },
                    {
                        "panel_id": "panel.pie_2008",
                        "role": "inset_pie_chart",
                        "bounds": None,
                        "anchor": {"type": "x_value", "value": 2008},
                        "placement_policy": {"region": "inside_main_axes_upper_band"},
                        "layers": [{"layer_id": "layer.pie", "chart_type": "pie", "role": "ratio"}],
                    },
                ],
                "global_elements": [{"type": "legend", "status": "explicit"}],
            }
        }
        code = """
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.bar([1], [2])
pie_ax = fig.add_axes([0.2, 0.7, 0.1, 0.1])
pie_ax.pie([1, 2])
ax.legend()
fig.savefig(OUTPUT_PATH)
"""
        report = validate_executor_fidelity(code, context=context)
        codes = {issue.code for issue in report.issues}

        self.assertIn("missing_computed_layout_json", codes)
        self.assertIn("missing_layout_decisions_md", codes)

    def test_accepts_recorded_executor_computed_layout(self) -> None:
        context = {
            "construction_plan": {
                "layout_strategy": "main_axes_with_overlaid_insets",
                "panels": [
                    {
                        "panel_id": "panel.main",
                        "role": "primary_composite_chart",
                        "bounds": None,
                        "placement_policy": {"region": "main"},
                        "layers": [{"layer_id": "layer.bar", "chart_type": "bar", "role": "main"}],
                    }
                ],
                "global_elements": [{"type": "legend", "status": "explicit"}],
            }
        }
        code = """
import json
from pathlib import Path
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.bar([1], [2])
ax.legend()
Path(OUTPUT_PATH).with_name('computed_layout.json').write_text(json.dumps({'panel.main': {'bounds': [0.1, 0.1, 0.8, 0.8]}}))
Path(OUTPUT_PATH).with_name('layout_decisions.md').write_text('# Layout decisions')
fig.savefig(OUTPUT_PATH)
"""
        report = validate_executor_fidelity(code, context=context)
        codes = {issue.code for issue in report.issues}

        self.assertNotIn("missing_computed_layout_json", codes)
        self.assertNotIn("missing_layout_decisions_md", codes)


if __name__ == "__main__":
    unittest.main()
