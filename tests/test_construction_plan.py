from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from grounded_chart.construction_plan import HeuristicChartConstructionPlanner, validate_construction_plan
from grounded_chart.source_data import SourceDataPlanner


CASE_98_QUERY = """Create a multi-layered graph using Python with my data, follow these streamlined steps:
1. Data Preparation:
  - Load 'Imports.csv' and 'Consumption.csv'. 'Imports.csv' contains 'Year', 'Urban', and 'Rural' for grain imports, and 'Consumption.csv' has similar columns for consumption data.
  - Load 'Grain_Consumption_Ratio.csv' for 2002, 2008, and 2016, with 'Year', 'Age Group', and 'Consumption Ratio'.
2. Graph Structure:
  - Title the graph as "Grain Import and Consumption Trends".
  - Create a Multi Category Waterfall Chart for urban and rural imports.
  - Plot a stacked area chart for consumption data on a secondary y-axis.
  - Embed pie charts for grain consumption ratios in 2002, 2008, and 2016.
  - Align the waterfall and area charts on a common x-axis (years), overlaying the pie charts at corresponding years.
  - Include legends for both imports and consumption.
"""


class ConstructionPlanTest(unittest.TestCase):
    def test_composite_chart_gets_whole_figure_layout_and_inset_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "Imports.csv").write_text("Year,Urban,Rural\n2000,1,2\n", encoding="utf-8")
            (root / "Consumption.csv").write_text("Year,Urban,Rural\n2000,3,4\n", encoding="utf-8")
            (root / "Grain_Consumption_Ratio.csv").write_text(
                "Year,Age Group,Consumption Ratio\n2002,0-14 years old,8\n",
                encoding="utf-8",
            )
            source_plan = SourceDataPlanner().build_plan(workspace=root, instruction=CASE_98_QUERY)

            plan = HeuristicChartConstructionPlanner().build_plan(
                query=CASE_98_QUERY,
                source_data_plan=source_plan,
            )

        self.assertEqual("main_axes_with_top_insets", plan.layout_strategy)
        self.assertEqual("chart_construction_plan_v2", plan.plan_type)
        panel_roles = [panel.role for panel in plan.panels]
        self.assertIn("primary_composite_chart", panel_roles)
        self.assertEqual(3, sum(1 for role in panel_roles if role == "inset_pie_chart"))
        main = plan.panels[0]
        layer_roles = {layer.role for layer in main.layers}
        self.assertIn("yearly_change_bars", layer_roles)
        self.assertIn("stacked_area", layer_roles)
        self.assertNotIn("trend_or_cumulative_total", layer_roles)
        waterfall = next(layer for layer in main.layers if layer.role == "yearly_change_bars")
        self.assertIn("connector_lines", {item["type"] for item in waterfall.components})
        channel_plan = waterfall.visual_channel_plan
        self.assertEqual("case_specific_waterfall_encoding_v2", channel_plan["channel_contract"])
        self.assertEqual("series", channel_plan["dimensions"]["series_identity"]["field"])
        self.assertEqual("change_role", channel_plan["dimensions"]["change_direction"]["field"])
        self.assertEqual("series", channel_plan["channel_allocation"]["x_group_offset"]["field"])
        self.assertEqual("series", channel_plan["channel_allocation"]["fill_color"]["field"])
        self.assertTrue(channel_plan["legend_policy"]["avoid_flattening_dimensions"])
        pie_panels = [panel for panel in plan.panels if panel.role == "inset_pie_chart"]
        self.assertTrue(all(panel.anchor.get("type") == "x_value" for panel in pie_panels))
        self.assertIn("data_transform_plan", plan.to_dict())
        self.assertTrue(any(decision.status == "inferred" for decision in plan.decisions))
        self.assertTrue(any(item["type"] == "legend" for item in plan.global_elements))

    def test_plan_validator_accepts_composite_plan_with_anchored_insets(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "Imports.csv").write_text("Year,Urban,Rural\n2000,1,2\n", encoding="utf-8")
            (root / "Consumption.csv").write_text("Year,Urban,Rural\n2000,3,4\n", encoding="utf-8")
            (root / "Grain_Consumption_Ratio.csv").write_text(
                "Year,Age Group,Consumption Ratio\n2002,0-14 years old,8\n",
                encoding="utf-8",
            )
            source_plan = SourceDataPlanner().build_plan(workspace=root, instruction=CASE_98_QUERY)
            plan = HeuristicChartConstructionPlanner().build_plan(
                query=CASE_98_QUERY,
                source_data_plan=source_plan,
            )

            report = validate_construction_plan(plan, query=CASE_98_QUERY, source_data_plan=source_plan)

        self.assertTrue(report.ok, [issue.to_dict() for issue in report.issues])

    def test_chart_type_detection_uses_word_boundaries(self) -> None:
        plan = HeuristicChartConstructionPlanner().build_plan(
            query="Create a streamlined bar chart of sales by year.",
        )
        layer_types = [layer.chart_type for panel in plan.panels for layer in panel.layers]

        self.assertIn("bar", layer_types)
        self.assertNotIn("line", layer_types)

    def test_area_modifiers_preserve_overlap_and_axis_range(self) -> None:
        query = (
            "Plot a stacked area chart for consumption data on a secondary y-axis, "
            "with translucent colors for urban and rural data to indicate overlapping consumption. "
            "Set y-axis scale from 35 to 105 kg."
        )
        plan = HeuristicChartConstructionPlanner().build_plan(query=query)
        area = next(layer for panel in plan.panels for layer in panel.layers if layer.chart_type == "area")

        self.assertEqual("overlap", area.semantic_modifiers["composition"])
        self.assertEqual("translucent", area.semantic_modifiers["opacity"])
        self.assertEqual("secondary", area.semantic_modifiers["axis_binding"])
        self.assertEqual({"type": "explicit_range", "min": 35.0, "max": 105.0}, area.semantic_modifiers["scale_policy"])
        self.assertEqual("overlap", area.encoding["composition"])


if __name__ == "__main__":
    unittest.main()
