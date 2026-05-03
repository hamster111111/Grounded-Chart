from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from grounded_chart.data.source_data import SourceDataExecutor, SourceDataPlanner


class SourceDataTest(unittest.TestCase):
    def test_source_data_plan_reads_csv_schema_preview_and_constraints(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "Imports.csv").write_text("Year,Urban,Rural\n2000,1.5,2\n2001,3,4\n", encoding="utf-8")

            plan = SourceDataPlanner().build_plan(
                workspace=root,
                instruction="Use Imports.csv to draw imports.",
            )

        self.assertEqual(("Imports.csv",), plan.mentioned_files)
        self.assertEqual("Imports.csv", plan.files[0].name)
        self.assertEqual(("Year", "Urban", "Rural"), plan.files[0].columns)
        self.assertEqual("wide_year_measure_table", plan.schema_constraints[0]["schema_type"])
        self.assertTrue(any("do not pivot on Category/Value" in item for item in plan.schema_constraints[0]["usage_constraints"]))

    def test_source_data_executor_loads_csv_rows_with_numeric_scalars(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "Imports.csv").write_text("Year,Urban,Rural\n2000,1.5,2\n", encoding="utf-8")
            plan = SourceDataPlanner().build_plan(workspace=root, instruction="Use Imports.csv")
            execution = SourceDataExecutor().execute(plan)

        table = execution.loaded_tables[0]
        self.assertEqual(["Year", "Urban", "Rural"], table["columns"])
        self.assertEqual({"Year": 2000, "Urban": 1.5, "Rural": 2}, table["rows"][0])
        self.assertFalse(table["truncated"])


if __name__ == "__main__":
    unittest.main()
