import unittest

from examples.build_matplotbench_expected_artifact_repair_bench import _reconcile_figure_requirements


class MatplotbenchExpectedArtifactBuilderTest(unittest.TestCase):
    def test_reconciles_axes_count_with_panel_contract_count(self):
        figure_requirements = {"axes_count": 4, "source_spans": {"axes_count": "2x2 bar plots plus bottom plot"}}
        contracts = [
            {"artifact_type": "panel_chart_type", "locator": {"panel_id": "bar1"}},
            {"artifact_type": "panel_chart_type", "locator": {"panel_id": "bar2"}},
            {"artifact_type": "panel_chart_type", "locator": {"panel_id": "bar3"}},
            {"artifact_type": "panel_chart_type", "locator": {"panel_id": "bar4"}},
            {"artifact_type": "panel_chart_type", "locator": {"panel_id": "cosine"}},
            {"artifact_type": "connector", "locator": {"panel_id": "cosine"}},
        ]

        reconciled, metadata = _reconcile_figure_requirements(figure_requirements, contracts)

        self.assertEqual(5, reconciled["axes_count"])
        self.assertEqual(contracts, reconciled["artifact_contracts"])
        self.assertEqual("panel_contract_count_exceeds_axes_count", metadata["reason"])
        self.assertEqual(4, metadata["original"])
        self.assertEqual(5, metadata["reconciled"])

    def test_does_not_lower_existing_axes_count(self):
        figure_requirements = {"axes_count": 6}
        contracts = [
            {"artifact_type": "panel_chart_type", "locator": {"panel_id": "panel_0"}},
            {"artifact_type": "panel_chart_type", "locator": {"panel_id": "panel_1"}},
        ]

        reconciled, metadata = _reconcile_figure_requirements(figure_requirements, contracts)

        self.assertEqual(6, reconciled["axes_count"])
        self.assertIsNone(metadata)

    def test_does_not_invent_axes_count_from_partial_panel_contracts(self):
        figure_requirements = {"axes_count": None}
        contracts = [
            {"artifact_type": "panel_chart_type", "locator": {"panel_id": "panel_0"}},
            {"artifact_type": "panel_chart_type", "locator": {"panel_id": "panel_1"}},
        ]

        reconciled, metadata = _reconcile_figure_requirements(figure_requirements, contracts)

        self.assertIsNone(reconciled["axes_count"])
        self.assertIsNone(metadata)


if __name__ == "__main__":
    unittest.main()