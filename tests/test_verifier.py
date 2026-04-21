import unittest

from grounded_chart import DataPoint, PlotTrace
from grounded_chart.verifier import OperatorLevelVerifier


class OperatorLevelVerifierTest(unittest.TestCase):
    def test_detects_missing_groupby_style_length_mismatch(self):
        expected = PlotTrace("bar", (DataPoint("A", 25), DataPoint("B", 7)), source="expected")
        actual = PlotTrace("bar", (DataPoint("A", 10), DataPoint("A", 15), DataPoint("B", 7)), source="actual")
        report = OperatorLevelVerifier().verify(expected, actual)
        self.assertFalse(report.ok)
        self.assertIn("length_mismatch_extra_points", report.error_codes)

    def test_detects_wrong_aggregation_value(self):
        expected = PlotTrace("bar", (DataPoint("A", 25), DataPoint("B", 7)), source="expected")
        actual = PlotTrace("bar", (DataPoint("A", 2), DataPoint("B", 1)), source="actual")
        report = OperatorLevelVerifier().verify(expected, actual)
        self.assertFalse(report.ok)
        self.assertIn("wrong_aggregation_value", report.error_codes)

    def test_passes_matching_trace(self):
        expected = PlotTrace("bar", (DataPoint("A", 25), DataPoint("B", 7)), source="expected")
        actual = PlotTrace("bar", (DataPoint("A", 25.0), DataPoint("B", 7.0)), source="actual")
        report = OperatorLevelVerifier().verify(expected, actual)
        self.assertTrue(report.ok)

    def test_can_ignore_order_when_not_required(self):
        expected = PlotTrace("bar", (DataPoint("A", 25), DataPoint("B", 7)), source="expected")
        actual = PlotTrace("bar", (DataPoint("B", 7), DataPoint("A", 25)), source="actual")
        report = OperatorLevelVerifier().verify(expected, actual, enforce_order=False)

        self.assertTrue(report.ok)
        self.assertNotIn("wrong_order", report.error_codes)

    def test_detects_wrong_order_when_required(self):
        expected = PlotTrace("bar", (DataPoint("A", 25), DataPoint("B", 7)), source="expected")
        actual = PlotTrace("bar", (DataPoint("B", 7), DataPoint("A", 25)), source="actual")
        report = OperatorLevelVerifier().verify(expected, actual, enforce_order=True)

        self.assertFalse(report.ok)
        self.assertIn("wrong_order", report.error_codes)


if __name__ == "__main__":
    unittest.main()
