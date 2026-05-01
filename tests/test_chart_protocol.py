from __future__ import annotations

import unittest

from grounded_chart.chart_protocol import ChartProtocolAgent, validate_protocol, waterfall_protocol


class ChartProtocolTest(unittest.TestCase):
    def test_waterfall_protocol_has_required_render_columns(self) -> None:
        protocol = waterfall_protocol()
        report = validate_protocol(protocol)

        self.assertTrue(report.ok, [issue.to_dict() for issue in report.issues])
        required = set(protocol.required_artifact_columns)
        self.assertIn("bar_bottom", required)
        self.assertIn("bar_height", required)
        self.assertIn("bar_top", required)
        self.assertIn("series", required)
        protocol_text = str(protocol.to_dict()).lower()
        self.assertIn("increase", protocol_text)
        self.assertIn("decrease", protocol_text)
        self.assertIn("total", protocol_text)

    def test_agent_uses_deterministic_waterfall_protocol(self) -> None:
        protocol = ChartProtocolAgent().build_protocol(chart_type="Waterfall")

        self.assertEqual("waterfall", protocol.chart_type)
        self.assertEqual("deterministic_fallback", protocol.source)
        self.assertIn("bottom", str(protocol.plotting_primitives).lower())


if __name__ == "__main__":
    unittest.main()
