import unittest

from grounded_chart_adapters import JsonCaseAdapter


class BenchMetadataTest(unittest.TestCase):
    def test_small_error_bench_has_failure_family_metadata(self):
        cases = list(JsonCaseAdapter("benchmarks/small_error_bench.json").iter_cases())

        self.assertTrue(cases)
        for case in cases:
            self.assertIn("failure_family", case.metadata)
            self.assertIn("repairability", case.metadata)
            self.assertIn("expected_repair_scope", case.metadata)

    def test_repair_loop_bench_declares_repair_expectations(self):
        cases = list(JsonCaseAdapter("benchmarks/repair_loop_bench.json").iter_cases())

        self.assertEqual(4, len(cases))
        auto_repairable = [case for case in cases if case.metadata.get("repairability") == "auto_repairable"]
        self.assertGreaterEqual(len(auto_repairable), 2)
        self.assertTrue(any(case.metadata.get("expected_improvement") == "no_auto_fix" for case in cases))


if __name__ == "__main__":
    unittest.main()
