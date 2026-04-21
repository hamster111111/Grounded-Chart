import os
import unittest

from grounded_chart.llm import _apply_no_proxy_for_base_url, _merge_no_proxy_value, _merge_no_proxy_values


class NoProxyHelpersTest(unittest.TestCase):
    def test_merge_no_proxy_value_appends_host_once(self):
        merged = _merge_no_proxy_value("localhost,127.0.0.1", "api.deepseek.com")
        self.assertEqual("localhost,127.0.0.1,api.deepseek.com", merged)

        merged_again = _merge_no_proxy_value(merged, "api.deepseek.com")
        self.assertEqual("localhost,127.0.0.1,api.deepseek.com", merged_again)

    def test_apply_no_proxy_for_base_url_updates_both_env_keys(self):
        old_upper = os.environ.get("NO_PROXY")
        old_lower = os.environ.get("no_proxy")
        os.environ["NO_PROXY"] = "localhost,127.0.0.1"
        try:
            _apply_no_proxy_for_base_url("https://api.deepseek.com/v1")
            self.assertEqual("localhost,127.0.0.1,api.deepseek.com", os.environ["NO_PROXY"])
            self.assertEqual("localhost,127.0.0.1,api.deepseek.com", os.environ["no_proxy"])
        finally:
            if old_upper is None:
                os.environ.pop("NO_PROXY", None)
            else:
                os.environ["NO_PROXY"] = old_upper
            if old_lower is None:
                os.environ.pop("no_proxy", None)
            else:
                os.environ["no_proxy"] = old_lower

    def test_merge_no_proxy_values_combines_upper_and_lower(self):
        merged = _merge_no_proxy_values("localhost", "127.0.0.1", "api.deepseek.com")
        self.assertEqual("localhost,127.0.0.1,api.deepseek.com", merged)


if __name__ == "__main__":
    unittest.main()
