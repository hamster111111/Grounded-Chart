import os
import tempfile
import unittest
from pathlib import Path

from grounded_chart import load_ablation_run_config


class AblationConfigTest(unittest.TestCase):
    def test_loads_deepseek_style_yaml_config(self):
        content = """
run:
  bench: repair_loop_bench
  output_name: repair_loop_bench_deepseek
  parser_backend: llm
  repair_backend: rule
  enable_repair_loop: true
  max_repair_rounds: 2

llm:
  default:
    provider: openai_compatible
    model: deepseek-chat
    base_url: https://api.deepseek.com
    api_key: sk-test
    temperature: 0.0
    max_tokens: 1200
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"
            path.write_text(content, encoding="utf-8")
            config = load_ablation_run_config(path)

        self.assertEqual("repair_loop_bench", config.bench_name)
        self.assertEqual("repair_loop_bench_deepseek", config.output_name)
        self.assertEqual("llm", config.parser_backend)
        self.assertEqual("rule", config.repair_backend)
        self.assertTrue(config.enable_repair_loop)
        self.assertEqual(2, config.max_repair_rounds)
        self.assertEqual("deepseek-chat", config.parser_provider.model)
        self.assertEqual("https://api.deepseek.com", config.parser_provider.base_url)
        self.assertEqual("sk-test", config.parser_provider.api_key)
        self.assertFalse(config.parser_provider.no_proxy)

    def test_config_can_resolve_api_key_from_environment(self):
        content = """
run:
  parser_backend: heuristic
  repair_backend: llm

llm:
  default:
    provider: openai_compatible
    model: deepseek-chat
    base_url: https://api.deepseek.com
    api_key_env: DEEPSEEK_API_KEY
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"
            path.write_text(content, encoding="utf-8")
            previous = os.environ.get("DEEPSEEK_API_KEY")
            os.environ["DEEPSEEK_API_KEY"] = "sk-env"
            try:
                config = load_ablation_run_config(path)
            finally:
                if previous is None:
                    os.environ.pop("DEEPSEEK_API_KEY", None)
                else:
                    os.environ["DEEPSEEK_API_KEY"] = previous

        self.assertEqual("sk-env", config.repair_provider.api_key)

    def test_yaml_config_can_enable_no_proxy_for_domestic_endpoint(self):
        content = """
run:
  parser_backend: heuristic
  repair_backend: llm

llm:
  default:
    provider: openai_compatible
    model: deepseek-chat
    base_url: https://api.deepseek.com/v1
    api_key: sk-test
    no_proxy: true
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"
            path.write_text(content, encoding="utf-8")
            config = load_ablation_run_config(path)

        self.assertTrue(config.repair_provider.no_proxy)

    def test_toml_config_is_still_supported_for_back_compat(self):
        content = """
[run]
parser_backend = "heuristic"
repair_backend = "rule"
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.toml"
            path.write_text(content, encoding="utf-8")
            config = load_ablation_run_config(path)

        self.assertEqual("heuristic", config.parser_backend)
        self.assertEqual("rule", config.repair_backend)


if __name__ == "__main__":
    unittest.main()
