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
  repair_policy_mode: exploratory
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
        self.assertEqual("exploratory", config.repair_policy_mode)
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

    def test_repair_policy_mode_defaults_to_strict(self):
        content = """
run:
  parser_backend: heuristic
  repair_backend: rule
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"
            path.write_text(content, encoding="utf-8")
            config = load_ablation_run_config(path)

        self.assertEqual("strict", config.repair_policy_mode)

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


    def test_yaml_config_can_override_codegen_provider(self):
        content = """
run:
  parser_backend: heuristic
  repair_backend: rule

llm:
  default:
    provider: openai_compatible
    model: default-model
    base_url: https://api.default.test/v1
    api_key: sk-default
  codegen:
    provider: openai_compatible
    model: codegen-model
    base_url: https://api.codegen.test/v1
    api_key: sk-codegen
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"
            path.write_text(content, encoding="utf-8")
            config = load_ablation_run_config(path)

        self.assertEqual("codegen-model", config.codegen_provider.model)
        self.assertEqual("https://api.codegen.test/v1", config.codegen_provider.base_url)
        self.assertEqual("sk-codegen", config.codegen_provider.api_key)

if __name__ == "__main__":
    unittest.main()
