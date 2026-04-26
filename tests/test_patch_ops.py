import unittest

from grounded_chart import PatchAnchor, PatchOperation, apply_patch_operations, parse_patch_operations


class PatchOpsTest(unittest.TestCase):
    def test_apply_replace_call_arg_updates_title(self):
        code = """
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.set_title("Wrong Title")
""".strip()
        result = apply_patch_operations(
            code,
            (
                PatchOperation(
                    op="replace_call_arg",
                    anchor=PatchAnchor(kind="method_call", name="set_title", occurrence=1),
                    arg_index=0,
                    new_value="Sales by Category",
                ),
            ),
        )

        self.assertTrue(result.applied)
        self.assertIn("Sales by Category", result.code)

    def test_apply_remove_keyword_arg_updates_call(self):
        code = """
from matplotlib.sankey import Sankey
sankey.add(flows=[1, -1], labels=['a', 'b'], trunkcolor='red', patchlabel='a')
""".strip()
        result = apply_patch_operations(
            code,
            (
                PatchOperation(
                    op="remove_keyword_arg",
                    anchor=PatchAnchor(kind="method_call", name="add", occurrence=1),
                    keyword="trunkcolor",
                ),
            ),
        )

        self.assertTrue(result.applied)
        self.assertNotIn("trunkcolor", result.code)
        self.assertIn("patchlabel='a'", result.code)

    def test_parse_patch_operations_normalizes_payload(self):
        operations = parse_patch_operations(
            [
                {
                    "op": "replace_call_arg",
                    "anchor": {"kind": "method_call", "name": "set_xlabel", "occurrence": 1},
                    "arg_index": 0,
                    "new_value": "Category",
                    "description": "Fix x label.",
                }
            ]
        )

        self.assertEqual(1, len(operations))
        self.assertEqual("replace_call_arg", operations[0].op)
        self.assertEqual("set_xlabel", operations[0].anchor.name)
        self.assertEqual("Category", operations[0].new_value)

    def test_replace_keyword_arg_can_insert_safe_keyword(self):
        code = "ax.bar(['A'], [1])"
        result = apply_patch_operations(
            code,
            (
                PatchOperation(
                    op="replace_keyword_arg",
                    anchor=PatchAnchor(kind="method_call", name="bar", occurrence=1),
                    keyword="label",
                    new_value="Sales",
                ),
            ),
        )

        self.assertTrue(result.applied)
        self.assertEqual("ax.bar(['A'], [1], label='Sales')", result.code)

    def test_insert_after_anchor_supports_method_call_anchor(self):
        code = "ax.bar(['A'], [1])\nax.set_xlabel('Category')"
        result = apply_patch_operations(
            code,
            (
                PatchOperation(
                    op="insert_after_anchor",
                    anchor=PatchAnchor(kind="method_call", name="bar", occurrence=1),
                    new_value="ax.legend()",
                ),
            ),
        )

        self.assertTrue(result.applied)
        self.assertIn("ax.bar(['A'], [1])\nax.legend()\nax.set_xlabel", result.code)
    def test_replace_keyword_arg_preserves_multiline_call_layout(self):
        code = """sankey.add(
    flows=[1, -1],
    trunkcolor='red',
    labels=['a', 'b'],
)"""
        result = apply_patch_operations(
            code,
            (
                PatchOperation(
                    op="replace_keyword_arg",
                    anchor=PatchAnchor(kind="method_call", name="add", occurrence=1),
                    keyword="trunkcolor",
                    new_value="blue",
                ),
            ),
            max_changed_lines=2,
        )

        self.assertTrue(result.applied)
        self.assertIn("trunkcolor='blue'", result.code)
        self.assertIn("\n    labels=['a', 'b'],\n", result.code)

    def test_remove_keyword_arg_preserves_multiline_call_layout(self):
        code = """sankey.add(
    flows=[1, -1],
    trunkcolor='red',
    labels=['a', 'b'],
)"""
        result = apply_patch_operations(
            code,
            (
                PatchOperation(
                    op="remove_keyword_arg",
                    anchor=PatchAnchor(kind="method_call", name="add", occurrence=1),
                    keyword="trunkcolor",
                ),
            ),
            max_changed_lines=2,
        )

        self.assertTrue(result.applied)
        self.assertNotIn("trunkcolor", result.code)
        self.assertIn("\n    labels=['a', 'b'],\n", result.code)

if __name__ == "__main__":
    unittest.main()
