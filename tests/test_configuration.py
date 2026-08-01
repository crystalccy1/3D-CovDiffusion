from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from omegaconf import OmegaConf
import yaml

from covdiffusion.configuration import load_args, pformat_dict, save_config


class ConfigurationTest(unittest.TestCase):
    def make_root(self, directory: str) -> Path:
        root = Path(directory) / "configs"
        root.mkdir()
        (root / "default.yaml").write_text(
            """\
value: default
nested:
  kept: true
  selected: default
dataset: windows-v2
""",
            encoding="utf-8",
        )
        (root / "first.yaml").write_text(
            """\
value: first
nested:
  selected: first
first_only: 1
""",
            encoding="utf-8",
        )
        (root / "second.yaml").write_text(
            """\
value: second
nested:
  selected: second
second_only: 2
""",
            encoding="utf-8",
        )
        return root

    def test_merge_order_and_cli_overrides(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.make_root(directory)
            config = load_args(
                root,
                [
                    "config=[first,second]",
                    "value=cli",
                    "nested.selected=cli",
                    "nested.added=3",
                ],
            )

        self.assertEqual(config.value, "cli")
        self.assertEqual(config.nested.selected, "cli")
        self.assertTrue(config.nested.kept)
        self.assertEqual(config.nested.added, 3)
        self.assertEqual(config.first_only, 1)
        self.assertEqual(config.second_only, 2)

    def test_list_fields_are_normalized(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.make_root(directory)
            config = load_args(root, ["dataset=[windows-v2,cuboids-v2]"])

        self.assertEqual(list(config.dataset), ["windows-v2", "cuboids-v2"])

    def test_missing_default_and_named_profile_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "configs"
            root.mkdir()
            with self.assertRaisesRegex(FileNotFoundError, "default.yaml"):
                load_args(root, [])

            (root / "default.yaml").write_text("value: 1\n", encoding="utf-8")
            with self.assertRaisesRegex(FileNotFoundError, "missing.yaml"):
                load_args(root, ["config=[missing]"])

    def test_profile_path_traversal_and_malformed_arguments_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.make_root(directory)
            with self.assertRaisesRegex(ValueError, "profile name"):
                load_args(root, ["config=[../private]"])
            with self.assertRaisesRegex(ValueError, "expected key=value"):
                load_args(root, ["seed"])

    def test_utf8_yaml_round_trip(self):
        original = OmegaConf.create(
            {
                "title": "三维覆盖轨迹",
                "dataset": ["windows-v2"],
                "nested": {"label": "几何条件"},
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            path = save_config(original, directory)
            raw = path.read_text(encoding="utf-8")
            loaded = yaml.safe_load(raw)

        self.assertIn("三维覆盖轨迹", raw)
        self.assertEqual(loaded["title"], "三维覆盖轨迹")
        self.assertEqual(loaded["nested"]["label"], "几何条件")

    def test_pretty_format_resolves_omegaconf(self):
        value = OmegaConf.create({"base": "data", "path": "${base}/windows"})
        formatted = pformat_dict(value)
        self.assertIn("'path': 'data/windows'", formatted)


if __name__ == "__main__":
    unittest.main()
