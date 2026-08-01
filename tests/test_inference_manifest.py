import hashlib
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "configs/inference/seed42_selected_episodes.json"
CATEGORIES = ("windows", "cuboids", "shelves", "containers")


class InferenceManifestTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile = json.loads(MANIFEST.read_text(encoding="utf-8"))

    def test_complete_and_raw_tensor_only_release(self):
        self.assertEqual(
            self.profile["format_version"], "3dcov-selected-inference-v1"
        )
        self.assertEqual(set(self.profile["categories"]), set(CATEGORIES))
        self.assertEqual(self.profile["protocol"]["weight_variant"], "raw")
        self.assertEqual(
            self.profile["protocol"]["condition_mode_sequence"],
            ["GT_Cond", "Pred_Cond"],
        )
        self.assertEqual(
            self.profile["protocol"]["reported_condition_mode"],
            "Pred_Cond",
        )
        for category, case in self.profile["categories"].items():
            self.assertEqual(case["training_seed"], 42)
            self.assertEqual(case["rollout_seed"], 42)
            self.assertEqual(
                case["source_checkpoint"]["loaded_state"], "state_dicts.model"
            )
            self.assertEqual(
                case["public_checkpoint"]["path"],
                f"raw/{category}/model.safetensors",
            )
            self.assertEqual(
                set(case["public_metric_reference"]),
                {"pcd", "jerk", "coverage"},
            )

    def test_all_integrity_values_are_sha256(self):
        for case in self.profile["categories"].values():
            values = [
                case["test_split_sha256"],
                case["evaluation_ready_sha256"],
                case["source_checkpoint"]["sha256"],
                case["public_checkpoint"]["model_sha256"],
                case["public_checkpoint"]["config_sha256"],
                case["public_checkpoint"]["metrics_sha256"],
                case["public_render_reference"]["prediction_ply_sha256"],
            ]
            for value in values:
                self.assertEqual(len(value), 64)
                int(value, 16)

    def test_selected_items_are_unique_and_in_range(self):
        sample_ids = set()
        for case in self.profile["categories"].values():
            self.assertGreaterEqual(case["test_split_index"], 0)
            self.assertLess(case["test_split_index"], case["test_split_size"])
            sample_ids.add(case["sample_id"])
        self.assertEqual(len(sample_ids), len(CATEGORIES))

    def test_evaluator_is_locked_by_file_content_not_an_old_git_tag(self):
        evaluation_code = self.profile["evaluation_code"]
        self.assertNotIn("git_tag", evaluation_code)
        self.assertTrue(evaluation_code["critical_files"])
        self.assertEqual(len(evaluation_code["sha256"]), 64)
        actual = {}
        for relative, digest in evaluation_code["critical_files"].items():
            path = ROOT / relative
            self.assertTrue(path.is_file(), relative)
            self.assertEqual(len(digest), 64)
            actual[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(actual[relative], digest, relative)
        aggregate = hashlib.sha256(
            json.dumps(actual, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        self.assertEqual(aggregate, evaluation_code["sha256"])


if __name__ == "__main__":
    unittest.main()
