import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from utils.dataset.canonical_evaluation_dataset import (
    CanonicalEvaluationDataset,
    MANIFEST_FILENAME,
    MANIFEST_VERSION,
    SCHEMA_VERSION,
    validate_evaluation_sample,
)
from utils.dataset.covdiffusion_rollout_dataset import CovDiffusionRolloutDataset


class _Config:
    dataset = ["windows-v2"]


class CanonicalEvaluationDatasetTest(unittest.TestCase):
    def _fixture(self, *, sample_id="fixed_sample", mutate=None):
        temporary = tempfile.TemporaryDirectory()
        dataset_root = Path(temporary.name) / "dataset"
        cache_root = dataset_root / "evaluation-cache"
        category_root = cache_root / "windows-v2"
        category_root.mkdir(parents=True)

        arrays = {
            "schema_version": np.array(SCHEMA_VERSION),
            "sample_id": np.array(sample_id),
            "point_cloud": np.arange(5120 * 3, dtype=np.float64).reshape(5120, 3),
            "trajectory": np.arange(3 * 24, dtype=np.float64).reshape(3, 24) / 100.0,
            "gt_trajectory": np.arange(10 * 6, dtype=np.float64).reshape(10, 6) / 100.0,
            "stroke_ids": np.arange(10, dtype=np.int64) // 4,
            "mesh_vertices": np.arange(12, dtype=np.float32).reshape(4, 3),
            "mesh_faces": np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64),
        }
        if mutate is not None:
            mutate(arrays)
        path = category_root / f"{sample_id}.npz"
        np.savez(path, **arrays)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest = {
            "format_version": MANIFEST_VERSION,
            "schema_version": SCHEMA_VERSION,
            "categories": {
                "windows-v2": {
                    "split": "test",
                    "sample_ids": [sample_id],
                    "files": {sample_id: digest},
                }
            },
        }
        (dataset_root / MANIFEST_FILENAME).write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        return temporary, cache_root, arrays, path

    def test_loads_evaluation_ready_npz_without_preprocessing(self):
        temporary, cache_root, expected, _ = self._fixture()
        self.addCleanup(temporary.cleanup)
        dataset = CanonicalEvaluationDataset(
            cache_root, "windows-v2", verify_hashes=True
        )

        self.assertEqual(len(dataset), 1)
        item = dataset[0]
        self.assertEqual(item["sample_id"], "fixed_sample")
        for key in (
            "point_cloud",
            "trajectory",
            "gt_trajectory",
            "stroke_ids",
            "mesh_vertices",
            "mesh_faces",
        ):
            np.testing.assert_array_equal(item[key], expected[key])

    def test_rollout_adapter_preserves_runner_contract(self):
        temporary, cache_root, expected, _ = self._fixture()
        self.addCleanup(temporary.cleanup)
        dataset = CovDiffusionRolloutDataset(
            config=_Config(), evaluation_cache_root=cache_root
        )

        item = dataset[0]
        self.assertEqual(item["obs"]["point_cloud"].shape, (1, 5120, 3))
        np.testing.assert_array_equal(item["full_trajectory"], expected["trajectory"])
        np.testing.assert_array_equal(item["traj_as_pc"], expected["gt_trajectory"])
        np.testing.assert_array_equal(item["gt_traj_as_pc"], expected["gt_trajectory"])
        np.testing.assert_array_equal(item["mesh_vertices"], expected["mesh_vertices"])
        np.testing.assert_array_equal(item["mesh_faces"], expected["mesh_faces"])
        self.assertEqual(item["sample_id"], "fixed_sample")
        self.assertEqual(item["episode_idx"], 0)

    def test_rejects_current_v1_cache_missing_geometry_and_chunks(self):
        current_v1 = {
            "point_cloud": np.zeros((5120, 3), dtype=np.float64),
            "traj": np.zeros((10, 6), dtype=np.float64),
            "stroke_ids": np.zeros(10, dtype=np.float64),
        }
        with self.assertRaisesRegex(ValueError, "missing"):
            validate_evaluation_sample(current_v1)

    def test_rejects_padding_or_misaligned_ground_truth(self):
        temporary, cache_root, _, _ = self._fixture(
            mutate=lambda arrays: arrays["trajectory"].__setitem__((0, 0), -100.0)
        )
        self.addCleanup(temporary.cleanup)
        with self.assertRaisesRegex(RuntimeError, "invalid evaluation sample"):
            CanonicalEvaluationDataset(cache_root, "windows-v2")[0]

        valid = {
            "schema_version": np.array(SCHEMA_VERSION),
            "sample_id": np.array("sample"),
            "point_cloud": np.zeros((5120, 3), dtype=np.float32),
            "trajectory": np.zeros((1, 24), dtype=np.float32),
            "gt_trajectory": np.zeros((2, 6), dtype=np.float32),
            "stroke_ids": np.zeros(3, dtype=np.int64),
            "mesh_vertices": np.zeros((3, 3), dtype=np.float32),
            "mesh_faces": np.array([[0, 1, 2]], dtype=np.int64),
        }
        with self.assertRaisesRegex(ValueError, "aligned"):
            validate_evaluation_sample(valid)

    def test_rejects_manifest_order_or_hash_drift(self):
        temporary, cache_root, _, path = self._fixture()
        self.addCleanup(temporary.cleanup)
        manifest_path = cache_root.parent / MANIFEST_FILENAME
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["categories"]["windows-v2"]["sample_ids"].append("missing")
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "exactly cover"):
            CanonicalEvaluationDataset(cache_root, "windows-v2")

        manifest["categories"]["windows-v2"]["sample_ids"] = ["fixed_sample"]
        manifest["categories"]["windows-v2"]["files"]["fixed_sample"] = "0" * 64
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            CanonicalEvaluationDataset(
                cache_root, "windows-v2", verify_hashes=True
            )[0]
        self.assertTrue(path.is_file())

    def test_requires_v2_manifest_and_safe_sample_id(self):
        temporary, cache_root, _, _ = self._fixture()
        self.addCleanup(temporary.cleanup)
        manifest_path = cache_root.parent / MANIFEST_FILENAME
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["format_version"] = "3dcov-evaluation-cache-v1"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Regenerate"):
            CanonicalEvaluationDataset(cache_root, "windows-v2")

        arrays = {
            "schema_version": np.array(SCHEMA_VERSION),
            "sample_id": np.array("../unsafe"),
            "point_cloud": np.zeros((5120, 3), dtype=np.float32),
            "trajectory": np.zeros((1, 24), dtype=np.float32),
            "gt_trajectory": np.zeros((1, 6), dtype=np.float32),
            "stroke_ids": np.zeros(1, dtype=np.int64),
            "mesh_vertices": np.zeros((3, 3), dtype=np.float32),
            "mesh_faces": np.array([[0, 1, 2]], dtype=np.int64),
        }
        with self.assertRaisesRegex(ValueError, "safe path"):
            validate_evaluation_sample(arrays)


if __name__ == "__main__":
    unittest.main()
