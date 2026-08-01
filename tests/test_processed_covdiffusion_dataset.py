import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import zarr
from torch.utils.data import DataLoader

from utils.dataset.processed_covdiffusion_dataset import (
    SCHEMA_VERSION,
    ProcessedCovDiffusionDataset,
)


class ProcessedCovDiffusionDatasetTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.zarr_path = Path(self.tmp.name) / "train.zarr"

        root = zarr.open(str(self.zarr_path), mode="w")
        root.attrs["schema_version"] = SCHEMA_VERSION
        meta = root.create_group("meta")
        data = root.create_group("data")
        obs = root.create_group("obs")

        self.action = np.arange(8 * 24, dtype=np.float32).reshape(8, 24)
        self.stroke_ids = np.arange(8, dtype=np.float32).reshape(8, 1)
        self.point_clouds = np.stack(
            (
                np.zeros((5120, 3), dtype=np.float32),
                np.full((5120, 3), 2.0, dtype=np.float32),
            )
        )
        meta.create_dataset("episode_ends", data=np.array([4, 8], dtype=np.int64))
        data.create_dataset("action", data=self.action)
        data.create_dataset("stroke_ids", data=self.stroke_ids)
        obs.create_dataset("point_cloud", data=self.point_clouds)

    def tearDown(self):
        self.tmp.cleanup()

    def _dataset(self, n_obs_steps=1):
        return ProcessedCovDiffusionDataset(
            self.zarr_path,
            horizon=4,
            config=SimpleNamespace(n_obs_steps=n_obs_steps),
        )

    def test_sample_and_batch_only_materialize_consumed_observation_steps(self):
        dataset = self._dataset(n_obs_steps=1)
        first = dataset[0]
        second = dataset[1]

        self.assertEqual(len(dataset), 2)
        self.assertEqual(
            set(first), {"obs", "action", "prev_true_trajectory"}
        )
        self.assertEqual(set(first["obs"]), {"point_cloud"})
        self.assertEqual(tuple(first["obs"]["point_cloud"].shape), (1, 5120, 3))
        self.assertEqual(tuple(first["action"].shape), (4, 24))
        torch.testing.assert_close(first["action"], torch.from_numpy(self.action[:4]))
        torch.testing.assert_close(second["action"], torch.from_numpy(self.action[4:]))
        torch.testing.assert_close(
            first["prev_true_trajectory"], torch.from_numpy(self.action[0])
        )
        torch.testing.assert_close(
            second["prev_true_trajectory"], torch.from_numpy(self.action[4])
        )

        batch = next(iter(DataLoader(dataset, batch_size=2, shuffle=False)))
        self.assertEqual(tuple(batch["obs"]["point_cloud"].shape), (2, 1, 5120, 3))
        self.assertEqual(tuple(batch["action"].shape), (2, 4, 24))
        self.assertEqual(tuple(batch["prev_true_trajectory"].shape), (2, 24))
        torch.testing.assert_close(batch["action"][0], first["action"])
        torch.testing.assert_close(batch["action"][1], second["action"])
        torch.testing.assert_close(
            batch["prev_true_trajectory"][0], first["prev_true_trajectory"]
        )
        torch.testing.assert_close(
            batch["prev_true_trajectory"][1], second["prev_true_trajectory"]
        )

    def test_returned_point_clouds_do_not_share_mutable_storage(self):
        dataset = self._dataset(n_obs_steps=2)
        first = dataset[0]
        second = dataset[0]
        expected = float(self.point_clouds[0, 0, 0])

        first["obs"]["point_cloud"][0, 0, 0] = 99.0
        self.assertEqual(float(first["obs"]["point_cloud"][1, 0, 0]), expected)
        self.assertEqual(float(second["obs"]["point_cloud"][0, 0, 0]), expected)
        self.assertEqual(float(dataset.point_clouds[0, 0, 0]), expected)

    def test_normalizer_still_uses_one_point_cloud_per_episode(self):
        dataset = self._dataset(n_obs_steps=1)
        stats = dataset.get_normalizer().get_input_stats()

        torch.testing.assert_close(
            stats["point_cloud"]["mean"],
            torch.from_numpy(self.point_clouds).reshape(-1, 3).mean(dim=0),
        )
        torch.testing.assert_close(
            stats["action"]["mean"],
            torch.from_numpy(self.action).mean(dim=0),
        )


if __name__ == "__main__":
    unittest.main()
