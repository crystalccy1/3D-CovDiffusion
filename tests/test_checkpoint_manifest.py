import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import dill
from omegaconf import OmegaConf
import torch

from train import (
    BEST_MANIFEST_FORMAT,
    TopKCheckpointManager,
    TrainCovDiffusionWorkspace,
    capture_torch_backend_state,
    restore_torch_backend_state,
)


class TopKCheckpointManifestTest(unittest.TestCase):
    monitor = "selection_metric"

    def metrics(self, score, epoch=0, global_step=1):
        return {
            self.monitor: score,
            "epoch": epoch,
            "global_step": global_step,
        }

    def save_candidate(self, manager, metrics):
        path = manager.get_ckpt_path(metrics)
        self.assertIsNotNone(path)
        path.write_bytes(b"checkpoint")
        manager.record_checkpoint(path, metrics)
        return path

    def test_manifest_preserves_full_precision_and_survives_move(self):
        score = 1.234567890123
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_dir = Path(temporary_directory) / "checkpoints"
            manager = TopKCheckpointManager(
                checkpoint_dir, monitor_key=self.monitor, mode="min"
            )
            path = self.save_candidate(manager, self.metrics(score))

            manifest = json.loads((checkpoint_dir / "best.json").read_text())
            self.assertEqual(manifest["format_version"], BEST_MANIFEST_FORMAT)
            self.assertEqual(manifest["best"]["score"], score)
            self.assertEqual(manifest["best"]["path"], path.name)

            restored = TopKCheckpointManager(
                checkpoint_dir, monitor_key=self.monitor, mode="min"
            )
            self.assertEqual(restored.checkpoints[0]["score"], score)
            self.assertEqual(restored.checkpoints[0]["path"], path)

    def test_better_checkpoint_is_published_before_old_file_is_removed(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_dir = Path(temporary_directory) / "checkpoints"
            manager = TopKCheckpointManager(
                checkpoint_dir, monitor_key=self.monitor, mode="min"
            )
            old_path = self.save_candidate(manager, self.metrics(2.0))
            new_path = self.save_candidate(
                manager, self.metrics(1.0, epoch=1, global_step=2)
            )

            self.assertFalse(old_path.exists())
            self.assertTrue(new_path.exists())
            manifest = json.loads((checkpoint_dir / "best.json").read_text())
            self.assertEqual(manifest["best"]["path"], new_path.name)

    def test_missing_candidate_cannot_change_selection_state(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_dir = Path(temporary_directory) / "checkpoints"
            manager = TopKCheckpointManager(
                checkpoint_dir, monitor_key=self.monitor, mode="min"
            )
            metrics = self.metrics(1.0)
            candidate = manager.get_ckpt_path(metrics)
            with self.assertRaisesRegex(RuntimeError, "missing or empty"):
                manager.record_checkpoint(candidate, metrics)
            self.assertFalse((checkpoint_dir / "best.json").exists())


class ResumeConfigurationTest(unittest.TestCase):
    def config(self, dataset="windows-v2", **overrides):
        values = {
            "dataset": dataset,
            "seed": 42,
            "batch_size": 512,
            "epochs": 4800,
            "model": {"backbone": "dp3"},
            "processed_data_root": "/old/artifacts/dataset",
            "evaluation_cache_root": "/old/artifacts/evaluation-cache",
            "output_dir": "/old/run",
        }
        values.update(overrides)
        return OmegaConf.create(values)

    def workspace(self, config, output_dir):
        workspace = TrainCovDiffusionWorkspace.__new__(TrainCovDiffusionWorkspace)
        workspace.cfg = config
        workspace._output_dir = str(output_dir)
        workspace.model = Mock()
        workspace.ema_model = Mock()
        workspace.optimizer = Mock()
        workspace.global_step = 0
        workspace.epoch = 0
        workspace.latest_rollout_monitor_metric = None
        workspace._resume_rng_state = None
        return workspace

    def checkpoint(self, path, config):
        path.parent.mkdir(parents=True)
        payload = {
            "cfg": config,
            "state_dicts": {"model": {}, "ema_model": {}, "optimizer": {}},
            "pickles": {},
            "rng_state": None,
        }
        with path.open("wb") as handle:
            torch.save(payload, handle, pickle_module=dill)

    def test_matching_resume_config_allows_relocated_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / "checkpoints/latest.ckpt"
            self.checkpoint(checkpoint_path, self.config())
            current_config = self.config(
                processed_data_root="/new/artifacts/dataset",
                evaluation_cache_root="/new/artifacts/evaluation-cache",
                output_dir="/new/run",
                resume=True,
            )
            workspace = self.workspace(current_config, temporary_directory)

            workspace.load_checkpoint(checkpoint_path)

            workspace.model.load_state_dict.assert_called_once_with({})

    def test_cross_category_resume_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_path = Path(temporary_directory) / "checkpoints/latest.ckpt"
            self.checkpoint(checkpoint_path, self.config(dataset="windows-v2"))
            current_config = self.config(dataset="cuboids-v2", resume=True)
            workspace = self.workspace(current_config, temporary_directory)

            with self.assertRaisesRegex(
                ValueError, r"dataset: checkpoint='windows-v2', current='cuboids-v2'"
            ):
                workspace.load_checkpoint(checkpoint_path)

            workspace.model.load_state_dict.assert_not_called()

    def test_checkpoint_restores_torch_backend_state(self):
        original_state = capture_torch_backend_state()
        saved_state = dict(original_state)
        tested_flags = [
            "cudnn.deterministic",
            "cudnn.benchmark",
            "cudnn.allow_tf32",
            "cuda.matmul.allow_tf32",
        ]
        tested_flags = [key for key in tested_flags if key in saved_state]
        if not tested_flags:
            self.skipTest("Torch exposes no tested CUDA backend flags")
        for key in tested_flags:
            saved_state[key] = not saved_state[key]

        try:
            restore_torch_backend_state(saved_state)
            with tempfile.TemporaryDirectory() as temporary_directory:
                checkpoint_path = Path(temporary_directory) / "checkpoints/latest.ckpt"
                workspace = self.workspace(self.config(), temporary_directory)
                workspace.model.state_dict.return_value = {}
                workspace.ema_model.state_dict.return_value = {}
                workspace.optimizer.state_dict.return_value = {}
                workspace.save_checkpoint(checkpoint_path)

                restore_torch_backend_state(original_state)
                workspace.load_checkpoint(checkpoint_path)

                restored_state = capture_torch_backend_state()
                for key, value in saved_state.items():
                    self.assertEqual(restored_state[key], value, key)
        finally:
            restore_torch_backend_state(original_state)


if __name__ == "__main__":
    unittest.main()
