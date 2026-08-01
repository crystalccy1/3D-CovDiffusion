import hashlib
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import reproduce


class ReproduceCliTest(unittest.TestCase):
    def test_evaluation_ready_cache_preserves_order_and_hashes(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            artifact_root = Path(temporary_directory) / "artifacts"
            dataset_name = "windows-v2"
            sample_id = "object-001"
            cache_path = (
                artifact_root
                / "dataset"
                / reproduce.EVALUATION_CACHE_DIRECTORY
                / dataset_name
                / f"{sample_id}.npz"
            )
            cache_path.parent.mkdir(parents=True)
            np.savez(
                cache_path,
                schema_version=np.asarray("3dcov-evaluation-v2"),
                sample_id=np.asarray(sample_id),
                point_cloud=np.zeros((5120, 3), dtype=np.float32),
                trajectory=np.zeros((1, 24), dtype=np.float32),
                gt_trajectory=np.zeros((1, 6), dtype=np.float32),
                stroke_ids=np.zeros((1,), dtype=np.int64),
                mesh_vertices=np.asarray(
                    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                    dtype=np.float32,
                ),
                mesh_faces=np.asarray([[0, 1, 2]], dtype=np.int64),
            )
            cache_sha = hashlib.sha256(cache_path.read_bytes()).hexdigest()
            split_sha = "1" * 64
            manifest = {
                "format_version": "3dcov-evaluation-ready-v2",
                "schema_version": "3dcov-evaluation-v2",
                "categories": {
                    dataset_name: {
                        "split": "test",
                        "sample_ids": [sample_id],
                        "test_split_sha256": split_sha,
                        "files": {sample_id: cache_sha},
                    }
                },
            }
            (artifact_root / "dataset" / reproduce.EVALUATION_CACHE_MANIFEST).write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            profile = {
                "categories": {
                    "windows": {
                        "test_split_sha256": split_sha,
                        "test_split_size": 1,
                        "test_split_index": 0,
                        "sample_id": sample_id,
                        "evaluation_ready_sha256": cache_sha,
                    }
                }
            }
            with patch(
                "reproduce.load_selected_inference_profile", return_value=profile
            ):
                self.assertEqual(
                    reproduce.validate_evaluation_cache(artifact_root, "windows"),
                    cache_path.parent.resolve(),
                )

    def test_doctor_checks_one_complete_runtime(self):
        args = reproduce.build_parser().parse_args(["doctor"])
        self.assertFalse(args.visualization)
        visual = reproduce.build_parser().parse_args(
            ["doctor", "--visualization"]
        )
        self.assertTrue(visual.visualization)

    def test_doctor_rejects_an_unavailable_cuda_runtime(self):
        args = reproduce.build_parser().parse_args(["doctor"])

        def fake_import(name):
            if name == "torch":
                return SimpleNamespace(
                    cuda=SimpleNamespace(is_available=lambda: False)
                )
            return object()

        with patch("reproduce.importlib.import_module", side_effect=fake_import):
            with self.assertRaisesRegex(SystemExit, "dependency check"):
                with redirect_stdout(io.StringIO()):
                    reproduce.cmd_doctor(args)

    def test_doctor_accepts_an_available_cuda_runtime(self):
        args = reproduce.build_parser().parse_args(["doctor"])

        def fake_import(name):
            if name == "torch":
                return SimpleNamespace(
                    cuda=SimpleNamespace(
                        is_available=lambda: True,
                        get_device_name=lambda index: "test GPU",
                    )
                )
            return object()

        with patch("reproduce.importlib.import_module", side_effect=fake_import):
            output = io.StringIO()
            with redirect_stdout(output):
                reproduce.cmd_doctor(args)

        self.assertIn("OK      CUDA", output.getvalue())

    def test_git_state_includes_untracked_files_in_dirty_check(self):
        outputs = {
            ("rev-parse", "HEAD"): "a" * 40,
            ("status", "--porcelain", "--untracked-files=all"): "?? local.py",
            ("tag", "--points-at", "HEAD"): "inference-v1",
        }

        def fake_run(command, **kwargs):
            key = tuple(command[1:])
            return type("Completed", (), {"stdout": outputs[key]})()

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / ".git").mkdir()
            with patch("reproduce.ROOT", root), patch(
                "reproduce.subprocess.run", side_effect=fake_run
            ):
                state = reproduce.current_git_state()

        self.assertTrue(state["dirty"])
        self.assertEqual(state["tags"], ["inference-v1"])

    def test_train_command_uses_release_config_stack_and_profile(self):
        command = reproduce.train_command(
            category="windows",
            data_root=Path("/tmp/data"),
            seed=42,
            epochs=4800,
            batch_size=None,
            workers=0,
        )
        self.assertEqual(command[:2], [sys.executable, "train.py"])
        self.assertIn(
            "config=[covdiffusion,windows]",
            command,
        )
        self.assertIn("epochs=4800", command)
        self.assertIn("batch_size=512", command)
        self.assertIn("workers=0", command)
        self.assertIn("training.replay_epoch_sampling=true", command)
        self.assertFalse(any(item.startswith("wandb=") for item in command))

    def test_train_smoke_disables_checkpoints_and_limits_steps(self):
        command = reproduce.train_command(
            category="shelves",
            data_root=Path("/tmp/data"),
            seed=123,
            epochs=1,
            batch_size=2,
            workers=0,
            max_train_steps=1,
            save_checkpoint=False,
            replay_epoch_sampling=False,
        )
        self.assertIn("training.max_train_steps=1", command)
        self.assertIn("checkpoint.save_ckpt=false", command)
        self.assertIn("training.replay_epoch_sampling=false", command)

    def test_train_resume_targets_existing_run_directory(self):
        run_directory = Path("/tmp/existing-run")
        command = reproduce.train_command(
            category="windows",
            data_root=Path("/tmp/data"),
            seed=42,
            epochs=4800,
            batch_size=None,
            workers=0,
            resume_from=run_directory,
        )
        self.assertIn("resume=true", command)
        self.assertIn(
            f"output_dir={run_directory.resolve()}",
            command,
        )

    def test_train_command_targets_an_explicit_new_run_directory(self):
        run_directory = Path("/tmp/artifacts/runs/windows/seed42")
        command = reproduce.train_command(
            category="windows",
            data_root=Path("/tmp/data"),
            output_dir=run_directory,
        )
        self.assertIn(f"output_dir={run_directory.resolve()}", command)
        self.assertNotIn("resume=true", command)

    def test_normal_train_uses_stable_seed42_directory(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            artifact_root = Path(temporary_directory) / "artifacts"
            (artifact_root / "dataset/data/windows-v2/train.zarr").mkdir(
                parents=True
            )
            args = reproduce.build_parser().parse_args(
                [
                    "train",
                    "windows",
                    "--artifact-root",
                    str(artifact_root),
                ]
            )
            with patch("reproduce.validate_evaluation_cache"), patch(
                "reproduce.run_checked"
            ) as run_checked:
                reproduce.cmd_train(args)

            command = run_checked.call_args.args[0]
            expected = (artifact_root / "runs/windows/seed42").resolve()
            self.assertIn(f"output_dir={expected}", command)
            self.assertNotIn("resume=true", command)
            self.assertNotIn("WORKDIR", run_checked.call_args.kwargs["env"])
            self.assertEqual(
                run_checked.call_args.kwargs["env"]["PYTORCH_CUDA_ALLOC_CONF"],
                "max_split_size_mb:128",
            )

    def test_normal_train_refuses_a_nonempty_seed42_directory(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            artifact_root = Path(temporary_directory) / "artifacts"
            run_dir = artifact_root / "runs/windows/seed42"
            run_dir.mkdir(parents=True)
            (run_dir / "config.yaml").write_text("seed: 42\n", encoding="utf-8")
            args = reproduce.build_parser().parse_args(
                [
                    "train",
                    "windows",
                    "--artifact-root",
                    str(artifact_root),
                ]
            )
            with patch("reproduce.run_checked") as run_checked:
                with self.assertRaisesRegex(FileExistsError, "--resume-from"):
                    reproduce.cmd_train(args)
            run_checked.assert_not_called()

    def test_smoke_train_keeps_a_temporary_random_run(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            artifact_root = Path(temporary_directory) / "artifacts"
            (artifact_root / "dataset/data/windows-v2/train.zarr").mkdir(
                parents=True
            )
            smoke_root = Path(temporary_directory) / "smoke"
            args = reproduce.build_parser().parse_args(
                [
                    "train",
                    "windows",
                    "--artifact-root",
                    str(artifact_root),
                    "--smoke",
                ]
            )
            with patch("reproduce.tempfile.mkdtemp", return_value=str(smoke_root)):
                with patch("reproduce.validate_evaluation_cache"), patch(
                    "reproduce.run_checked"
                ) as run_checked:
                    reproduce.cmd_train(args)

            command = run_checked.call_args.args[0]
            self.assertFalse(any(item.startswith("output_dir=") for item in command))
            self.assertEqual(
                run_checked.call_args.kwargs["env"]["WORKDIR"], str(smoke_root)
            )

    def test_processed_training_can_attach_evaluation_ready_root(self):
        command = reproduce.train_command(
            category="cuboids",
            data_root=Path("/tmp/train-ready"),
            evaluation_data_root=Path("/tmp/evaluation-ready"),
            seed=42,
            epochs=4800,
            batch_size=None,
            workers=0,
            rollout_episodes=1,
        )
        self.assertIn(
            f"processed_data_root={Path('/tmp/train-ready').resolve()}",
            command,
        )
        self.assertIn(
            "evaluation_cache_root="
            f"{Path('/tmp/evaluation-ready').resolve()}",
            command,
        )
        self.assertIn("training.eval_episodes=1", command)

    def test_training_profile_matches_selected_checkpoint_counters(self):
        profile = reproduce.training_profile()
        self.assertEqual(profile["common"]["epochs"], 4800)
        self.assertEqual(
            profile["dataset_release"]["revision"],
            reproduce.DATASET_REVISION,
        )
        self.assertEqual(
            profile["categories"]["windows"]["batches_per_epoch"] * 166,
            profile["categories"]["windows"]["selected_checkpoint_global_step"],
        )
        self.assertEqual(
            profile["categories"]["containers"]["batch_size"], 128
        )

    def test_evaluate_command_is_metrics_only_by_default(self):
        command = reproduce.evaluate_command(
            category="cuboids",
            checkpoint=Path("/tmp/model.safetensors"),
            config=Path("/tmp/config.yaml"),
            episodes=0,
            workers=2,
            rollout_seed=7,
            training_seed=42,
            output_dir=Path("/tmp/out"),
            coverage_mode="area-weighted",
            spray_radius=0.1,
            save_artifacts=False,
        )
        self.assertIn("cuboids-v2", command)
        self.assertNotIn("--save_artifacts", command)
        self.assertEqual(command[command.index("--eval_episodes") + 1], "0")
        self.assertEqual(command[command.index("--run_name") + 1], "cuboids-s42")
        self.assertEqual(command[command.index("--seed") + 1], "7")
        self.assertEqual(command[command.index("--training_seed") + 1], "42")

    def test_legacy_checkpoint_does_not_require_public_config(self):
        command = reproduce.evaluate_command(
            category="windows",
            checkpoint=Path("/tmp/training.ckpt"),
            config=None,
            episodes=1,
            workers=0,
            rollout_seed=42,
            training_seed=None,
            output_dir=Path("/tmp/out"),
            coverage_mode="face-count",
            spray_radius=0.1,
            save_artifacts=False,
        )
        self.assertNotIn("--config_path", command)
        self.assertNotIn("--training_seed", command)

    def test_training_run_resolves_manifest_checkpoint_and_saved_config(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_dir = Path(temporary_directory) / "seed42"
            checkpoint_dir = run_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True)
            checkpoint = checkpoint_dir / "epoch=4.ckpt"
            checkpoint.write_bytes(b"checkpoint")
            config = run_dir / "config.yaml"
            config.write_text("seed: 123\n", encoding="utf-8")
            (checkpoint_dir / "best.json").write_text(
                json.dumps(
                    {
                        "format_version": "3dcov-best-checkpoint-v1",
                        "best": {"path": checkpoint.name, "score": 1.25},
                    }
                ),
                encoding="utf-8",
            )

            self.assertEqual(
                reproduce.resolve_training_run(run_dir),
                (checkpoint.resolve(), config.resolve(), 123),
            )

    def test_training_run_manifest_cannot_escape_checkpoint_directory(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_dir = Path(temporary_directory) / "seed42"
            checkpoint_dir = run_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True)
            (run_dir / "outside.ckpt").write_bytes(b"checkpoint")
            (checkpoint_dir / "best.json").write_text(
                json.dumps(
                    {
                        "format_version": "3dcov-best-checkpoint-v1",
                        "best": {"path": "../outside.ckpt"},
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "escapes"):
                reproduce.resolve_training_run(run_dir)

    def test_selected_episode_flags_are_forwarded(self):
        command = reproduce.evaluate_command(
            category="windows",
            checkpoint=Path("/tmp/model.safetensors"),
            config=Path("/tmp/config.yaml"),
            episodes=1,
            workers=4,
            rollout_seed=42,
            training_seed=42,
            output_dir=Path("/tmp/out"),
            coverage_mode="face-count",
            spray_radius=0.1,
            save_artifacts=True,
            run_name="windows-selected",
            weight_variant="raw",
            single_episode_idx=5,
            include_gt_conditioned=True,
        )
        self.assertEqual(command[command.index("--weight_variant") + 1], "raw")
        self.assertEqual(command[command.index("--single_episode_idx") + 1], "5")
        self.assertIn("--save_artifacts", command)
        self.assertIn("--include_gt_conditioned", command)

    def test_selected_profile_has_all_categories(self):
        profile = reproduce.load_selected_inference_profile()
        self.assertEqual(set(profile["categories"]), set(reproduce.CATEGORIES))
        self.assertEqual(
            profile["categories"]["containers"]["test_split_index"], 1
        )
        self.assertEqual(profile["protocol"]["weight_variant"], "raw")

    def test_public_cli_has_four_direct_pipeline_commands(self):
        prepare = reproduce.build_parser().parse_args(["prepare", "all"])
        data = reproduce.build_parser().parse_args(
            ["prepare", "windows", "--data-only"]
        )
        checkpoint = reproduce.build_parser().parse_args(
            ["prepare", "cuboids", "--checkpoint-only"]
        )
        train = reproduce.build_parser().parse_args(["train", "windows"])
        evaluate = reproduce.build_parser().parse_args(["evaluate", "cuboids"])
        infer = reproduce.build_parser().parse_args(["infer", "shelves"])
        self.assertEqual(prepare.category, "all")
        self.assertTrue(data.data_only)
        self.assertTrue(checkpoint.checkpoint_only)
        self.assertEqual(train.category, "windows")
        self.assertEqual(evaluate.category, "cuboids")
        self.assertEqual(infer.category, "shelves")

    def test_prepare_modes_are_mutually_exclusive(self):
        with patch("sys.stderr"), self.assertRaises(SystemExit):
            reproduce.build_parser().parse_args(
                ["prepare", "windows", "--data-only", "--checkpoint-only"]
            )

    def test_data_download_uses_one_worker_to_avoid_hub_throttling(self):
        with tempfile.TemporaryDirectory() as temporary_directory, patch(
            "reproduce._snapshot_download"
        ) as snapshot_download_factory:
            reproduce._download_hf_data(
                Path(temporary_directory), ["windows"]
            )

        download = snapshot_download_factory.return_value
        download.assert_called_once()
        self.assertEqual(download.call_args.kwargs["max_workers"], 1)
        self.assertIn(
            "data/windows-v2/**",
            download.call_args.kwargs["allow_patterns"],
        )

    def test_data_only_prepare_skips_checkpoint_download(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            args = reproduce.build_parser().parse_args(
                [
                    "prepare",
                    "windows",
                    "--data-only",
                    "--artifact-root",
                    temporary_directory,
                ]
            )
            with (
                patch("reproduce._download_hf_data") as download_data,
                patch("reproduce._validate_prepared_data") as validate_data,
                patch("reproduce._download_hf_checkpoints") as download_checkpoints,
                patch(
                    "reproduce._validate_prepared_checkpoints"
                ) as validate_checkpoints,
            ):
                reproduce.cmd_prepare(args)
            download_data.assert_called_once()
            validate_data.assert_called_once()
            download_checkpoints.assert_not_called()
            validate_checkpoints.assert_not_called()

    def test_checkpoint_only_prepare_downloads_ema_without_data(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            args = reproduce.build_parser().parse_args(
                [
                    "prepare",
                    "shelves",
                    "--checkpoint-only",
                    "--artifact-root",
                    temporary_directory,
                ]
            )
            with (
                patch("reproduce._download_hf_data") as download_data,
                patch("reproduce._validate_prepared_data") as validate_data,
                patch("reproduce._download_hf_checkpoints") as download_checkpoints,
                patch(
                    "reproduce._validate_prepared_checkpoints"
                ) as validate_checkpoints,
            ):
                reproduce.cmd_prepare(args)
            download_data.assert_not_called()
            validate_data.assert_not_called()
            download_checkpoints.assert_called_once_with(
                Path(temporary_directory).resolve(), ["shelves"], include_raw=False
            )
            validate_checkpoints.assert_called_once_with(
                Path(temporary_directory).resolve(),
                ["shelves"],
                "cpu",
                variants=("ema",),
            )

    def test_default_prepare_keeps_data_ema_and_raw_compatibility(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            args = reproduce.build_parser().parse_args(
                [
                    "prepare",
                    "containers",
                    "--artifact-root",
                    temporary_directory,
                ]
            )
            with (
                patch("reproduce._download_hf_data") as download_data,
                patch("reproduce._validate_prepared_data") as validate_data,
                patch("reproduce._download_hf_checkpoints") as download_checkpoints,
                patch(
                    "reproduce._validate_prepared_checkpoints"
                ) as validate_checkpoints,
            ):
                reproduce.cmd_prepare(args)
            artifact_root = Path(temporary_directory).resolve()
            download_data.assert_called_once_with(artifact_root, ["containers"])
            validate_data.assert_called_once_with(artifact_root, ["containers"])
            download_checkpoints.assert_called_once_with(
                artifact_root, ["containers"], include_raw=True
            )
            validate_checkpoints.assert_called_once_with(
                artifact_root,
                ["containers"],
                "cpu",
                variants=("ema", "raw"),
            )

    def test_public_train_cli_has_no_raw_training_branch(self):
        with patch("sys.stderr"), self.assertRaises(SystemExit):
            reproduce.build_parser().parse_args(
                ["train", "windows", "--raw-data-root", "/tmp/raw"]
            )

    def test_evaluate_checkpoint_and_run_dir_are_mutually_exclusive(self):
        with patch("sys.stderr"), self.assertRaises(SystemExit):
            reproduce.build_parser().parse_args(
                [
                    "evaluate",
                    "windows",
                    "--checkpoint",
                    "/tmp/model.ckpt",
                    "--run-dir",
                    "/tmp/run",
                ]
            )

    def test_evaluate_run_dir_uses_legacy_ema_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            run_dir = root / "run"
            checkpoint_dir = run_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True)
            checkpoint = checkpoint_dir / "best.ckpt"
            checkpoint.write_bytes(b"checkpoint")
            (run_dir / "config.yaml").write_text("seed: 42\n", encoding="utf-8")
            (checkpoint_dir / "best.json").write_text(
                json.dumps(
                    {
                        "format_version": "3dcov-best-checkpoint-v1",
                        "best": {"path": checkpoint.name},
                    }
                ),
                encoding="utf-8",
            )
            args = reproduce.build_parser().parse_args(
                ["evaluate", "windows", "--run-dir", str(run_dir)]
            )
            resolved = reproduce._resolve_evaluation_checkpoint(args, "windows")
            self.assertEqual(
                resolved,
                (checkpoint.resolve(), (run_dir / "config.yaml").resolve(), 42),
            )

    def test_evaluate_run_dir_requires_one_category(self):
        args = reproduce.build_parser().parse_args(
            ["evaluate", "all", "--run-dir", "/tmp/run"]
        )
        with self.assertRaisesRegex(ValueError, "one category"):
            reproduce.cmd_evaluate(args)

    def test_evaluation_cache_follows_custom_artifact_root(self):
        args = reproduce.build_parser().parse_args(
            ["prepare", "windows", "--artifact-root", "/tmp/custom-artifacts"]
        )
        self.assertEqual(
            reproduce.evaluation_cache_root(args.artifact_root),
            Path("/tmp/custom-artifacts/dataset/evaluation-cache").resolve(),
        )

    def test_unversioned_inference_escape_hatch_is_not_public(self):
        with patch("sys.stderr"), self.assertRaises(SystemExit):
            reproduce.build_parser().parse_args(
                ["infer", "windows", "--allow-unversioned-code"]
            )

    def test_complete_training_has_no_incomplete_public_switch(self):
        args = reproduce.build_parser().parse_args(["train", "windows"])
        self.assertEqual(
            reproduce.evaluation_cache_root(args.artifact_root),
            Path("artifacts/dataset/evaluation-cache").resolve(),
        )
        with patch("sys.stderr"), self.assertRaises(SystemExit):
            reproduce.build_parser().parse_args(
                ["train", "windows", "--train-only"]
            )


if __name__ == "__main__":
    unittest.main()
