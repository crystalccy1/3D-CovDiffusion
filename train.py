"""Train one 3D-CovDiffusion v2 category.

The public training path intentionally contains only the components needed to
reproduce the released experiments: processed train-ready data, optional
evaluation-ready rollouts, Adam, EMA, deterministic checkpoint/resume, and
historical top-k selection.
"""

from __future__ import annotations

import copy
import json
import math
import os
import pathlib
import random
import string

import dill
import numpy as np
from omegaconf import ListConfig, OmegaConf
import torch
from tqdm import tqdm

from covdiffusion.common.pytorch_util import dict_apply, optimizer_to
from covdiffusion.configuration import load_args, pformat_dict, save_config
from covdiffusion.model.diffusion.ema_model import EMAModel
from models import get_model_diffusion
from utils.dataset.processed_covdiffusion_dataset import ProcessedCovDiffusionDataset


ROLLOUT_INTERVAL = 5
ROLLOUT_CONDITION_MODES = ("GT_Cond", "Pred_Cond")
TOPK_MONITOR = "pred_cond_Pred_Cond_mean_historical_pose_chamfer_distance"
BEST_MANIFEST_FORMAT = "3dcov-best-checkpoint-v1"
RESUME_RELOCATABLE_CONFIG_KEYS = frozenset(
    {"output_dir", "processed_data_root", "evaluation_cache_root", "resume"}
)
TORCH_BACKEND_FLAG_PATHS = {
    "cudnn.enabled": (torch.backends.cudnn, "enabled"),
    "cudnn.deterministic": (torch.backends.cudnn, "deterministic"),
    "cudnn.benchmark": (torch.backends.cudnn, "benchmark"),
    "cudnn.allow_tf32": (torch.backends.cudnn, "allow_tf32"),
    "cuda.matmul.allow_tf32": (torch.backends.cuda.matmul, "allow_tf32"),
    "cuda.matmul.allow_fp16_reduced_precision_reduction": (
        torch.backends.cuda.matmul,
        "allow_fp16_reduced_precision_reduction",
    ),
    "cuda.matmul.allow_bf16_reduced_precision_reduction": (
        torch.backends.cuda.matmul,
        "allow_bf16_reduced_precision_reduction",
    ),
}


def capture_torch_backend_state():
    """Capture global Torch switches that can change CUDA numerical behavior."""

    state = {}
    if hasattr(torch, "get_deterministic_debug_mode"):
        state["deterministic_debug_mode"] = int(
            torch.get_deterministic_debug_mode()
        )
    for key, (namespace, attribute) in TORCH_BACKEND_FLAG_PATHS.items():
        try:
            value = getattr(namespace, attribute)
        except (AttributeError, AssertionError):
            # Torch 1.13's backend proxy raises AssertionError for some flags
            # that only exist in newer releases.
            continue
        state[key] = bool(value)
    return state


def restore_torch_backend_state(state):
    """Restore a checkpoint's Torch backend state without changing fresh runs."""

    if not isinstance(state, dict):
        raise ValueError("Checkpoint Torch backend state must be a mapping.")

    debug_mode = state.get("deterministic_debug_mode")
    if debug_mode is not None and hasattr(torch, "set_deterministic_debug_mode"):
        if type(debug_mode) is not int or debug_mode not in {0, 1, 2}:
            raise ValueError("Invalid deterministic_debug_mode in checkpoint.")
        torch.set_deterministic_debug_mode(debug_mode)

    for key, (namespace, attribute) in TORCH_BACKEND_FLAG_PATHS.items():
        if key not in state:
            continue
        try:
            getattr(namespace, attribute)
        except (AttributeError, AssertionError):
            continue
        value = state[key]
        if type(value) is not bool:
            raise ValueError(f"Invalid boolean Torch backend flag {key!r}.")
        setattr(namespace, attribute, value)


def _resume_config(config):
    """Return the exact training configuration minus relocatable paths."""

    if config is None:
        raise ValueError("Resume checkpoint has no saved training configuration.")
    resolved = OmegaConf.to_container(config, resolve=True, enum_to_str=True)
    if not isinstance(resolved, dict):
        raise ValueError("Resume checkpoint training configuration is not a mapping.")
    return {
        key: value
        for key, value in resolved.items()
        if key not in RESUME_RELOCATABLE_CONFIG_KEYS
    }


def validate_resume_config(checkpoint_config, current_config):
    """Refuse to resume a checkpoint under a different experiment profile."""

    checkpoint = _resume_config(checkpoint_config)
    current = _resume_config(current_config)
    if checkpoint == current:
        return

    missing = object()
    changed_keys = sorted(
        key
        for key in set(checkpoint) | set(current)
        if checkpoint.get(key, missing) != current.get(key, missing)
    )
    details = ", ".join(changed_keys)
    if checkpoint.get("dataset") != current.get("dataset"):
        details = (
            f"dataset: checkpoint={checkpoint.get('dataset')!r}, "
            f"current={current.get('dataset')!r}; changed keys: {details}"
        )
    raise ValueError(
        "Resume checkpoint configuration does not match the current CLI "
        f"configuration; refusing to mix experiments. {details}"
    )


def get_output_dir(cfg):
    if getattr(cfg, "output_dir", None):
        return os.path.expanduser(str(cfg.output_dir))
    if os.environ.get("WORKDIR"):
        return os.path.expanduser(os.environ["WORKDIR"])
    return "runs"


def load_training_config():
    """Load the release config stack and enforce the public one-category path."""

    cfg = load_args(root="configs/covdiffusion")
    cfg.task_name = "CovDiffusion"

    dataset = cfg.dataset
    if isinstance(dataset, (list, tuple, ListConfig)):
        if len(dataset) != 1:
            raise ValueError("Public reproduction supports exactly one category per run.")
        dataset = dataset[0]
    if not dataset:
        raise ValueError("A v2 category config is required.")
    cfg.dataset = str(dataset)

    cfg.model.backbone = "dp3"
    cfg.overfitting = False

    if not hasattr(cfg, "processed_data_root"):
        cfg.processed_data_root = None
    if not hasattr(cfg, "evaluation_cache_root"):
        cfg.evaluation_cache_root = None

    # Direct train.py calls may omit evaluation data. reproduce.py supplies the
    # self-contained cache so checkpoint selection uses the fixed test split.
    cfg.skip_rollout_eval = not bool(cfg.evaluation_cache_root)

    if not hasattr(cfg, "shape_meta"):
        cfg.shape_meta = OmegaConf.create(
            {
                "obs": {
                    "point_cloud": {"shape": [5120, 3]},
                    "low_dim": {"shape": [24]},
                },
                "action": {"shape": [24]},
            }
        )
    return cfg


class TopKCheckpointManager:
    """Keep top-k rollout checkpoints with exact, crash-safe resume metadata."""

    def __init__(self, save_dir, k=1, monitor_key=TOPK_MONITOR, mode="min"):
        self.save_dir = pathlib.Path(save_dir)
        self.k = int(k)
        self.monitor_key = monitor_key
        self.mode = mode
        if self.k < 1:
            raise ValueError("Top-k checkpoint count must be at least one.")
        if self.mode not in {"min", "max"}:
            raise ValueError("Top-k checkpoint mode must be 'min' or 'max'.")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.save_dir / "best.json"
        self.checkpoints = self._restore_existing_checkpoints()

    def _sort(self, checkpoints):
        return sorted(
            checkpoints,
            key=lambda item: item["score"],
            reverse=self.mode == "max",
        )

    def _record_from_manifest(self, item):
        path = pathlib.Path(item["path"])
        if path.is_absolute() or path.name != str(path):
            raise ValueError("best.json paths must be checkpoint filenames")
        path = self.save_dir / path
        if not path.is_file():
            raise FileNotFoundError(path)
        score = float(item["score"])
        if not math.isfinite(score):
            raise ValueError("best.json score is not finite")
        return {
            "score": score,
            "path": path,
            "epoch": int(item["epoch"]),
            "global_step": int(item["global_step"]),
        }

    def _restore_manifest(self):
        if not self.manifest_path.is_file():
            return None
        try:
            with self.manifest_path.open(encoding="utf-8") as handle:
                manifest = json.load(handle)
            if manifest.get("format_version") != BEST_MANIFEST_FORMAT:
                raise ValueError("unsupported best.json format_version")
            if (
                manifest.get("monitor") != self.monitor_key
                or manifest.get("mode") != self.mode
            ):
                raise ValueError("best.json monitor or mode does not match this run")
            best = self._record_from_manifest(manifest["best"])
            items = manifest.get("checkpoints") or [manifest["best"]]
            recovered = [self._record_from_manifest(item) for item in items]
            if not recovered:
                raise ValueError("best.json contains no checkpoints")
            recovered = self._sort(recovered)
            if recovered[0] != best:
                raise ValueError("best.json best entry does not match checkpoint ranking")
        except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as error:
            print(f"Ignoring invalid best.json ({error}); using filename fallback.")
            return None
        return recovered[: self.k]

    def _record_from_filename(self, path):
        name = path.name
        epoch_token, remainder = name.split("-step=", 1)
        step_token, score_token = remainder.split(f"-{self.monitor_key}=", 1)
        score = float(pathlib.Path(score_token).stem.replace("p", "."))
        return {
            "score": score,
            "path": path,
            "epoch": int(epoch_token.removeprefix("epoch=")),
            "global_step": int(step_token),
        }

    def _restore_existing_checkpoints(self):
        recovered = self._restore_manifest()
        if recovered is not None:
            return recovered

        recovered = []
        pattern = f"epoch=*-{self.monitor_key}=*.ckpt"
        for path in self.save_dir.glob(pattern):
            try:
                recovered.append(self._record_from_filename(path))
            except (IndexError, TypeError, ValueError):
                print(f"Ignoring checkpoint with an invalid score: {path}")
        return self._sort(recovered)[: self.k]

    def get_ckpt_path(self, metrics):
        if self.monitor_key not in metrics:
            return None
        score = float(metrics[self.monitor_key])
        if not math.isfinite(score):
            raise ValueError(f"Top-k score must be finite, got {score}.")
        should_add = len(self.checkpoints) < self.k
        if not should_add:
            boundary = self.checkpoints[-1]["score"]
            should_add = (self.mode == "max" and score > boundary) or (
                self.mode == "min" and score < boundary
            )
        if not should_add:
            return None

        score_token = f"{score:.4f}".replace(".", "p")
        return self.save_dir / (
            f"epoch={metrics['epoch']}-step={metrics['global_step']}-"
            f"{self.monitor_key}={score_token}.ckpt"
        )

    def _manifest_record(self, record):
        path = record["path"]
        if path.parent.resolve() != self.save_dir.resolve():
            raise ValueError(f"Top-k checkpoint must be inside {self.save_dir}: {path}")
        return {
            "score": float(record["score"]),
            "path": path.name,
            "epoch": int(record["epoch"]),
            "global_step": int(record["global_step"]),
        }

    def _write_manifest(self):
        records = [self._manifest_record(item) for item in self.checkpoints]
        manifest = {
            "format_version": BEST_MANIFEST_FORMAT,
            "monitor": self.monitor_key,
            "mode": self.mode,
            "best": records[0],
            "checkpoints": records,
        }
        temp_path = self.manifest_path.with_suffix(".json.tmp")
        try:
            with temp_path.open("w", encoding="utf-8") as handle:
                json.dump(manifest, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, self.manifest_path)
        except Exception:
            if temp_path.exists():
                temp_path.unlink()
            raise

    def record_checkpoint(self, path, metrics):
        """Commit selection state only after ``path`` was saved successfully."""

        path = pathlib.Path(path)
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Cannot record missing or empty checkpoint: {path}")
        record = {
            "score": float(metrics[self.monitor_key]),
            "path": path,
            "epoch": int(metrics["epoch"]),
            "global_step": int(metrics["global_step"]),
        }
        if not math.isfinite(record["score"]):
            raise ValueError(f"Top-k score must be finite, got {record['score']}.")
        candidates = [item for item in self.checkpoints if item["path"] != path]
        candidates.append(record)
        selected = self._sort(candidates)[: self.k]
        evicted = [item for item in self.checkpoints if item not in selected]

        self.checkpoints = selected
        # Publish the new exact state before cleanup so a crash cannot leave the
        # manifest pointing at an already-deleted checkpoint.
        self._write_manifest()
        for item in evicted:
            old_path = item["path"]
            if old_path != path and old_path.exists():
                old_path.unlink()


class TrainCovDiffusionWorkspace:
    include_keys = ("global_step", "epoch", "latest_rollout_monitor_metric")

    def __init__(self, cfg, output_dir):
        self.cfg = cfg
        self._output_dir = output_dir

        checkpoint_defaults = OmegaConf.create(
            {
                "save_ckpt": True,
                "checkpoint_every": ROLLOUT_INTERVAL,
                "save_last_ckpt": True,
                "topk": {"k": 1, "monitor_key": TOPK_MONITOR, "mode": "min"},
            }
        )
        cfg.checkpoint = OmegaConf.merge(
            checkpoint_defaults,
            cfg.checkpoint if hasattr(cfg, "checkpoint") else {},
        )
        training_defaults = OmegaConf.create(
            {
                "max_train_steps": None,
                "replay_epoch_sampling": True,
                "eval_episodes": 10,
            }
        )
        cfg.training = OmegaConf.merge(
            training_defaults,
            cfg.training if hasattr(cfg, "training") else {},
        )

        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = get_model_diffusion(
            config=cfg,
            which=cfg.model.backbone,
            io_type=cfg.task_name,
            device=device,
        )
        self.ema_model = copy.deepcopy(self.model)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)

        self.global_step = 0
        self.epoch = 0
        self.latest_rollout_monitor_metric = None
        self._resume_rng_state = None

        self.env_runner = None
        if not cfg.skip_rollout_eval:
            from covdiffusion.env_runner.covdiffusion_runner import CovDiffusionRunner

            self.env_runner = CovDiffusionRunner(
                output_dir=self.output_dir,
                eval_episodes=int(cfg.training.eval_episodes),
                chamfer_distance_threshold=0.05,
                batch_size=1,
                num_workers=cfg.workers,
                seed=cfg.seed,
                training_seed=cfg.seed,
                checkpoint_selection=True,
                config=cfg,
                condition_modes=list(ROLLOUT_CONDITION_MODES),
            )

    @property
    def output_dir(self):
        return self._output_dir

    def get_checkpoint_path(self, tag="latest"):
        return pathlib.Path(self.output_dir) / "checkpoints" / f"{tag}.ckpt"

    def save_checkpoint(self, path=None, tag="latest"):
        path = pathlib.Path(path) if path is not None else self.get_checkpoint_path(tag)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "cfg": self.cfg,
            "state_dicts": {
                "model": self.model.state_dict(),
                "ema_model": self.ema_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            "pickles": {
                key: dill.dumps(getattr(self, key)) for key in self.include_keys
            },
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": (
                    torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
                ),
            },
            "torch_backend_state": capture_torch_backend_state(),
        }

        temp_path = path.with_suffix(path.suffix + ".tmp")
        try:
            with temp_path.open("wb") as handle:
                torch.save(payload, handle, pickle_module=dill)
            os.replace(temp_path, path)
        except Exception as error:
            if temp_path.exists():
                temp_path.unlink()
            raise RuntimeError(f"Could not save checkpoint to {path}: {error}") from error
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Checkpoint write did not produce a valid file: {path}")
        print(f"Saved checkpoint to: {path}")
        return str(path.absolute())

    def load_checkpoint(self, path=None, tag="latest"):
        path = pathlib.Path(path) if path is not None else self.get_checkpoint_path(tag)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint file not found: {path}")
        try:
            with path.open("rb") as handle:
                payload = torch.load(handle, pickle_module=dill, map_location="cpu")
        except Exception as error:
            raise RuntimeError(f"Could not load checkpoint from {path}: {error}") from error

        validate_resume_config(payload.get("cfg"), self.cfg)
        state_dicts = payload["state_dicts"]
        self.model.load_state_dict(state_dicts["model"])
        self.ema_model.load_state_dict(state_dicts["ema_model"])
        self.optimizer.load_state_dict(state_dicts["optimizer"])
        for key in self.include_keys:
            if key in payload.get("pickles", {}):
                setattr(self, key, dill.loads(payload["pickles"][key]))

        backend_state = payload.get("torch_backend_state")
        if backend_state is None:
            print(
                "Warning: checkpoint has no Torch backend state; exact resume "
                "across backend flag changes is unavailable."
            )
        else:
            restore_torch_backend_state(backend_state)
            print("Restored Torch backend state.")

        self._resume_rng_state = payload.get("rng_state")
        if self._resume_rng_state is None:
            print("Warning: checkpoint has no RNG state; exact resume is unavailable.")
        print(
            f"Loaded checkpoint from: {path}. Resuming at epoch {self.epoch}, "
            f"global_step {self.global_step}"
        )
        return payload

    def _build_datasets(self, cfg):
        dataset_name = cfg.dataset
        processed_root = getattr(cfg, "processed_data_root", None)
        if not processed_root:
            raise ValueError(
                "processed_data_root is required. Use `python reproduce.py train "
                "<category>` after `python reproduce.py prepare <category> "
                "--data-only`."
            )

        zarr_train = os.path.join(
            os.path.expanduser(str(processed_root)),
            "data",
            dataset_name,
            "train.zarr",
        )
        train_dataset = ProcessedCovDiffusionDataset(
            zarr_path=zarr_train,
            config=cfg,
            horizon=cfg.horizon,
            pad_before=0,
            pad_after=0,
            seed=cfg.seed,
        )
        print(f"Using processed training data: {zarr_train}")

        evaluation_cache_root = None
        if not cfg.skip_rollout_eval:
            cache_root = os.path.abspath(
                os.path.expanduser(str(cfg.evaluation_cache_root))
            )
            if not os.path.isdir(cache_root):
                raise FileNotFoundError(
                    "Evaluation-ready cache not found: "
                    f"{cache_root}. Run `python reproduce.py prepare <category> "
                    "--data-only` first."
                )
            evaluation_cache_root = cache_root
            os.environ.setdefault("COVDIFFUSION_EVAL_CACHE_ROOT", cache_root)
        return train_dataset, evaluation_cache_root

    def _build_loaders(self, cfg):
        train_dataset, evaluation_cache_root = self._build_datasets(cfg)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=min(cfg.batch_size, len(train_dataset)),
            shuffle=True,
            num_workers=cfg.workers,
            drop_last=False,
        )

        rollout_loader = None
        if not cfg.skip_rollout_eval:
            from utils.dataset.covdiffusion_rollout_dataset import (
                CovDiffusionRolloutDataset,
            )

            rollout_dataset = CovDiffusionRolloutDataset(
                config=cfg,
                split="test",
                seed=cfg.seed,
                evaluation_cache_root=evaluation_cache_root,
            )
            print(f"Rollout dataset episodes: {len(rollout_dataset)}")
            rollout_loader = torch.utils.data.DataLoader(
                rollout_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=cfg.workers,
            )
        else:
            print("Evaluation-ready rollout/top-k evaluation is disabled.")
        return train_dataset, train_loader, rollout_loader

    def _restore_rng_state(self):
        if self._resume_rng_state is None:
            return
        state = self._resume_rng_state
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch_cpu"])
        if torch.cuda.is_available() and state.get("torch_cuda") is not None:
            torch.cuda.set_rng_state_all(state["torch_cuda"])
        print("Restored Python, NumPy, Torch CPU, and CUDA RNG state.")

    def _run_rollout(self, cfg, rollout_loader, step_log):
        if (
            self.env_runner is None
            or rollout_loader is None
            or self.epoch % ROLLOUT_INTERVAL != 0
        ):
            return

        print("\n=== Rollout Eval via env_runner ===")
        self.ema_model.eval()
        runner_log = self.env_runner.run(
            self.ema_model,
            dataloader=rollout_loader,
            split="test",
            run_name=os.path.basename(self.output_dir),
            dataset_name=cfg.dataset,
        )
        self.ema_model.train()

        if not isinstance(runner_log, dict):
            return
        for mode_key, mode_data in runner_log.items():
            if not isinstance(mode_data, dict):
                continue
            for metric_name, metric_value in mode_data.items():
                if torch.is_tensor(metric_value) and metric_value.numel() == 1:
                    metric_value = metric_value.item()
                elif isinstance(metric_value, np.ndarray) and metric_value.size == 1:
                    metric_value = metric_value.item()
                if not isinstance(metric_value, (float, int, np.number)):
                    continue
                metric_key = f"{mode_key}_{metric_name}"
                step_log[metric_key] = metric_value
                if metric_key == cfg.checkpoint.topk.monitor_key:
                    self.latest_rollout_monitor_metric = metric_value

    def _save_epoch_checkpoints(self, cfg, step_log, topk_manager):
        if not cfg.checkpoint.save_ckpt:
            return
        if self.epoch % int(cfg.checkpoint.checkpoint_every) != 0:
            return

        if cfg.checkpoint.save_last_ckpt:
            self.save_checkpoint(tag="latest")
        if topk_manager is None or self.latest_rollout_monitor_metric is None:
            return

        metrics = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            cfg.checkpoint.topk.monitor_key: self.latest_rollout_monitor_metric,
        }
        metrics.update(
            {
                key: value.item() if torch.is_tensor(value) else value
                for key, value in step_log.items()
                if isinstance(value, (float, int, np.number))
                or (torch.is_tensor(value) and value.numel() == 1)
            }
        )
        topk_path = topk_manager.get_ckpt_path(metrics)
        if topk_path is not None:
            self.save_checkpoint(path=topk_path)
            topk_manager.record_checkpoint(topk_path, metrics)

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        resume_training = bool(getattr(cfg, "resume", False))
        if resume_training:
            self.load_checkpoint()
        first_epoch = self.epoch + 1 if resume_training else self.epoch

        train_dataset, train_loader, rollout_loader = self._build_loaders(cfg)
        normalizer = train_dataset.get_normalizer()
        self.model.set_normalizer(normalizer)
        self.ema_model.set_normalizer(normalizer)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        ema = EMAModel(model=self.ema_model)
        if resume_training:
            ema.optimization_step = self.global_step

        topk_manager = None
        if not cfg.skip_rollout_eval:
            topk_manager = TopKCheckpointManager(
                pathlib.Path(self.output_dir) / "checkpoints",
                k=cfg.checkpoint.topk.k,
                monitor_key=cfg.checkpoint.topk.monitor_key,
                mode=cfg.checkpoint.topk.mode,
            )

        # Dataset/runner construction may consume RNG. Resume restoration must be
        # the final initialization step before the next DataLoader iterator.
        self._restore_rng_state()

        replay_epoch_sampling = bool(cfg.training.replay_epoch_sampling)
        replay_batch = None
        for epoch_idx in range(first_epoch, cfg.epochs):
            self.epoch = epoch_idx
            self.model.train()
            train_losses = []

            with tqdm(train_loader, desc=f"Epoch {self.epoch}", leave=False) as progress:
                for batch_data in progress:
                    batch = dict_apply(
                        batch_data,
                        lambda value: value.to(device)
                        if isinstance(value, torch.Tensor)
                        else value,
                    )
                    if replay_epoch_sampling and replay_batch is None:
                        replay_batch = {
                            "obs": dict_apply(
                                batch["obs"], lambda value: value.clone()
                            ),
                            "prev_true_trajectory": batch[
                                "prev_true_trajectory"
                            ].clone(),
                        }

                    raw_loss, _ = self.model.compute_loss(batch, save_dir=None)
                    if not torch.isfinite(raw_loss):
                        raise FloatingPointError(
                            f"Non-finite loss at step {self.global_step}: "
                            f"{raw_loss.item()}"
                        )
                    raw_loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    ema.step(self.model)

                    train_losses.append(raw_loss.item())
                    self.global_step += 1
                    max_steps = cfg.training.max_train_steps
                    if max_steps and self.global_step >= max_steps:
                        if not self.optimizer.state:
                            raise RuntimeError("Adam state was not created by the smoke step.")
                        print(
                            "TRAIN_STEP_OK "
                            f"dataset={cfg.dataset} global_step={self.global_step} "
                            f"loss={raw_loss.item():.8f} device={device}"
                        )
                        break

            step_log = {
                "train_loss_epoch": (
                    float(np.mean(train_losses)) if train_losses else float("nan")
                ),
                "epoch": self.epoch,
                "global_step": self.global_step,
            }

            # The archived loop sampled one fixed training batch after every
            # epoch. The output is intentionally discarded, but this prediction
            # must remain because it advances the diffusion RNG stream.
            if replay_epoch_sampling and replay_batch is not None:
                with torch.no_grad():
                    self.ema_model.eval()
                    self.ema_model.predict_action(replay_batch)

            self._run_rollout(cfg, rollout_loader, step_log)
            self._save_epoch_checkpoints(cfg, step_log, topk_manager)

            max_steps = cfg.training.max_train_steps
            if max_steps and self.global_step >= max_steps:
                print("Max training steps reached.")
                break

        if cfg.checkpoint.save_ckpt and cfg.checkpoint.save_last_ckpt:
            self.save_checkpoint(tag="latest")
        print("\nTraining complete")


def main():
    config = load_training_config()
    resume_training = bool(getattr(config, "resume", False))
    if resume_training:
        if not getattr(config, "output_dir", None):
            raise ValueError("resume=true requires output_dir=<existing-run-dir>")
        save_dir = os.path.abspath(os.path.expanduser(str(config.output_dir)))
        run_name = os.path.basename(os.path.normpath(save_dir))
        latest = pathlib.Path(save_dir) / "checkpoints" / "latest.ckpt"
        if not latest.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {latest}")
    else:
        requested_output_dir = getattr(config, "output_dir", None)
        if requested_output_dir:
            save_dir = os.path.abspath(
                os.path.expanduser(str(requested_output_dir))
            )
            run_name = os.path.basename(os.path.normpath(save_dir))
        else:
            suffix = "".join(
                random.choice(string.ascii_uppercase + string.digits)
                for _ in range(5)
            )
            run_name = f"{suffix}-S{config.seed}"
            save_dir = os.path.join(get_output_dir(config), run_name)

    pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
    workspace = TrainCovDiffusionWorkspace(config, output_dir=save_dir)
    save_config(
        workspace.cfg,
        save_dir,
        filename="config.resume.yaml" if resume_training else "config.yaml",
    )
    print(f"\n===== RUN: {run_name} @ {save_dir} =====\n")
    print(pformat_dict(workspace.cfg))
    workspace.run()


if __name__ == "__main__":
    main()
