#!/usr/bin/env python3
"""Reproduce 3D-CovDiffusion through one of two explicit paths.

Retraining:

    python reproduce.py prepare windows --data-only
    python reproduce.py train windows
    python reproduce.py evaluate windows --run-dir artifacts/runs/windows/seed42

Released checkpoints:

    python reproduce.py prepare windows --data-only
    python reproduce.py prepare windows --checkpoint-only
    python reproduce.py evaluate windows

Every subprocess is invoked without a shell, so OmegaConf arguments behave the
same in zsh and bash.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Iterable, Optional, Sequence


ROOT = Path(__file__).resolve().parent
DATASET_REPO = "ChenyuanC/3D-CovDiffusion-Train-Ready"
DATASET_REVISION = "38bf84c4c8a34bc497167b21b2b68182e0099dc8"
EVALUATION_CACHE_DIRECTORY = "evaluation-cache"
EVALUATION_CACHE_MANIFEST = "evaluation_cache_manifest.json"
MODEL_REPO = "ChenyuanC/3D-CovDiffusion"
SELECTED_INFERENCE_PROFILE = (
    ROOT / "configs" / "inference" / "seed42_selected_episodes.json"
)
TRAINING_PROFILE = ROOT / "configs" / "training" / "seed42_v2.json"
CATEGORIES = {
    "windows": ("windows-v2", "windows"),
    "cuboids": ("cuboids-v2", "cuboids"),
    "shelves": ("shelves-v2", "shelves"),
    "containers": ("containers-v2", "containers"),
}
CATEGORY_CHOICES = (*CATEGORIES, "all")

CORE_MODULES = (
    "torch",
    "diffusers",
    "einops",
    "omegaconf",
    "dill",
    "safetensors",
    "huggingface_hub",
    "numpy",
    "zarr",
    "numcodecs",
    "tqdm",
    "models",
    "evaluate",
)
VIS_MODULES = ("open3d",)
DEFAULT_TRAIN_CUDA_ALLOC_CONF = "max_split_size_mb:128"


def run_checked(command: Sequence[str], *, env=None) -> None:
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def isolated_subprocess_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONNOUSERSITE"] = "1"
    return environment


def training_subprocess_environment() -> dict[str, str]:
    """Return the isolated environment with the validated CUDA allocator split."""

    environment = isolated_subprocess_environment()
    environment.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF", DEFAULT_TRAIN_CUDA_ALLOC_CONF
    )
    return environment


def selected_categories(value: str) -> list[str]:
    return list(CATEGORIES) if value == "all" else [value]


def category_values(category: str) -> tuple[str, str]:
    return CATEGORIES[category]


def evaluation_cache_root(artifact_root: Path) -> Path:
    """Return the self-contained fixed-test evaluation release directory."""

    return (
        artifact_root / "dataset" / EVALUATION_CACHE_DIRECTORY
    ).expanduser().resolve()


def artifact_paths(
    artifact_root: Path, category: str, weight_variant: str = "ema"
) -> tuple[Path, Path, Path, Path]:
    dataset_root = artifact_root / "dataset"
    model_root = artifact_root / "models"
    category_root = (
        model_root / category
        if weight_variant == "ema"
        else model_root / "raw" / category
    )
    return (
        dataset_root,
        model_root,
        category_root / "model.safetensors",
        category_root / "config.yaml",
    )


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path, description: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{description} not found: {resolved}")
    return resolved


def training_profile() -> dict:
    return json.loads(TRAINING_PROFILE.read_text(encoding="utf-8"))


def historical_batch_size(category: str) -> int:
    return int(training_profile()["categories"][category]["batch_size"])


def load_selected_inference_profile(
    path: Path = SELECTED_INFERENCE_PROFILE,
) -> dict:
    profile = json.loads(path.read_text(encoding="utf-8"))
    if profile.get("format_version") != "3dcov-selected-inference-v1":
        raise ValueError(f"Unsupported inference profile: {path}")
    if set(profile.get("categories") or {}) != set(CATEGORIES):
        raise ValueError("Inference profile must define all four categories")
    cache_source = profile.get("evaluation_data_source") or {}
    if (
        cache_source.get("canonical_cache_repository") != DATASET_REPO
        or cache_source.get("canonical_cache_revision") != DATASET_REVISION
    ):
        raise ValueError("Inference profile does not pin the canonical cache release")
    return profile


def current_git_state() -> Optional[dict]:
    """Return the exact source revision used for a run."""

    if not (ROOT / ".git").exists():
        return None

    def git(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=ROOT,
            check=True,
            text=True,
            capture_output=True,
        )
        return completed.stdout.strip()

    return {
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain", "--untracked-files=all")),
        "tags": sorted(git("tag", "--points-at", "HEAD").splitlines()),
    }


def evaluator_fingerprint(profile: dict) -> dict:
    """Validate the numerical evaluator by content, independent of Git tags."""

    expected = profile.get("evaluation_code", {}).get("critical_files")
    if not isinstance(expected, dict) or not expected:
        raise ValueError("Inference profile has no critical evaluator file hashes")
    actual = {}
    for relative, expected_sha in sorted(expected.items()):
        path = require_file(ROOT / relative, f"evaluator file {relative}")
        actual_sha = sha256_file(path)
        if actual_sha != expected_sha:
            raise RuntimeError(
                f"Evaluator fingerprint mismatch for {relative}: {actual_sha}; "
                f"expected {expected_sha}"
            )
        actual[relative] = actual_sha
    aggregate = hashlib.sha256(
        json.dumps(actual, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    expected_aggregate = profile["evaluation_code"].get("sha256")
    if not isinstance(expected_aggregate, str) or len(expected_aggregate) != 64:
        raise ValueError("Inference profile has no evaluator aggregate SHA-256")
    if aggregate != expected_aggregate:
        raise RuntimeError(
            f"Evaluator aggregate fingerprint is {aggregate}; expected "
            f"{expected_aggregate}"
        )
    return {"sha256": aggregate, "critical_files": actual}


def validate_evaluation_cache(artifact_root: Path, category: str) -> Path:
    """Validate manifest identity, file hashes, and every NPZ payload."""

    artifact_root = artifact_root.expanduser().resolve()
    dataset_name, _ = category_values(category)
    dataset_root = artifact_root / "dataset"
    manifest_path = require_file(
        dataset_root / EVALUATION_CACHE_MANIFEST,
        "evaluation cache manifest",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != "3dcov-evaluation-ready-v2":
        raise ValueError(f"unsupported evaluation cache manifest: {manifest_path}")
    if manifest.get("schema_version") != "3dcov-evaluation-v2":
        raise ValueError(f"unsupported evaluation sample schema: {manifest_path}")
    case = load_selected_inference_profile()["categories"][category]
    entry = (manifest.get("categories") or {}).get(dataset_name)
    if not isinstance(entry, dict):
        raise ValueError(f"evaluation cache has no {dataset_name} entry")
    if entry.get("split") != "test":
        raise ValueError(f"evaluation cache split is not test for {category}")
    if entry.get("test_split_sha256") != case["test_split_sha256"]:
        raise ValueError(f"evaluation cache split mismatch for {category}")
    sample_ids = entry.get("sample_ids")
    files = entry.get("files")
    if not isinstance(sample_ids, list) or not isinstance(files, dict):
        raise ValueError(f"evaluation cache files are invalid for {category}")
    if len(sample_ids) != case["test_split_size"]:
        raise ValueError(f"evaluation cache split size mismatch for {category}")
    if len(set(sample_ids)) != len(sample_ids) or set(files) != set(sample_ids):
        raise ValueError(
            f"evaluation cache does not exactly cover {dataset_name} fixed test split"
        )
    selected_index = int(case["test_split_index"])
    if sample_ids[selected_index] != case["sample_id"]:
        raise ValueError(f"selected evaluation sample mismatch for {category}")

    from utils.dataset.canonical_evaluation_dataset import (
        CanonicalEvaluationDataset,
    )

    cache_root = evaluation_cache_root(artifact_root)
    dataset = CanonicalEvaluationDataset(
        cache_root,
        dataset_name,
        split="test",
        verify_hashes=True,
    )
    if list(dataset.sample_ids) != sample_ids:
        raise ValueError(f"evaluation cache order mismatch for {category}")
    for index in range(len(dataset)):
        dataset[index]

    selected_sha = files.get(case["sample_id"])
    if selected_sha != case["evaluation_ready_sha256"]:
        raise ValueError(f"selected evaluation cache mismatch for {category}")
    return cache_root / dataset_name


def train_command(
    *,
    category: str,
    seed: int = 42,
    workers: int = 0,
    data_root: Path,
    evaluation_data_root: Optional[Path] = None,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    max_train_steps: Optional[int] = None,
    save_checkpoint: bool = True,
    replay_epoch_sampling: bool = True,
    rollout_episodes: Optional[int] = None,
    output_dir: Optional[Path] = None,
    resume_from: Optional[Path] = None,
) -> list[str]:
    _, category_config = category_values(category)
    resolved_batch = historical_batch_size(category) if batch_size is None else batch_size
    command = [
        sys.executable,
        "train.py",
        f"config=[covdiffusion,{category_config}]",
        f"seed={seed}",
        f"batch_size={resolved_batch}",
        f"workers={workers}",
        "training.replay_epoch_sampling="
        f"{'true' if replay_epoch_sampling else 'false'}",
        f"checkpoint.save_ckpt={'true' if save_checkpoint else 'false'}",
    ]
    command.append(f"processed_data_root={data_root.expanduser().resolve()}")
    if evaluation_data_root is not None:
        command.append(
            "evaluation_cache_root="
            f"{evaluation_data_root.expanduser().resolve()}"
        )
    if epochs is not None:
        command.append(f"epochs={epochs}")
    if max_train_steps is not None:
        command.append(f"training.max_train_steps={max_train_steps}")
    if rollout_episodes is not None:
        command.append(f"training.eval_episodes={rollout_episodes}")
    if output_dir is not None and resume_from is not None:
        if output_dir.expanduser().resolve() != resume_from.expanduser().resolve():
            raise ValueError("output_dir and resume_from must identify the same run")
    run_dir = resume_from if resume_from is not None else output_dir
    if resume_from is not None:
        command.append("resume=true")
    if run_dir is not None:
        command.append(f"output_dir={run_dir.expanduser().resolve()}")
    return command


def evaluate_command(
    *,
    category: str,
    checkpoint: Path,
    config: Optional[Path],
    episodes: int,
    workers: int,
    rollout_seed: int,
    training_seed: Optional[int],
    output_dir: Path,
    coverage_mode: str,
    spray_radius: float,
    save_artifacts: bool,
    run_name: Optional[str] = None,
    weight_variant: str = "auto",
    single_episode_idx: Optional[int] = None,
    include_gt_conditioned: bool = False,
) -> list[str]:
    dataset_name, _ = category_values(category)
    command = [
        sys.executable,
        "evaluate.py",
        "--checkpoint_path",
        str(checkpoint.expanduser().resolve()),
        "--weight_variant",
        weight_variant,
    ]
    if config is not None:
        command.extend(["--config_path", str(config.expanduser().resolve())])
    command.extend(
        [
            "--dataset_name",
            dataset_name,
            "--dataset_split",
            "test",
            "--eval_episodes",
            str(episodes),
            "--workers",
            str(workers),
            "--seed",
            str(rollout_seed),
            "--output_dir_base",
            str(output_dir.expanduser().resolve()),
            "--coverage_mode",
            coverage_mode,
            "--spray_radius",
            str(spray_radius),
        ]
    )
    if training_seed is not None:
        command.extend(["--training_seed", str(training_seed)])
    resolved_name = run_name or (
        f"{category}-s{training_seed}" if training_seed is not None else None
    )
    if resolved_name:
        command.extend(["--run_name", resolved_name])
    if save_artifacts:
        command.append("--save_artifacts")
    if single_episode_idx is not None:
        command.extend(["--single_episode_idx", str(single_episode_idx)])
    if include_gt_conditioned:
        command.append("--include_gt_conditioned")
    return command


def released_training_seed(checkpoint: Path, requested_seed: Optional[int]) -> int:
    metadata_path = checkpoint.with_name("metrics.json")
    metadata = json.loads(
        require_file(metadata_path, "checkpoint metadata").read_text(encoding="utf-8")
    )
    source_seed = metadata.get("seed")
    if not isinstance(source_seed, int):
        raise ValueError(f"checkpoint metadata has no integer seed: {metadata_path}")
    if requested_seed is not None and requested_seed != source_seed:
        raise ValueError(
            f"requested training seed {requested_seed} does not match checkpoint "
            f"seed {source_seed}"
        )
    return source_seed


def cmd_doctor(args) -> None:
    modules = list(CORE_MODULES)
    if args.visualization:
        modules.extend(VIS_MODULES)
    failures = {}
    checks = list(dict.fromkeys(modules))
    for name in checks:
        try:
            importlib.import_module(name)
        except Exception as error:
            failures[name] = f"{type(error).__name__}: {error}"
    if "torch" not in failures:
        torch = importlib.import_module("torch")
        try:
            if not torch.cuda.is_available():
                failures["CUDA"] = "torch.cuda.is_available() is False"
            else:
                torch.cuda.get_device_name(0)
        except Exception as error:
            failures["CUDA"] = f"{type(error).__name__}: {error}"
    checks.append("CUDA")
    print(f"Python: {sys.version.split()[0]}")
    for name in checks:
        detail = "" if name not in failures else f" — {failures[name]}"
        print(f"  {'OK' if name not in failures else 'BROKEN':7} {name}{detail}")
    if failures:
        visualization_flag = " --visualization" if args.visualization else ""
        raise SystemExit(
            f"{len(failures)} dependency check(s) failed. Create the locked "
            "environment with `bash scripts/create_locked_environment.sh "
            f"3dcov-cu117{visualization_flag}`."
        )
    print("Environment check passed.")


def _snapshot_download():
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise SystemExit(
            "Install the project dependencies with `python -m pip install "
            "-r requirements.txt` before prepare."
        ) from error

    return snapshot_download


def _download_hf_data(artifact_root: Path, categories: list[str]) -> None:
    snapshot_download = _snapshot_download()

    dataset_root = artifact_root / "dataset"
    dataset_patterns = [
        "dataset_manifest.json",
        EVALUATION_CACHE_MANIFEST,
        "README.md",
    ]
    for category in categories:
        dataset_name, _ = category_values(category)
        dataset_patterns.append(f"data/{dataset_name}/**")
        dataset_patterns.append(
            f"{EVALUATION_CACHE_DIRECTORY}/{dataset_name}/**"
        )

    print(f"Downloading immutable training data to {dataset_root}")
    snapshot_download(
        repo_id=DATASET_REPO,
        repo_type="dataset",
        revision=DATASET_REVISION,
        allow_patterns=dataset_patterns,
        local_dir=dataset_root,
    )


def _download_hf_checkpoints(
    artifact_root: Path, categories: list[str], *, include_raw: bool
) -> None:
    snapshot_download = _snapshot_download()
    profile = load_selected_inference_profile()
    model_root = artifact_root / "models"
    model_patterns = ["manifest.json", "SHA256SUMS"]
    if include_raw:
        model_patterns.extend(["raw/manifest.json", "raw/SHA256SUMS"])
    for category in categories:
        model_patterns.append(f"{category}/**")
        if include_raw:
            model_patterns.append(f"raw/{category}/**")

    print(f"Downloading immutable released checkpoints to {model_root}")
    snapshot_download(
        repo_id=MODEL_REPO,
        revision=profile["model_revision"],
        allow_patterns=model_patterns,
        local_dir=model_root,
    )


def _validate_prepared_data(artifact_root: Path, categories: list[str]) -> None:
    environment = isolated_subprocess_environment()
    dataset_root = artifact_root / "dataset"
    dataset_names = [category_values(category)[0] for category in categories]
    run_checked(
        [
            sys.executable,
            "scripts/validate_train_ready_dataset.py",
            str(dataset_root),
            "--categories",
            *dataset_names,
        ],
        env=environment,
    )
    for category in categories:
        validate_evaluation_cache(artifact_root, category)


def _validate_prepared_checkpoints(
    artifact_root: Path,
    categories: list[str],
    device: str,
    *,
    variants: tuple[str, ...],
) -> None:
    environment = isolated_subprocess_environment()
    for category in categories:
        for variant in variants:
            _, _, checkpoint, config = artifact_paths(artifact_root, category, variant)
            run_checked(
                [
                    sys.executable,
                    "scripts/validate_hf_checkpoint.py",
                    "--checkpoint",
                    str(require_file(checkpoint, f"{variant} checkpoint")),
                    "--config",
                    str(require_file(config, f"{variant} checkpoint config")),
                    "--device",
                    device,
                ],
                env=environment,
            )


def cmd_prepare(args) -> None:
    categories = selected_categories(args.category)
    artifact_root = args.artifact_root.expanduser().resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    if not args.checkpoint_only:
        _download_hf_data(artifact_root, categories)
        _validate_prepared_data(artifact_root, categories)
    if not args.data_only:
        include_raw = not args.checkpoint_only
        _download_hf_checkpoints(
            artifact_root, categories, include_raw=include_raw
        )
        _validate_prepared_checkpoints(
            artifact_root,
            categories,
            args.device,
            variants=("ema", "raw") if include_raw else ("ema",),
        )

    if args.data_only:
        prepared = "Data prepared and validated"
    elif args.checkpoint_only:
        prepared = "Released EMA checkpoints downloaded and validated"
    else:
        prepared = "Data and released checkpoints prepared and validated"
    print(f"{prepared}: {', '.join(categories)}")


def require_empty_run_directory(path: Path) -> Path:
    """Refuse to overwrite a prior canonical training run."""

    resolved = path.expanduser().resolve()
    if resolved.exists():
        if not resolved.is_dir() or any(resolved.iterdir()):
            raise FileExistsError(
                f"Training run directory is not empty: {resolved}. "
                "Resume it with --resume-from, or move it before retraining."
            )
    return resolved


def cmd_train(args) -> None:
    categories = selected_categories(args.category)
    if args.resume_from and len(categories) != 1:
        raise ValueError("--resume-from requires one category")
    if args.resume_from and args.smoke:
        raise ValueError("--resume-from cannot be combined with --smoke")
    artifact_root = args.artifact_root.expanduser().resolve()
    processed_root = artifact_root / "dataset"
    fixed_eval_root = evaluation_cache_root(artifact_root)

    resume_from = args.resume_from.expanduser().resolve() if args.resume_from else None
    run_dirs = {}
    if not args.smoke:
        for category in categories:
            run_dir = resume_from or artifact_root / "runs" / category / "seed42"
            run_dirs[category] = (
                run_dir if resume_from else require_empty_run_directory(run_dir)
            )

    for category in categories:
        environment = training_subprocess_environment()
        environment.pop("WORKDIR", None)
        if args.smoke:
            environment["WORKDIR"] = tempfile.mkdtemp(
                prefix=f"3dcov-{category}-smoke-"
            )
        dataset_name, _ = category_values(category)
        if not (processed_root / "data" / dataset_name / "train.zarr").is_dir():
            raise FileNotFoundError(
                f"processed training data not found for {category}: {processed_root}. "
                f"Run `python reproduce.py prepare {category} --data-only` first."
            )
        validate_evaluation_cache(artifact_root, category)
        environment["COVDIFFUSION_EVAL_CACHE_ROOT"] = str(fixed_eval_root)

        if resume_from and not (resume_from / "checkpoints" / "latest.ckpt").is_file():
            raise FileNotFoundError(
                f"resume checkpoint not found: {resume_from / 'checkpoints/latest.ckpt'}"
            )
        command = train_command(
            category=category,
            data_root=processed_root,
            evaluation_data_root=fixed_eval_root,
            workers=args.workers,
            epochs=1 if args.smoke else training_profile()["common"]["epochs"],
            batch_size=2 if args.smoke else None,
            max_train_steps=1 if args.smoke else None,
            save_checkpoint=not args.smoke,
            replay_epoch_sampling=not args.smoke,
            rollout_episodes=1 if args.smoke else None,
            output_dir=run_dirs.get(category),
            resume_from=resume_from,
        )
        run_checked(command, env=environment)


def _seed_from_saved_config(config: Path) -> int:
    """Read the top-level integer seed without importing the training runtime."""

    for line in config.read_text(encoding="utf-8").splitlines():
        if not line.startswith("seed:"):
            continue
        value = line.split(":", 1)[1].split("#", 1)[0].strip().strip("'\"")
        try:
            return int(value)
        except ValueError as error:
            raise ValueError(f"invalid seed in saved config: {config}") from error
    return 42


def resolve_training_run(run_dir: Path) -> tuple[Path, Path, int]:
    """Resolve the exact best legacy checkpoint recorded by a training run."""

    run_dir = run_dir.expanduser().resolve()
    checkpoint_dir = run_dir / "checkpoints"
    manifest_path = require_file(
        checkpoint_dir / "best.json", "best checkpoint manifest"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != "3dcov-best-checkpoint-v1":
        raise ValueError(f"unsupported best checkpoint manifest: {manifest_path}")
    best = manifest.get("best")
    relative_path = best.get("path") if isinstance(best, dict) else None
    if not isinstance(relative_path, str) or not relative_path:
        raise ValueError(f"best checkpoint manifest has no best.path: {manifest_path}")
    relative_path = Path(relative_path)
    if relative_path.is_absolute():
        raise ValueError(f"best.path must be relative: {manifest_path}")

    checkpoint_dir = checkpoint_dir.resolve()
    checkpoint = (checkpoint_dir / relative_path).resolve()
    try:
        checkpoint.relative_to(checkpoint_dir)
    except ValueError as error:
        raise ValueError(
            f"best.path escapes the checkpoints directory: {manifest_path}"
        ) from error
    checkpoint = require_file(checkpoint, "best training checkpoint")
    if checkpoint.suffix != ".ckpt":
        raise ValueError(
            f"training run best checkpoint must be a .ckpt file: {checkpoint}"
        )

    config = require_file(run_dir / "config.yaml", "saved training config")
    return checkpoint, config, _seed_from_saved_config(config)


def _resolve_evaluation_checkpoint(
    args, category: str
) -> tuple[Path, Optional[Path], int]:
    if args.run_dir:
        if args.category == "all":
            raise ValueError("--run-dir requires one category")
        return resolve_training_run(args.run_dir)

    artifact_root = args.artifact_root.expanduser().resolve()
    _, _, default_checkpoint, default_config = artifact_paths(
        artifact_root, category, "ema"
    )
    checkpoint = require_file(default_checkpoint, "released EMA checkpoint")
    config = require_file(default_config, "released EMA checkpoint config")
    training_seed = released_training_seed(checkpoint, 42)
    return checkpoint, config, training_seed


def cmd_evaluate(args) -> None:
    categories = selected_categories(args.category)
    if args.run_dir and len(categories) != 1:
        raise ValueError("--run-dir requires one category")
    artifact_root = args.artifact_root.expanduser().resolve()
    fixed_eval_root = evaluation_cache_root(artifact_root)
    environment = isolated_subprocess_environment()
    environment["COVDIFFUSION_EVAL_CACHE_ROOT"] = str(fixed_eval_root)
    if args.render:
        environment.setdefault("MPLBACKEND", "Agg")
    for category in categories:
        validate_evaluation_cache(artifact_root, category)
        checkpoint, config, training_seed = _resolve_evaluation_checkpoint(args, category)
        run_checked(
            evaluate_command(
                category=category,
                checkpoint=checkpoint,
                config=config,
                episodes=args.episodes,
                workers=args.workers,
                rollout_seed=42,
                training_seed=training_seed,
                output_dir=args.output_dir,
                coverage_mode="area-weighted",
                spray_radius=0.1,
                save_artifacts=args.render,
                run_name=f"{category}-s{training_seed}",
                weight_variant="ema",
            ),
            env=environment,
        )


def validate_selected_case_artifacts(
    *, category: str, checkpoint: Path, config: Path, case: dict
) -> None:
    metadata_path = require_file(checkpoint.with_name("metrics.json"), "checkpoint metadata")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_values = {
        "dataset": case["dataset"],
        "seed": case["training_seed"],
        "run_id": case["source_checkpoint"]["run_id"],
        "weight_variant": "raw",
    }
    for key, expected in expected_values.items():
        if metadata.get(key) != expected:
            raise ValueError(
                f"{metadata_path}: {key}={metadata.get(key)!r}, expected {expected!r}"
            )
    if metadata.get("source", {}).get("sha256") != case["source_checkpoint"]["sha256"]:
        raise ValueError(f"Source checkpoint SHA mismatch in {metadata_path}")
    expected_files = case["public_checkpoint"]
    actual_hashes = {
        "model_sha256": sha256_file(checkpoint),
        "config_sha256": sha256_file(config),
        "metrics_sha256": sha256_file(metadata_path),
    }
    for key, actual in actual_hashes.items():
        if actual != expected_files[key]:
            raise ValueError(f"{category} {key} mismatch: {actual}")


def _run_selected_inference(args, category: str, profile: dict) -> None:
    case = profile["categories"][category]
    artifact_root = args.artifact_root.expanduser().resolve()
    _, _, checkpoint, config = artifact_paths(artifact_root, category, "raw")
    checkpoint = require_file(checkpoint, "raw inference checkpoint")
    config = require_file(config, "raw inference config")
    validate_selected_case_artifacts(
        category=category, checkpoint=checkpoint, config=config, case=case
    )

    category_root = validate_evaluation_cache(artifact_root, category)
    sample_id = case["sample_id"]
    sample_path = require_file(
        category_root / f"{sample_id}.npz", "selected evaluation-ready input"
    )
    if sha256_file(sample_path) != case["evaluation_ready_sha256"]:
        raise ValueError(f"Input SHA mismatch for {sample_path}")

    fingerprint = evaluator_fingerprint(profile)
    code_state = current_git_state()
    if code_state is None:
        raise RuntimeError("Locked inference requires a Git checkout.")
    elif code_state["dirty"]:
        raise RuntimeError(
            "Locked inference refuses a modified or untracked worktree. "
            "Commit the exact evaluator first."
        )

    episode_index = int(case["test_split_index"])
    run_name = f"selected_{case['dataset']}_s42_ep{episode_index}"
    run_dir = args.output_dir.expanduser().resolve() / run_name
    if run_dir.exists():
        raise FileExistsError(
            f"Refusing to reuse selected-inference output: {run_dir}. "
            "Move it first so stale files cannot pass validation."
        )
    environment = isolated_subprocess_environment()
    environment["COVDIFFUSION_EVAL_CACHE_ROOT"] = str(
        evaluation_cache_root(artifact_root)
    )
    if args.render:
        environment.setdefault("MPLBACKEND", "Agg")
    run_checked(
        evaluate_command(
            category=category,
            checkpoint=checkpoint,
            config=config,
            episodes=1,
            workers=args.workers,
            rollout_seed=case["rollout_seed"],
            training_seed=case["training_seed"],
            output_dir=args.output_dir,
            coverage_mode=profile["protocol"]["coverage_mode"],
            spray_radius=profile["protocol"]["spray_radius"],
            save_artifacts=args.render,
            run_name=run_name,
            weight_variant="raw",
            single_episode_idx=episode_index,
            include_gt_conditioned=True,
        ),
        env=environment,
    )

    result_path = require_file(run_dir / "test_results.json", "inference result")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["release_provenance"] = {
        "model_repository": profile["model_repository"],
        "model_revision": profile["model_revision"],
        "evaluation_cache_repository": DATASET_REPO,
        "evaluation_cache_revision": DATASET_REVISION,
        "evaluation_code": profile["evaluation_code"],
        "evaluator_fingerprint": fingerprint,
        "actual_git_state": code_state,
        "profile_sha256": sha256_file(SELECTED_INFERENCE_PROFILE),
        "checkpoint_sha256": sha256_file(checkpoint),
        "config_sha256": sha256_file(config),
        "metrics_sha256": sha256_file(checkpoint.with_name("metrics.json")),
        "test_split_sha256": case["test_split_sha256"],
        "evaluation_ready_sha256": case["evaluation_ready_sha256"],
        "weight_variant": "raw",
        "sample_id": sample_id,
        "test_split_index": episode_index,
        "condition_mode_sequence": profile["protocol"]["condition_mode_sequence"],
        "reported_condition_mode": "Pred_Cond",
    }
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    verify = [
        sys.executable,
        "scripts/verify_selected_inference.py",
        "--category",
        category,
        "--run-dir",
        str(run_dir),
        "--profile",
        str(SELECTED_INFERENCE_PROFILE),
    ]
    if not args.render:
        verify.append("--skip-ply-hash")
    run_checked(verify, env=environment)


def cmd_infer(args) -> None:
    profile = load_selected_inference_profile()
    for category in selected_categories(args.category):
        _run_selected_inference(args, category, profile)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="check the complete runtime")
    doctor.add_argument("--visualization", action="store_true")
    doctor.set_defaults(func=cmd_doctor)

    prepare = subparsers.add_parser(
        "prepare", help="download and validate data and/or released checkpoints"
    )
    prepare.add_argument("category", nargs="?", choices=CATEGORY_CHOICES, default="all")
    prepare.add_argument("--artifact-root", type=Path, default=Path("artifacts"))
    prepare.add_argument("--device", default="cpu")
    prepare_mode = prepare.add_mutually_exclusive_group()
    prepare_mode.add_argument(
        "--data-only",
        action="store_true",
        help="download and validate only training and fixed-test data",
    )
    prepare_mode.add_argument(
        "--checkpoint-only",
        action="store_true",
        help="download and validate only the released best EMA checkpoint",
    )
    prepare.set_defaults(func=cmd_prepare)

    train = subparsers.add_parser(
        "train", help="run the complete seed-42 v2 training protocol"
    )
    train.add_argument("category", choices=CATEGORY_CHOICES)
    train.add_argument("--artifact-root", type=Path, default=Path("artifacts"))
    train.add_argument("--workers", type=int, default=0)
    train.add_argument("--resume-from", type=Path)
    train.add_argument(
        "--smoke",
        action="store_true",
        help="run one optimizer step and one raw rollout episode per category",
    )
    train.set_defaults(func=cmd_train)

    evaluate = subparsers.add_parser(
        "evaluate",
        help="evaluate released EMA, or a training run selected by best.json",
    )
    evaluate.add_argument("category", choices=CATEGORY_CHOICES)
    evaluate.add_argument("--artifact-root", type=Path, default=Path("artifacts"))
    evaluate.add_argument(
        "--run-dir",
        type=Path,
        help="evaluate the EMA weights from a training run's best.json",
    )
    evaluate.add_argument("--episodes", type=int, default=0, help="0 means all")
    evaluate.add_argument("--workers", type=int, default=4)
    evaluate.add_argument("--output-dir", type=Path, default=Path("outputs/evaluation"))
    evaluate.add_argument("--render", action="store_true")
    evaluate.set_defaults(func=cmd_evaluate)

    infer = subparsers.add_parser(
        "infer", help="replay and verify locked selected-visualization cases"
    )
    infer.add_argument("category", choices=CATEGORY_CHOICES)
    infer.add_argument("--artifact-root", type=Path, default=Path("artifacts"))
    infer.add_argument("--output-dir", type=Path, default=Path("outputs/selected"))
    infer.add_argument("--workers", type=int, default=4)
    infer.add_argument("--render", action="store_true")
    infer.set_defaults(func=cmd_infer)

    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
