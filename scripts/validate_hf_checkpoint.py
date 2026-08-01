#!/usr/bin/env python3
"""Validate a released 3D-CovDiffusion safetensors checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

import dill
import torch
from omegaconf import OmegaConf
from safetensors.torch import load_file
from safetensors import safe_open

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import get_model_diffusion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--config", type=pathlib.Path, required=True)
    parser.add_argument(
        "--source-checkpoint",
        type=pathlib.Path,
        help="Optional trusted training checkpoint for exact tensor comparison.",
    )
    parser.add_argument("--run-inference", action="store_true")
    parser.add_argument(
        "--allow-missing-manifest",
        action="store_true",
        help="Allow validation of an export before repository finalization.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def compare_with_source(weights, source_path: pathlib.Path, weight_variant: str) -> None:
    source = torch.load(
        source_path,
        map_location="cpu",
        pickle_module=dill,
        weights_only=False,
    )
    state_key = "ema_model" if weight_variant == "ema" else "model"
    source_weights = source["state_dicts"][state_key]
    if weights.keys() != source_weights.keys():
        raise RuntimeError(f"Release and {weight_variant} source keys differ")
    for key, released in weights.items():
        original = source_weights[key]
        if (
            released.shape != original.shape
            or released.dtype != original.dtype
            or not torch.equal(released, original)
        ):
            raise RuntimeError(f"Release differs from {weight_variant} source: {key}")


def sha256_file(path: pathlib.Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_release_manifest(
    checkpoint: pathlib.Path, config: pathlib.Path, allow_missing: bool = False
) -> bool:
    repository_root = checkpoint.parent.parent
    manifest_path = repository_root / "manifest.json"
    if not manifest_path.is_file():
        if allow_missing:
            return False
        raise FileNotFoundError(
            f"Release manifest is missing: {manifest_path}. "
            "Pass --allow-missing-manifest only for a pre-finalization local export."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files", {})
    candidates = [checkpoint, config, checkpoint.with_name("metrics.json")]
    for path in candidates:
        if not path.is_file():
            raise FileNotFoundError(f"Release artifact listed for validation is missing: {path}")
        relative = path.relative_to(repository_root).as_posix()
        expected = files.get(relative)
        if expected is None:
            raise RuntimeError(f"Release manifest does not list {relative}")
        if path.stat().st_size != expected.get("bytes"):
            raise RuntimeError(f"Release size mismatch: {relative}")
        if sha256_file(path) != expected.get("sha256"):
            raise RuntimeError(f"Release checksum mismatch: {relative}")
    return True


def make_smoke_input(model, device: torch.device):
    pc_stats = model.normalizer["point_cloud"].get_input_stats()
    action_stats = model.normalizer["action"].get_input_stats()
    generator = torch.Generator(device=device).manual_seed(20260712)
    pc_mean = pc_stats["mean"].to(device)
    pc_std = pc_stats["std"].to(device).clamp_min(1e-6)
    point_cloud = pc_mean.view(1, 1, 1, 3).expand(1, 1, 5120, 3).clone()
    point_cloud += 0.05 * pc_std.view(1, 1, 1, 3) * torch.randn(
        point_cloud.shape, generator=generator, device=device
    )
    action_mean = action_stats["mean"].to(device)
    action_std = action_stats["std"].to(device).clamp_min(1e-6)
    previous = (action_mean + 0.1 * action_std).view(1, -1)
    return {"obs": {"point_cloud": point_cloud}, "prev_true_trajectory": previous}


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    if checkpoint.parent != config_path.parent:
        raise ValueError("Checkpoint and config must come from the same category directory")
    cfg = OmegaConf.load(config_path)
    weights = load_file(str(checkpoint), device="cpu")
    if not weights:
        raise RuntimeError("Safetensors file is empty")

    category = checkpoint.parent.name
    cfg_dataset = cfg.get("dataset")
    if isinstance(cfg_dataset, (list, tuple)) or hasattr(cfg_dataset, "_content"):
        cfg_dataset = cfg_dataset[0]
    expected_dataset = f"{category}-v2"
    if str(cfg_dataset) != expected_dataset:
        raise ValueError(
            f"Config dataset {cfg_dataset!r} does not match category directory "
            f"{category!r} (expected {expected_dataset!r})"
        )

    metrics_path = checkpoint.with_name("metrics.json")
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Checkpoint metadata is missing: {metrics_path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if metrics.get("dataset") != expected_dataset:
        raise ValueError("metrics.json dataset does not match config/category")
    if metrics.get("category") != category:
        raise ValueError("metrics.json category does not match checkpoint directory")
    if not isinstance(metrics.get("seed"), int):
        raise ValueError("metrics.json seed must be an integer")
    weight_variant = metrics.get("weight_variant")
    if weight_variant not in {"ema", "raw"}:
        raise ValueError("metrics.json weight_variant must be 'ema' or 'raw'")

    with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
        tensor_metadata = handle.metadata() or {}
    if tensor_metadata.get("dataset") != expected_dataset:
        raise ValueError("Safetensors dataset metadata does not match config/category")
    if tensor_metadata.get("weight_variant") != weight_variant:
        raise ValueError("Safetensors and metrics weight variants do not match")
    if args.source_checkpoint is not None:
        compare_with_source(weights, args.source_checkpoint, weight_variant)

    model = get_model_diffusion(
        config=cfg,
        which=cfg.model.backbone,
        io_type=cfg.task_name,
        device="cpu",
    )
    model.load_state_dict(weights, strict=True)
    normalizer_fields = {
        key.split(".", 3)[2]
        for key in weights
        if key.startswith("normalizer.params_dict.")
    }
    if not {"action", "point_cloud"}.issubset(normalizer_fields):
        raise RuntimeError(f"Missing normalizer fields: {sorted(normalizer_fields)}")

    result = {
        "strict_load": True,
        "tensor_count": len(weights),
        "normalizer_fields": sorted(normalizer_fields),
        "exact_source_match": args.source_checkpoint is not None,
        "release_manifest_verified": verify_release_manifest(
            checkpoint,
            config_path,
            allow_missing=args.allow_missing_manifest,
        ),
        "category": category,
        "dataset": expected_dataset,
        "seed": metrics["seed"],
        "weight_variant": weight_variant,
    }
    if args.run_inference:
        device = torch.device(args.device)
        del weights
        model.to(device).eval()
        inference_generator = torch.Generator(device=device).manual_seed(20260712)
        with torch.inference_mode():
            prediction = model.predict_action(
                make_smoke_input(model, device),
                generator=inference_generator,
            )["action"]
        expected_shape = (1, cfg.horizon, cfg.action_dim)
        if tuple(prediction.shape) != expected_shape:
            raise RuntimeError(
                f"Unexpected prediction shape: {tuple(prediction.shape)} != {expected_shape}"
            )
        if not torch.isfinite(prediction).all():
            raise RuntimeError("Inference output contains non-finite values")
        result["inference"] = {
            "device": str(device),
            "shape": list(prediction.shape),
            "finite": True,
        }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
