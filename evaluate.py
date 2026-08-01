"""Canonical checkpoint evaluation for 3D-CovDiffusion."""

import argparse
import json
import pathlib
import random

import dill
import numpy as np
import torch
from omegaconf import ListConfig, OmegaConf

from models import get_model_diffusion


PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
EVAL_CONFIGS = PROJECT_ROOT / "configs" / "covdiffusion"


def compose_public_evaluation_config(released_config):
    """Restore evaluation fields intentionally omitted from released configs."""

    dataset = released_config.get("dataset")
    if isinstance(dataset, (list, tuple, ListConfig)):
        dataset_name = str(dataset[0])
    else:
        dataset_name = str(dataset)
    category_file = EVAL_CONFIGS / f"{dataset_name.removesuffix('-v2')}.yaml"
    if not category_file.is_file():
        raise FileNotFoundError(
            f"No public category config for dataset {dataset_name!r}: {category_file}"
        )
    return OmegaConf.merge(
        OmegaConf.load(EVAL_CONFIGS / "default.yaml"),
        OmegaConf.load(EVAL_CONFIGS / "covdiffusion.yaml"),
        OmegaConf.load(category_file),
        released_config,
    )


def load_evaluation_checkpoint(checkpoint_path, config_path=None, weight_variant="auto"):
    """Load a tensor-only release or a trusted legacy training checkpoint."""

    checkpoint_path = pathlib.Path(checkpoint_path)
    if checkpoint_path.suffix == ".safetensors":
        resolved_config = (
            pathlib.Path(config_path)
            if config_path is not None
            else checkpoint_path.with_name("config.yaml")
        )
        if not resolved_config.is_file():
            raise FileNotFoundError(
                f"Config not found at {resolved_config}. Pass --config_path explicitly."
            )

        metrics_path = checkpoint_path.with_name("metrics.json")
        if not metrics_path.is_file():
            raise FileNotFoundError(f"Checkpoint provenance is missing: {metrics_path}")
        metadata = json.loads(metrics_path.read_text(encoding="utf-8"))
        released_variant = metadata.get("weight_variant")
        if released_variant not in {"ema", "raw"}:
            raise ValueError(
                f"Unsupported or missing weight_variant in {metrics_path}: "
                f"{released_variant!r}"
            )
        if weight_variant != "auto" and weight_variant != released_variant:
            raise ValueError(
                f"Requested {weight_variant} weights, but {checkpoint_path} "
                f"contains {released_variant} weights."
            )

        from safetensors.torch import load_file

        cfg = compose_public_evaluation_config(OmegaConf.load(resolved_config))
        state_dict = load_file(str(checkpoint_path), device="cpu")
        return cfg, state_dict, f"{released_variant} safetensors"

    payload = torch.load(
        checkpoint_path,
        pickle_module=dill,
        map_location="cpu",
        weights_only=False,
    )
    if "cfg" not in payload:
        raise KeyError("'cfg' not found in legacy checkpoint payload")
    state_dicts = payload.get("state_dicts", {})
    if weight_variant == "raw":
        state_key = "model"
    elif weight_variant == "ema":
        state_key = "ema_model"
    else:
        state_key = "ema_model" if "ema_model" in state_dicts else "model"
    if state_key not in state_dicts:
        raise KeyError(f"state_dicts.{state_key} not found in legacy checkpoint")
    return payload["cfg"], state_dicts[state_key], f"legacy {state_key}"


def resolve_training_seed(checkpoint_path, cfg, requested_seed=None):
    """Read and optionally assert the checkpoint's training seed."""

    checkpoint_path = pathlib.Path(checkpoint_path)
    if checkpoint_path.suffix == ".safetensors":
        metrics_path = checkpoint_path.with_name("metrics.json")
        if not metrics_path.is_file():
            raise FileNotFoundError(f"Checkpoint provenance is missing: {metrics_path}")
        source_seed = json.loads(metrics_path.read_text(encoding="utf-8")).get("seed")
    else:
        source_seed = cfg.get("seed")
    if source_seed is None:
        raise ValueError("Checkpoint does not record its training seed")
    source_seed = int(source_seed)
    if requested_seed is not None and int(requested_seed) != source_seed:
        raise ValueError(
            f"Requested training seed {requested_seed} does not match "
            f"checkpoint training seed {source_seed}"
        )
    return source_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a 3D-CovDiffusion checkpoint on the fixed "
            "evaluation-ready split."
        )
    )
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument(
        "--config_path",
        default=None,
        help="Config for safetensors (default: config.yaml beside the weights).",
    )
    parser.add_argument(
        "--weight_variant",
        choices=["auto", "ema", "raw"],
        default="auto",
        help="Assert a released weight variant; legacy auto prefers EMA.",
    )
    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--dataset_split", default="test")
    parser.add_argument(
        "--eval_episodes",
        type=int,
        default=10,
        help="Episodes to run; use 0 to evaluate the complete split.",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42, help="Rollout seed.")
    parser.add_argument(
        "--training_seed",
        type=int,
        default=None,
        help="Optional assertion against checkpoint provenance.",
    )
    parser.add_argument("--output_dir_base", default="outputs")
    parser.add_argument("--run_name", default=None)
    parser.add_argument(
        "--coverage_mode",
        choices=["face-count", "area-weighted"],
        default="area-weighted",
    )
    parser.add_argument(
        "--spray_radius",
        type=float,
        default=0.1,
        help="Coverage radius in normalized object-centric coordinates.",
    )
    parser.add_argument(
        "--save_artifacts",
        action="store_true",
        help="Save the centered prediction PLY in addition to metrics.",
    )
    parser.add_argument(
        "--single_episode_idx",
        type=int,
        default=-1,
        help="Evaluate only this zero-based index from the fixed split.",
    )
    parser.add_argument(
        "--include_gt_conditioned",
        action="store_true",
        help="Also run the diagnostic ground-truth-history condition.",
    )
    return parser.parse_args()


def configured_dataset_names(cfg):
    dataset = cfg.dataset
    if isinstance(dataset, (list, tuple, ListConfig)):
        return [str(item) for item in dataset]
    return [str(dataset)]


def main():
    args = parse_args()

    # Preserve the released protocol's model, DataLoader, and rollout RNG order.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    checkpoint_path = pathlib.Path(args.checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    print(f"Loading checkpoint from: {checkpoint_path}")

    try:
        cfg, model_state_dict, loaded_variant = load_evaluation_checkpoint(
            checkpoint_path,
            config_path=args.config_path,
            weight_variant=args.weight_variant,
        )
    except Exception as error:
        raise RuntimeError(f"Could not load checkpoint: {error}") from error
    training_seed = resolve_training_seed(
        checkpoint_path, cfg, requested_seed=args.training_seed
    )
    run_name = args.run_name or f"test_{checkpoint_path.stem}_s{training_seed}"
    print(f"Loaded {loaded_variant} weights.")
    print(f"Starting test run: {run_name}")
    print(f"Checkpoint training seed: {training_seed}; rollout seed: {args.seed}")

    if args.dataset_name:
        configured = configured_dataset_names(cfg)
        if configured != [args.dataset_name]:
            raise ValueError(
                f"Requested dataset {args.dataset_name!r} does not match "
                f"checkpoint dataset {configured}. Cross-category evaluation is unsupported."
            )
        cfg.dataset = [args.dataset_name]
    cfg.task_name = "CovDiffusion"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_model_diffusion(
        config=cfg,
        which=cfg.model.backbone,
        io_type=cfg.task_name,
        device=device,
    )
    try:
        model.load_state_dict(model_state_dict, strict=True)
    except Exception as error:
        raise RuntimeError(f"Could not load model state_dict strictly: {error}") from error
    model.to(device)
    model.eval()
    print("Model and embedded normalizer loaded successfully (strict=True).")

    from covdiffusion.env_runner.covdiffusion_runner import CovDiffusionRunner
    from utils.dataset.covdiffusion_rollout_dataset import CovDiffusionRolloutDataset

    cfg.coverage_spray_radius = args.spray_radius
    cfg.coverage_area_weighted = args.coverage_mode == "area-weighted"
    rollout_dataset = CovDiffusionRolloutDataset(
        config=cfg,
        split=args.dataset_split,
        seed=args.seed,
    )
    if len(rollout_dataset) == 0:
        raise RuntimeError(
            f"Rollout dataset for split {args.dataset_split!r} is empty. "
            "Check COVDIFFUSION_EVAL_CACHE_ROOT and the evaluation-ready manifest."
        )

    if args.single_episode_idx >= 0:
        if args.single_episode_idx >= len(rollout_dataset):
            raise IndexError(
                f"single_episode_idx={args.single_episode_idx} out of range "
                f"for dataset of size {len(rollout_dataset)}"
            )
        print(f"Restricting evaluation to episode index {args.single_episode_idx}.")
        rollout_dataset = [rollout_dataset[args.single_episode_idx]]

    dataset_display_name = configured_dataset_names(cfg)[0]
    rollout_loader = torch.utils.data.DataLoader(
        rollout_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
    )
    available = len(rollout_loader)
    effective_episodes = (
        available if args.eval_episodes <= 0 else min(available, args.eval_episodes)
    )
    print(
        f"Loaded {dataset_display_name} ({args.dataset_split}): {available} episodes; "
        f"running {effective_episodes}."
    )

    runner = CovDiffusionRunner(
        output_dir=args.output_dir_base,
        eval_episodes=effective_episodes,
        batch_size=1,
        num_workers=args.workers,
        seed=args.seed,
        training_seed=training_seed,
        save_artifacts=args.save_artifacts,
        condition_modes=(
            ["GT_Cond", "Pred_Cond"]
            if args.include_gt_conditioned
            else ["Pred_Cond"]
        ),
        config=cfg,
    )
    runner.run(
        policy=model,
        dataloader=rollout_loader,
        split=args.dataset_split,
        run_name=run_name,
        dataset_name=dataset_display_name,
        paper_vis_mode=False,
    )
    print(
        "Evaluation complete: "
        f"{pathlib.Path(args.output_dir_base) / run_name / 'test_results.json'}"
    )


if __name__ == "__main__":
    main()
