"""Canonical evaluation-ready rollout adapter."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from utils.dataset.canonical_evaluation_dataset import CanonicalEvaluationDataset


class CovDiffusionRolloutDataset(Dataset):
    """Expose self-contained canonical samples in the runner input format."""

    def __init__(
        self,
        config,
        dataset_paths=None,
        split="test",
        seed=42,
        evaluation_cache_root=None,
        verify_hashes=False,
    ):
        super().__init__()
        del dataset_paths, seed
        self.split = split
        configured = config.dataset
        if isinstance(configured, str):
            categories = [configured]
        else:
            categories = [str(item) for item in configured]
        if len(categories) != 1:
            raise ValueError(
                "Canonical rollout evaluation requires exactly one category"
            )

        root = evaluation_cache_root or os.environ.get(
            "COVDIFFUSION_EVAL_CACHE_ROOT"
        )
        if root is None:
            raise RuntimeError(
                "Set COVDIFFUSION_EVAL_CACHE_ROOT to the evaluation-ready NPZ "
                "directory produced by `python reproduce.py prepare <category>`."
            )
        self.dataset = CanonicalEvaluationDataset(
            root=Path(root),
            category=categories[0],
            split=split,
            verify_hashes=verify_hashes,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        try:
            raw = self.dataset[index]
            trajectory = raw["trajectory"]
            gt_trajectory = raw["gt_trajectory"]
            return {
                "obs": {
                    "point_cloud": np.expand_dims(raw["point_cloud"], axis=0)
                },
                "full_trajectory": trajectory,
                "episode_idx": index,
                "sample_id": raw["sample_id"],
                "mesh_vertices": raw["mesh_vertices"],
                "mesh_faces": raw["mesh_faces"],
                "gt_traj_as_pc": gt_trajectory,
                "traj_as_pc": gt_trajectory,
                "stroke_ids": raw["stroke_ids"],
            }
        except Exception as error:
            raise RuntimeError(
                f"Failed to load evaluation sample {index}; refusing to "
                "silently evaluate a subset"
            ) from error
