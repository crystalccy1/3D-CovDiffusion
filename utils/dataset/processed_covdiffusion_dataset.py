"""Dataset loader for the compact 3D-CovDiffusion train-ready release format."""

from pathlib import Path
from typing import Dict

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset

from covdiffusion.common.trajectory_history import previous_action_index
from covdiffusion.model.common.normalizer import LinearNormalizer


SCHEMA_VERSION = "3dcov-train-v1"


class ProcessedCovDiffusionDataset(Dataset):
    """Read compact processed buffers without requiring the raw source dataset.

    The release format stores temporal arrays under ``data`` and one point cloud
    per episode under ``obs``. Actions are kept in memory (tens of megabytes per
    category), while point clouds remain in the on-disk Zarr store and are read
    per episode.
    """

    def __init__(
        self,
        zarr_path,
        horizon=16,
        pad_before=0,
        pad_after=0,
        seed=42,
        val_ratio=0.0,
        max_train_episodes=None,
        config=None,
    ):
        super().__init__()
        self.zarr_path = Path(zarr_path).expanduser().resolve()
        self.horizon = int(horizon)
        self.config = config
        self.n_obs_steps = int(getattr(config, "n_obs_steps", 1))
        if self.horizon < 1:
            raise ValueError("horizon must be at least 1")
        if self.n_obs_steps < 1:
            raise ValueError("n_obs_steps must be at least 1")
        if pad_before != 0 or pad_after != 0 or val_ratio != 0.0:
            raise ValueError(
                "The released training protocol uses no sequence padding or "
                "validation split."
            )
        if max_train_episodes is not None:
            raise ValueError("Episode subsampling is not part of the release protocol.")
        del seed

        if not self.zarr_path.is_dir():
            raise FileNotFoundError(
                f"Processed training buffer not found: {self.zarr_path}. "
                "Download the Hugging Face dataset and pass processed_data_root=<download-root>."
            )

        self.root = zarr.open(str(self.zarr_path), mode="r")
        schema_version = self.root.attrs.get("schema_version")
        if schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported processed dataset schema {schema_version!r}; "
                f"expected {SCHEMA_VERSION!r}."
            )

        for path in ("meta/episode_ends", "data/action", "data/stroke_ids", "obs/point_cloud"):
            if path not in self.root:
                raise ValueError(f"Processed dataset is missing required array: {path}")

        episode_ends = np.asarray(self.root["meta/episode_ends"][:], dtype=np.int64)
        action = np.asarray(self.root["data/action"][:], dtype=np.float32)
        stroke_ids = self.root["data/stroke_ids"]
        self.point_clouds = self.root["obs/point_cloud"]

        if episode_ends.ndim != 1 or len(episode_ends) == 0:
            raise ValueError("meta/episode_ends must be a non-empty one-dimensional array")
        if int(episode_ends[-1]) != len(action) or len(stroke_ids) != len(action):
            raise ValueError("Temporal array lengths do not match the final episode boundary")
        if len(self.point_clouds) != len(episode_ends):
            raise ValueError("obs/point_cloud must contain exactly one point cloud per episode")
        if action.ndim != 2 or action.shape[1] != 24:
            raise ValueError(f"Expected action shape [steps, 24], got {action.shape}")
        if self.point_clouds.ndim != 3 or tuple(self.point_clouds.shape[1:]) != (5120, 3):
            raise ValueError(
                f"Expected point-cloud shape [episodes, 5120, 3], got {self.point_clouds.shape}"
            )

        self.action = action
        self.episode_ends = episode_ends
        self.episode_starts = np.zeros_like(episode_ends)
        self.episode_starts[1:] = episode_ends[:-1]
        self.action_valid_mask = ~(action == -100).all(axis=1)
        self.indices = np.asarray(
            [
                (episode_idx, start, start + self.horizon)
                for episode_idx, (episode_start, episode_end) in enumerate(
                    zip(self.episode_starts, self.episode_ends)
                )
                for start in range(
                    int(episode_start), int(episode_end) - self.horizon + 1
                )
            ],
            dtype=np.int64,
        ).reshape(-1, 3)

        print(
            "Loaded processed 3D-CovDiffusion buffer: "
            f"{len(self.episode_ends)} episodes, "
            f"{len(self.action)} steps, {len(self.indices)} sequences"
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        episode_idx, start, end = map(int, self.indices[idx])
        episode_start = int(self.episode_starts[episode_idx])
        action = self.action[start:end]
        previous = previous_action_index(
            start, episode_start, self.action_valid_mask
        )
        prev_action = self.action[previous]
        point_cloud = np.asarray(self.point_clouds[episode_idx], dtype=np.float32)

        # The point cloud is static within an episode.  Materialize only the
        # observation steps consumed by DP3 instead of repeating it for the
        # entire action horizon.
        point_cloud_sequence = np.broadcast_to(
            point_cloud,
            (self.n_obs_steps,) + point_cloud.shape,
        ).copy()
        action_tensor = torch.from_numpy(action)

        return {
            "obs": {"point_cloud": torch.from_numpy(point_cloud_sequence)},
            "action": action_tensor,
            "prev_true_trajectory": torch.from_numpy(prev_action),
        }

    def get_normalizer(self, mode="limits", **kwargs):
        """Fit the same normalizer using compact, non-repeated observations."""

        action_data = self.action.copy()
        padding_mask = action_data == -100
        if padding_mask.any():
            for feature_idx in range(action_data.shape[1]):
                valid = action_data[:, feature_idx][~padding_mask[:, feature_idx]]
                replacement = valid.mean() if len(valid) else 0.0
                action_data[padding_mask[:, feature_idx], feature_idx] = replacement

        normalizer = LinearNormalizer()
        normalizer.fit(
            data={
                "action": action_data,
                "point_cloud": np.asarray(self.point_clouds[:], dtype=np.float32),
            },
            last_n_dims=1,
            mode=mode,
            **kwargs,
        )
        return normalizer
