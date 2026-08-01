"""Point-cloud encoder used by the released 3D-CovDiffusion policy."""

from typing import Dict

import torch
import torch.nn as nn


class PointNetEncoderXYZ(nn.Module):
    """Encode an XYZ point cloud with shared MLPs and global max pooling."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1024,
        use_layernorm: bool = False,
        final_norm: str = "none",
        use_projection: bool = True,
        **_kwargs,
    ):
        super().__init__()
        if in_channels != 3:
            raise ValueError(f"Expected XYZ input with 3 channels, got {in_channels}.")

        block_channels = [64, 128, 256]
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channels[0]),
            nn.LayerNorm(block_channels[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channels[0], block_channels[1]),
            nn.LayerNorm(block_channels[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channels[1], block_channels[2]),
            nn.LayerNorm(block_channels[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
        )

        if final_norm == "layernorm":
            self.final_projection = nn.Sequential(
                nn.Linear(block_channels[-1], out_channels),
                nn.LayerNorm(out_channels),
            )
        elif final_norm == "none":
            self.final_projection = nn.Linear(block_channels[-1], out_channels)
        else:
            raise ValueError(f"Unsupported final_norm: {final_norm!r}.")

        if not use_projection:
            self.final_projection = nn.Identity()

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        features = self.mlp(points)
        features = torch.max(features, dim=1)[0]
        return self.final_projection(features)


class DP3Encoder(nn.Module):
    """Canonical geometry encoder used by all released checkpoints.

    The public model uses XYZ points only: a shared 3-layer MLP, global max
    pooling, and a linear projection with layer normalization.  The argument
    names retained here are part of the archived configuration interface; only
    their canonical values are accepted.
    """

    def __init__(
        self,
        observation_space: Dict,
        img_crop_shape=None,
        out_channel: int = 256,
        state_mlp_size=(64, 64),
        state_mlp_activation_fn=nn.ReLU,
        pointcloud_encoder_cfg=None,
        use_pc_color: bool = False,
        pointnet_type: str = "pointnet",
        encoder_ablation: str = "covdiffusion",
    ):
        super().__init__()
        del img_crop_shape, state_mlp_size, state_mlp_activation_fn

        if use_pc_color:
            raise ValueError("The released policy uses XYZ point clouds without color.")
        if pointnet_type != "pointnet" or encoder_ablation != "covdiffusion":
            raise ValueError(
                "Only the released 3D-CovDiffusion point-cloud encoder is supported."
            )
        if "point_cloud" not in observation_space:
            raise KeyError("observation_space must contain 'point_cloud'.")
        if any(key in observation_space for key in ("agent_pos", "imagin_robot")):
            raise ValueError("The released encoder conditions on point clouds only.")
        if pointcloud_encoder_cfg is None:
            raise ValueError("pointcloud_encoder_cfg is required.")

        self.point_cloud_key = "point_cloud"
        self.n_output_channels = out_channel
        self.extractor = PointNetEncoderXYZ(**pointcloud_encoder_cfg)

    def forward(self, observations: Dict) -> torch.Tensor:
        points = observations[self.point_cloud_key]
        if points.ndim != 3:
            raise ValueError(
                f"Expected point cloud shape [batch, points, xyz], got {points.shape}."
            )
        return self.extractor(points)

    def output_shape(self) -> int:
        return self.n_output_channels
