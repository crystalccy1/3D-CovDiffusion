"""Canonical 3D-CovDiffusion policy used for training and inference."""

import copy
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from omegaconf import OmegaConf

from covdiffusion.common.model_util import print_params
from covdiffusion.common.pytorch_util import dict_apply
from covdiffusion.model.common.normalizer import LinearNormalizer
from covdiffusion.model.diffusion.conditional_unet1d import ConditionalUnet1D
from covdiffusion.model.diffusion.mask_generator import LowdimMaskGenerator
from covdiffusion.model.vision.pointnet_extractor import DP3Encoder
from covdiffusion.policy.base_policy import BasePolicy


class DP3(BasePolicy):
    """Geometry-conditioned diffusion policy from the released experiments.

    Archived checkpoints use a 128-dimensional point-cloud feature and a
    128-dimensional encoding of the previous 24-D trajectory chunk.  Several
    constructor arguments are retained because they occur in archived configs;
    the implementation intentionally supports only their released values.
    """

    def __init__(
        self,
        shape_meta: dict,
        noise_scheduler: DDPMScheduler,
        horizon,
        n_action_steps,
        n_obs_steps,
        num_inference_steps=None,
        obs_as_global_cond=True,
        diffusion_step_embed_dim=256,
        down_dims=(256, 512, 1024),
        kernel_size=5,
        n_groups=8,
        condition_type="film",
        use_down_condition=True,
        use_mid_condition=True,
        use_up_condition=True,
        encoder_output_dim=128,
        crop_shape=None,
        use_pc_color=False,
        pointnet_type="pointnet",
        pointcloud_encoder_cfg=None,
        encoder_ablation="covdiffusion",
        traj_mlp_hidden_dim=128,
        traj_mlp_output_dim=128,
        **kwargs,
    ):
        super().__init__()
        del crop_shape, pointcloud_encoder_cfg

        if kwargs:
            names = ", ".join(sorted(kwargs))
            raise TypeError(f"Unsupported DP3 options: {names}.")
        if not obs_as_global_cond:
            raise ValueError("The released policy requires obs_as_global_cond=True.")
        if condition_type != "film":
            raise ValueError("The released policy uses film conditioning.")
        if use_pc_color:
            raise ValueError("The released policy uses XYZ point clouds without color.")
        if pointnet_type != "pointnet" or encoder_ablation != "covdiffusion":
            raise ValueError(
                "Only the released 3D-CovDiffusion point-cloud encoder is supported."
            )
        if traj_mlp_hidden_dim != 128 or traj_mlp_output_dim != 128:
            raise ValueError("The released trajectory encoder dimensions are 128/128.")
        if noise_scheduler.config.prediction_type != "sample":
            raise ValueError("The released policy requires prediction_type='sample'.")

        action_shape = shape_meta["action"]["shape"]
        if len(action_shape) != 1:
            raise ValueError(f"Expected one-dimensional action shape, got {action_shape}.")
        action_dim = action_shape[0]
        if action_dim != 24:
            raise ValueError(f"The released policy expects 24-D actions, got {action_dim}.")

        # The historical constructor ignored encoder_output_dim and used 128.
        # Keeping that behavior preserves every released state_dict shape.
        del encoder_output_dim
        pointcloud_encoder_cfg = OmegaConf.create(
            {
                "in_channels": 3,
                "out_channels": 128,
                "use_layernorm": True,
                "final_norm": "layernorm",
                "normal_channel": False,
            }
        )
        obs_encoder = DP3Encoder(
            observation_space={"point_cloud": [5120, 3]},
            img_crop_shape=None,
            out_channel=128,
            pointcloud_encoder_cfg=pointcloud_encoder_cfg,
            use_pc_color=False,
            pointnet_type="pointnet",
            encoder_ablation="covdiffusion",
        )

        self.traj_mlp = nn.Sequential(
            nn.Linear(24, traj_mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(traj_mlp_hidden_dim, traj_mlp_output_dim),
        )

        obs_feature_dim = obs_encoder.output_shape()
        global_cond_dim = obs_feature_dim * n_obs_steps + traj_mlp_output_dim
        model = ConditionalUnet1D(
            input_dim=action_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            condition_type=condition_type,
            use_down_condition=use_down_condition,
            use_mid_condition=use_mid_condition,
            use_up_condition=use_up_condition,
        )

        self.obs_encoder = obs_encoder
        self.model = model
        self.noise_scheduler = noise_scheduler
        # Retained as non-parameter compatibility attributes for old workspaces.
        self.noise_scheduler_pc = copy.deepcopy(noise_scheduler)
        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False,
        )
        self.normalizer = LinearNormalizer()

        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = True
        self.condition_type = condition_type
        self.use_pc_color = False
        self.pointnet_type = "pointnet"
        self.traj_mlp_output_dim = traj_mlp_output_dim
        self.kwargs = {}
        self.num_inference_steps = (
            noise_scheduler.config.num_train_timesteps
            if num_inference_steps is None
            else num_inference_steps
        )

        print_params(self)

    def conditional_sample(
        self,
        condition_data,
        condition_mask,
        local_cond=None,
        global_cond=None,
        generator=None,
    ):
        """Generate one trajectory without retaining visualization snapshots."""
        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator,
        )

        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        for timestep in self.noise_scheduler.timesteps:
            trajectory[condition_mask] = condition_data[condition_mask]
            model_output = self.model(
                sample=trajectory,
                timestep=timestep,
                local_cond=local_cond,
                global_cond=global_cond,
            )
            trajectory = self.noise_scheduler.step(
                model_output,
                timestep,
                trajectory,
                generator=generator,
            ).prev_sample

        trajectory[condition_mask] = condition_data[condition_mask]
        return trajectory

    @staticmethod
    def _require_trajectory_matrix(trajectory: torch.Tensor) -> torch.Tensor:
        if trajectory.ndim != 2 or trajectory.shape[-1] != 24:
            raise ValueError(
                "prev_true_trajectory must have shape [batch, 24], "
                f"got {tuple(trajectory.shape)}."
            )
        return trajectory

    def _global_condition(
        self,
        normalized_obs: Dict[str, torch.Tensor],
        normalized_previous: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = normalized_obs["point_cloud"].shape[0]
        current_obs = dict_apply(
            normalized_obs,
            lambda value: value[:, : self.n_obs_steps, ...].reshape(
                -1, *value.shape[2:]
            ),
        )
        geometry_features = self.obs_encoder(current_obs).reshape(batch_size, -1)
        trajectory_features = self.traj_mlp(normalized_previous)
        return torch.cat([geometry_features, trajectory_features], dim=-1)

    def predict_action(
        self,
        obs_dict: Dict[str, torch.Tensor],
        generator: torch.Generator = None,
    ) -> Dict[str, torch.Tensor]:
        """Predict an ordered action sequence from geometry and recent motion."""
        if "prev_true_trajectory" not in obs_dict:
            raise KeyError("obs_dict must contain 'prev_true_trajectory'.")

        previous = self._require_trajectory_matrix(obs_dict["prev_true_trajectory"])
        normalized_previous = self.normalizer["action"].normalize(previous)
        if (normalized_previous == 0).all(dim=1).any():
            raise ValueError("prev_true_trajectory contains an all-zero row.")

        normalized_obs = self.normalizer.normalize(obs_dict["obs"])
        normalized_obs["point_cloud"] = normalized_obs["point_cloud"][..., :3]
        batch_size = normalized_obs["point_cloud"].shape[0]
        global_cond = self._global_condition(normalized_obs, normalized_previous)

        condition_data = torch.zeros(
            (batch_size, self.horizon, self.action_dim),
            device=self.device,
            dtype=self.dtype,
        )
        condition_mask = torch.zeros_like(condition_data, dtype=torch.bool)
        normalized_sample = self.conditional_sample(
            condition_data,
            condition_mask,
            global_cond=global_cond,
            generator=generator,
        )

        action_pred = self.normalizer["action"].unnormalize(normalized_sample)
        start = self.n_obs_steps - 1
        action = action_pred[:, start : start + self.n_action_steps]
        return {"action": action, "action_pred": action_pred}

    def compute_loss(self, batch, save_dir=None):
        """Compute the masked denoising objective used by the paper."""
        del save_dir
        normalized_obs = self.normalizer.normalize(batch["obs"])
        normalized_obs["point_cloud"] = normalized_obs["point_cloud"][..., :3]

        action = batch["action"]
        padding_mask = action != -100
        offset = self.normalizer["action"].params_dict["offset"]
        offset_expanded = offset.view(1, 1, -1).expand_as(action)
        action_for_norm = torch.where(padding_mask, action, offset_expanded)
        if (action_for_norm == -100).any():
            raise ValueError("Action padding could not be replaced before normalization.")
        trajectory = self.normalizer["action"].normalize(action_for_norm)
        trajectory[~padding_mask] = 0

        if "prev_true_trajectory" not in batch:
            raise KeyError("batch must contain 'prev_true_trajectory'.")
        previous = self._require_trajectory_matrix(batch["prev_true_trajectory"])
        if (previous == -100).any():
            raise ValueError("prev_true_trajectory contains padding values.")
        normalized_previous = self.normalizer["action"].normalize(previous)
        if (normalized_previous == 0).all(dim=1).any():
            raise ValueError("prev_true_trajectory contains an all-zero row.")

        global_cond = self._global_condition(normalized_obs, normalized_previous)
        condition_mask = self.mask_generator(trajectory.shape)
        if condition_mask.any():
            raise RuntimeError("The released global-conditioning mask must be empty.")

        noise = torch.randn(trajectory.shape, device=trajectory.device)
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (trajectory.shape[0],),
            device=trajectory.device,
        ).long()
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps
        )
        noisy_trajectory[condition_mask] = trajectory[condition_mask]

        prediction = self.model(
            sample=noisy_trajectory,
            timestep=timesteps,
            local_cond=None,
            global_cond=global_cond,
        )
        loss_mask = (~condition_mask) & padding_mask
        elementwise_loss = F.mse_loss(prediction, trajectory, reduction="none")
        elementwise_loss = elementwise_loss * loss_mask.to(elementwise_loss.dtype)
        valid_elements = loss_mask.sum()
        if valid_elements > 0:
            loss = elementwise_loss.sum() / valid_elements
        else:
            loss = elementwise_loss.sum() * 0.0

        return loss, {"bc_loss": loss.item()}

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())
