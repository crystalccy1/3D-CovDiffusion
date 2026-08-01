"""Model factory for the public 3D-CovDiffusion release."""

from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from covdiffusion.policy.dp3 import DP3


def _value(config, name, default):
    """Read an OmegaConf or object attribute with a fallback."""
    return getattr(config, name, default)


def get_model_diffusion(
    config,
    which,
    io_type=None,
    custom_model_config=None,
    device="cpu",
):
    """Build the 3D-CovDiffusion policy used by training and evaluation."""
    if which != "dp3":
        raise ValueError(f"Unsupported policy backbone: {which!r}; expected 'dp3'.")
    if io_type not in {None, "CovDiffusion"}:
        raise ValueError(f"Unsupported task type: {io_type!r}.")

    model_config = config.model if custom_model_config is None else custom_model_config
    if _value(model_config, "pretrained", False):
        raise NotImplementedError(
            "Encoder pretraining is not configured in the public release."
        )

    noise_config = _value(config, "noise_scheduler", {})
    noise_scheduler = DDIMScheduler(
        beta_schedule=noise_config.get("beta_schedule", "squaredcos_cap_v2"),
        beta_start=noise_config.get("beta_start", 0.0001),
        beta_end=noise_config.get("beta_end", 0.02),
        num_train_timesteps=noise_config.get("num_train_timesteps", 100),
        clip_sample=noise_config.get("clip_sample", True),
        set_alpha_to_one=noise_config.get("set_alpha_to_one", True),
        steps_offset=noise_config.get("steps_offset", 0),
        prediction_type=noise_config.get("prediction_type", "sample"),
    )

    action_dim = _value(config, "action_dim", 24)
    shape_meta = _value(
        config,
        "shape_meta",
        {
            "obs": {"point_cloud": {"shape": [5120, 3]}},
            "action": {"shape": [action_dim]},
        },
    )
    diffusion_config = _value(config, "diffusion", {})
    model = DP3(
        shape_meta=shape_meta,
        noise_scheduler=noise_scheduler,
        horizon=_value(config, "horizon", 16),
        n_action_steps=_value(config, "n_action_steps", 100),
        n_obs_steps=_value(config, "n_obs_steps", 1),
        num_inference_steps=diffusion_config.get("num_inference_steps", 10),
        obs_as_global_cond=diffusion_config.get("obs_as_global_cond", True),
        diffusion_step_embed_dim=diffusion_config.get(
            "diffusion_step_embed_dim", 256
        ),
        down_dims=diffusion_config.get("down_dims", (256, 512, 1024)),
        kernel_size=diffusion_config.get("kernel_size", 5),
        n_groups=diffusion_config.get("n_groups", 8),
        condition_type=diffusion_config.get("condition_type", "film"),
        use_down_condition=diffusion_config.get("use_down_condition", True),
        use_mid_condition=diffusion_config.get("use_mid_condition", True),
        use_up_condition=diffusion_config.get("use_up_condition", True),
        encoder_output_dim=_value(config, "encoder_output_dim", 256),
    )
    return model.to(device)
