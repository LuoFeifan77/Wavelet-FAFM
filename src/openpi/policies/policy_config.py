import logging
import os
import pathlib
from typing import Any

import jax.numpy as jnp
import numpy as np

import openpi.models.model as _model
from openpi.models import fafm_config as _fafm_config
import openpi.policies.policy as _policy
import openpi.shared.download as download
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
import openpi.transforms as transforms


def create_trained_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    *,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, transforms.NormStats] | None = None,
    pytorch_device: str | None = None,
) -> _policy.Policy:
    """Create a policy from a trained checkpoint.

    Args:
        train_config: The training config to use to create the model.
        checkpoint_dir: The directory to load the model from.
        repack_transforms: Optional transforms that will be applied before any other transforms.
        sample_kwargs: The kwargs to pass to the `sample_actions` method. If not provided, the default
            kwargs will be used.
        default_prompt: The default prompt to use for the policy. Will inject the prompt into the input
            data if it doesn't already exist.
        norm_stats: The norm stats to use for the policy. If not provided, the norm stats will be loaded
            from the checkpoint directory.
        pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda", "cuda:0").
                      If None and is_pytorch=True, will use "cuda" if available, otherwise "cpu".

    Note:
        The function automatically detects whether the model is PyTorch-based by checking for the
        presence of "model.safensors" in the checkpoint directory.
    """
    repack_transforms = repack_transforms or transforms.Group()
    checkpoint_dir = download.maybe_download(str(checkpoint_dir))

    # Check if this is a PyTorch model by looking for model.safetensors
    weight_path = os.path.join(checkpoint_dir, "model.safetensors")
    is_pytorch = os.path.exists(weight_path)

    logging.info("Loading model...")
    if is_pytorch:
        model = train_config.model.load_pytorch(train_config, weight_path)
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    else:
        model = train_config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16))
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if norm_stats is None:
        # We are loading the norm stats from the checkpoint instead of the config assets dir to make sure
        # that the policy is using the same normalization stats as the original training process.
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)

    # FAFM-specific inference setup: conditional norm stats override.
    # NOTE: This logic ONLY applies to FAFM models. Pi0Config and other models are NOT affected.
    # 
    # BACKGROUND (2026-04-26 Bug Discovery):
    # - Original code replaced action norm_stats with identity transform [-3, 3]
    # - This caused incorrect unnormalization: actions became ~2-3x smaller than expected
    # - Resulted in anomalous behaviors like "gripper always moving upward"
    # 
    # ROOT CAUSE:
    # - Training: real_velocity -> Normalize(real_q01/q99) -> normalized_velocity[-1,1] -> model
    # - Inference (BUG): model_output[-1,1] -> Unnormalize(identity) -> wrong_velocity -> wrong_delta
    # - Inference (FIXED): model_output[-1,1] -> Unnormalize(real_q01/q99) -> real_velocity -> correct_delta
    #
    # SOLUTION: 
    # - Use actual norm_stats from training by default (use_identity_action_norm=False)
    # - Optionally allow identity override via config flag (use_identity_action_norm=True)
    # - Identity mode may still cause scale issues but useful for debugging
    #
    # Evidence: /root/gjn/fafm/openpi/unnormalize_bug_proof.json
    # 
    if isinstance(train_config.model, _fafm_config.FAFMConfig):
        # FAFM model detected - apply special norm_stats handling
        fafm_cfg = train_config.model
        
        if fafm_cfg.use_identity_action_norm:
            # Use identity normalization (avoids norm_stats shape mismatch issues)
            logging.info(
                "FAFM: Using identity action normalization (q01=-3, q99=3). "
                "This avoids norm_stats shape mismatch errors."
            )
            d_real = fafm_cfg.action_dim
            norm_stats = dict(norm_stats)
            norm_stats["actions"] = transforms.NormStats(
                mean=np.zeros(d_real, dtype=np.float32),
                std=np.ones(d_real, dtype=np.float32),
                q01=np.full(d_real, -3.0, dtype=np.float32),
                q99=np.full(d_real, 3.0, dtype=np.float32),
            )
        else:
            # Use actual norm_stats from training (recommended)
            logging.info(
                "FAFM: Using actual norm_stats from training for correct unnormalization."
            )
        
        # ODE is trained in relative / action-cumsum space (A0=0), so no state
        # denormalization is needed at inference. state_norm stats are not used.
    else:
        # Pi0Config or other models - use norm_stats as-is without modification
        logging.info(
            f"{train_config.model.__class__.__name__}: Using standard norm_stats without modification."
        )

    # Determine the device to use for PyTorch models
    if is_pytorch and pytorch_device is None:
        try:
            import torch

            pytorch_device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            pytorch_device = "cpu"

    return _policy.Policy(
        model,
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs,
        metadata=train_config.policy_metadata,
        is_pytorch=is_pytorch,
        pytorch_device=pytorch_device if is_pytorch else None,
    )
