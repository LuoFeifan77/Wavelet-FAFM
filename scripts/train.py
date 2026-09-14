import dataclasses
import functools
import logging
import os
import platform
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.experimental
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import swanlab

import openpi.models.model as _model
import openpi.models.fafm_config as _fafm_config
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_swanlab(config: _config.TrainConfig, *, resuming: bool, enabled: bool = True):
    if not enabled:
        swanlab.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "swanlab_id.txt").read_text().strip()
        swanlab.init(
            id=run_id,
            # Local checkpoints can outlive their cloud run or originate offline.
            resume="allow",
            experiment_name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
    else:
        run = swanlab.init(
            experiment_name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "swanlab_id.txt").write_text(run.id)


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


@at.typecheck
def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, *, resume: bool
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            # This will produce an error if the partial params are not a subset of the state.
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


def _compute_fafm_loss_components(
    model: _model.BaseModel,
    rng: at.KeyArrayLike,
    observation: _model.Observation,
    actions: _model.Actions,
) -> dict[str, float]:
    """Compute FAFM loss components for logging.
    
    This recomputes L_FM and L_vel separately for logging purposes.
    It's called outside the JIT-compiled training step to avoid traced array issues.
    
    We compute:
    - loss_fm: Flow Matching loss in coefficient space (averaged over batch and modes)
    - loss_vel: Velocity supervision loss after Fourier decoding (averaged over batch and time)
    
    Returns:
        Dictionary with 'loss_fm' and 'loss_vel' components as Python floats
    """
    from openpi.models.fafm import FAFM, make_physical_times
    from openpi.models import fafm_dct_utils
    import jax
    import jax.numpy as jnp
    
    if not isinstance(model, FAFM):
        return {}
    
    # Recompute loss components (simplified version without gradients)
    B, K, d = actions.shape
    T = model._fafm_config.chunk_duration
    M = model._fafm_config.fourier_modes
    sigma = model._fafm_config.sigma_noise
    
    # Extract DCT coefficients from ground-truth velocities
    c_star = fafm_dct_utils.extract_dct_coeffs(actions, M)  # (B, M+1, d)
    
    # Sample noise for FM
    key = jax.random.fold_in(rng, 12345)  # Use a different fold for logging
    key_noise, key_t = jax.random.split(key)
    c0 = fafm_dct_utils.sample_freq_weighted_noise(M, d, T, B, sigma, key_noise)
    
    # Sample diffusion time
    t_diffusion = jax.random.beta(key_t, 2.0, 1.0, (B,))
    
    # FM interpolation
    c_t = t_diffusion[:, None, None] * c_star + (1 - t_diffusion[:, None, None]) * c0
    u_star = c_star - c0  # Target vector field
    
    # Compute L_FM: We approximate by measuring the magnitude of the vector field
    # (In actual training, this would be MSE(u_pred, u_star), but u_pred requires a forward pass)
    # Here we just report the scale of the target vector field as a proxy
    loss_fm_scale = float(jnp.mean(jnp.square(u_star)))
    
    # Compute L_vel: Decode coefficients and compare with original velocities
    tau = make_physical_times(K, T)
    tau_batch = jnp.broadcast_to(tau[None, :], (B, K))
    
    # Decode clean coefficients to velocities
    v_decoded = jax.vmap(
        lambda c, taus: fafm_dct_utils.decode_coeffs_to_velocity(c, taus, T, K)
    )(c_star, tau_batch)
    
    # Reconstruction error (how well DCT->decode recovers original velocities)
    loss_vel_recon = float(jnp.mean(jnp.square(v_decoded - actions)))
    
    return {
        'loss_fm_scale': loss_fm_scale,  # Scale of vector field (proxy for L_FM)
        'loss_vel_recon': loss_vel_recon,  # DCT reconstruction error (proxy for L_vel)
    }


@at.typecheck
def train_step(
    config: _config.TrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[_model.Observation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array]]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.BaseModel, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions
    ):
        result = model.compute_loss(rng, observation, actions, train=True)
        # Handle both old format (just loss) and new format (loss, metrics)
        if isinstance(result, tuple):
            chunked_loss, metrics = result
            return jnp.mean(chunked_loss), metrics
        else:
            # Old format for backward compatibility
            return jnp.mean(result), {}

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, loss_components), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(
        model, train_rng, observation, actions
    )

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)
    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new, state.ema_params, new_params
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )
    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
        **loss_components,
    }
    return new_state, info


def _compute_action_metrics_jax(
    deltas_pred: jax.Array, deltas_gt: jax.Array,
    delta_clip_threshold: float = 1.0,
    velocities_pred: jax.Array | None = None,
    velocities_gt: jax.Array | None = None,
) -> dict[str, jax.Array]:
    """JIT-compatible metric computation returning JAX scalar arrays.

    A sample is flagged as "exploded" if ANY of its predicted action deltas
    are non-finite (NaN/Inf) OR exceed ``delta_clip_threshold`` in absolute
    value.  The latter catches "soft blow-ups" where v_clip keeps values
    finite but the ODE still diverges far beyond the plausible action range.

    Delta definition:
      - Pi0: deltas are the model's direct output (position deltas)
      - FAFM v3: deltas are adjacent differences of integrated positions
        (pos[k+1] - pos[k]), matching Pi0's evaluation metric
    
    Typical LIBERO adjacent deltas are O(0.005); threshold 0.1 is ~20× above
    normal magnitude, reliably flagging runaway trajectories while allowing
    normal variations.

    Exploded samples are zeroed out before MSE/MAE aggregation so that a
    single divergent sample cannot dominate the metrics.  Boolean indexing is
    avoided (not JIT-compatible); instead we mask with zeros, which biases
    corr slightly downward but keeps the computation JIT-safe.
    
    Args:
        deltas_pred: (B, H, d) predicted position deltas (adjacent differences)
        deltas_gt: (B, H, d) ground-truth position deltas (adjacent differences)
        delta_clip_threshold: threshold for flagging exploded samples
        velocities_pred: (B, H, d) predicted velocities (FAFM only)
        velocities_gt: (B, H, d) ground-truth velocities (FAFM only)
    """
    # deltas shape: (B, H, d) — static H and d, dynamic B
    is_finite   = jnp.isfinite(deltas_pred).all(axis=(1, 2))           # (B,)
    in_range    = (jnp.abs(deltas_pred).max(axis=(1, 2)) < delta_clip_threshold)  # (B,)
    finite_mask = is_finite & in_range
    explode_ratio = 1.0 - jnp.mean(finite_mask.astype(jnp.float32))

    # Zero out non-finite predictions so they don't pollute sums.
    deltas_pred_safe = jnp.where(finite_mask[:, None, None], deltas_pred, jnp.zeros_like(deltas_pred))
    deltas_gt_safe   = jnp.where(finite_mask[:, None, None], deltas_gt,   jnp.zeros_like(deltas_gt))

    n_finite  = jnp.sum(finite_mask).astype(jnp.float32)
    H = deltas_gt.shape[1]
    d = deltas_gt.shape[2]
    denom     = jnp.maximum(n_finite * H * d, 1.0)
    denom_h   = jnp.maximum(n_finite * H, 1.0)

    err        = deltas_pred_safe - deltas_gt_safe
    action_mse = jnp.sum(err ** 2) / denom
    action_mae = jnp.sum(jnp.abs(err)) / denom
    per_dim_mae = jnp.sum(jnp.abs(err), axis=(0, 1)) / denom_h        # (d,)

    # Correlation over all elements (zeros from masked samples bias toward 0,
    # but this is acceptable and keeps the computation JIT-compatible).
    flat_pred = deltas_pred_safe.reshape(-1)
    flat_gt   = deltas_gt_safe.reshape(-1)
    
    # Compute correlation, handle edge cases (zero variance → NaN)
    # When all values are masked or constant, correlation is undefined
    pred_std = jnp.std(flat_pred)
    gt_std = jnp.std(flat_gt)
    # If either has zero variance, set corr to 0.0 instead of NaN
    corr = jnp.where(
        (pred_std > 1e-8) & (gt_std > 1e-8),
        jnp.corrcoef(jnp.stack([flat_pred, flat_gt]))[0, 1],
        0.0  # Default to 0 when correlation is undefined
    )

    metrics: dict[str, jax.Array] = {
        "action/position_mse":           action_mse,
        "action/position_mae":           action_mae,
        "action/position_corr":          corr,
        "action/explode_ratio": explode_ratio,
    }
    # d is a static compile-time constant — safe to loop in JIT.
    for i in range(d):
        metrics[f"action/position_dim{i}_mae"] = per_dim_mae[i]
    
    # FAFM v3: Add velocity-space metrics
    if velocities_pred is not None and velocities_gt is not None:
        vel_pred_safe = jnp.where(finite_mask[:, None, None], velocities_pred, jnp.zeros_like(velocities_pred))
        vel_gt_safe   = jnp.where(finite_mask[:, None, None], velocities_gt,   jnp.zeros_like(velocities_gt))
        
        vel_err = vel_pred_safe - vel_gt_safe
        vel_mse = jnp.sum(vel_err ** 2) / denom
        vel_mae = jnp.sum(jnp.abs(vel_err)) / denom
        vel_per_dim_mae = jnp.sum(jnp.abs(vel_err), axis=(0, 1)) / denom_h
        
        flat_vel_pred = vel_pred_safe.reshape(-1)
        flat_vel_gt   = vel_gt_safe.reshape(-1)
        
        # Compute correlation, handle edge cases (zero variance → NaN)
        vel_pred_std = jnp.std(flat_vel_pred)
        vel_gt_std = jnp.std(flat_vel_gt)
        vel_corr = jnp.where(
            (vel_pred_std > 1e-8) & (vel_gt_std > 1e-8),
            jnp.corrcoef(jnp.stack([flat_vel_pred, flat_vel_gt]))[0, 1],
            0.0  # Default to 0 when correlation is undefined
        )
        
        metrics.update({
            "action/velocity_mse": vel_mse,
            "action/velocity_mae": vel_mae,
            "action/velocity_corr": vel_corr,
        })
        for i in range(d):
            metrics[f"action/velocity_dim{i}_mae"] = vel_per_dim_mae[i]
    
    return metrics


def action_eval_step(
    config: _config.TrainConfig,
    state: training_utils.TrainState,
    rng: at.KeyArrayLike,
    batch: tuple[_model.Observation, _model.Actions],
) -> dict[str, jax.Array]:
    """JIT-compatible action-space evaluation — works for Pi0 and FAFM.

    Returns a dict of JAX scalar arrays.

    Pi0
    ----
    ``batch.actions`` are position delta chunks (B, H, d).
    Compare directly with model.sample_actions().

    FAFM v3
    --------
    ``batch.actions`` are velocity sequences (B, K, d).
    Model outputs velocities; integrate both GT and pred to positions,
    then compare deltas.
    """
    eval_params = state.ema_params if state.ema_params is not None else state.params
    model = nnx.merge(state.model_def, eval_params)
    model.eval()
    observation, actions = batch
    infer_rng, _ = jax.random.split(rng)

    if isinstance(config.model, _fafm_config.FAFMConfig):
        # ── FAFM v3 ─────────────────────────────────────────────────────────
        from openpi.models.fafm import integrate_velocities

        fafm_cfg = config.model
        d_action = fafm_cfg.action_dim  # Full action dimension (7 for LIBERO)
        T = fafm_cfg.chunk_duration
        B, K, _ = actions.shape
        dt = T / K

        # Extract initial action state from observation
        # observation.state: (B, state_dim) where first d_action dims are action state
        # For LIBERO: state_dim=8, action_dim=7
        # actions (velocities): (B, K, 7) = full action velocities
        A0 = observation.state[:, :d_action]  # (B, d_action) - initial action state

        # GT: integrate velocities to positions, then compute deltas
        vel_gt = jax.lax.stop_gradient(actions)  # (B, K, d_action)
        pos_gt = integrate_velocities(
            vel_gt,
            A0,  # Initial action state (d_action dims)
            dt,
        )  # (B, K+1, d_action)
        
        # Compute adjacent deltas (step-by-step displacement)
        # This matches Pi0's evaluation metric (actions are deltas)
        deltas_gt = pos_gt[:, 1:, :] - pos_gt[:, :-1, :]  # (B, K, d_action)

        # Pred: sample velocities, integrate, compute deltas
        vel_pred = jax.lax.stop_gradient(
            model.sample_actions(infer_rng, observation, num_steps=10)
        )  # (B, K, d_action)
        pos_pred = integrate_velocities(
            vel_pred,
            A0,  # Same initial action state
            dt,
        )  # (B, K+1, d_action)
        
        # Compute adjacent deltas (step-by-step displacement)
        deltas_pred = pos_pred[:, 1:, :] - pos_pred[:, :-1, :]  # (B, K, d_action)

        # FAFM v3: adaptive threshold scales with dt so the metric stays
        # meaningful regardless of chunk_duration.
        # delta = vel * dt, so threshold = 5 * dt corresponds to ~5x the
        # typical per-step displacement at unit velocity (same semantics as
        # the original hardcoded 0.1 when T=1.0, dt=0.02 → 5*0.02=0.1).
        delta_clip_threshold = 5.0 * dt
        return _compute_action_metrics_jax(
            deltas_pred, deltas_gt,
            delta_clip_threshold=delta_clip_threshold,
            velocities_pred=vel_pred,
            velocities_gt=vel_gt,
        )

    else:
        # ── Pi0 / Pi0-FAST / any direct-action model ─────────────────────────
        deltas_gt = jax.lax.stop_gradient(actions)
        deltas_pred = jax.lax.stop_gradient(
            model.sample_actions(infer_rng, observation, num_steps=10)
        )
        H = deltas_gt.shape[1]
        deltas_pred = deltas_pred[:, :H, : deltas_gt.shape[2]]

        return _compute_action_metrics_jax(deltas_pred, deltas_gt)


def main(config: _config.TrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")
    
    # Log FAFM v3 specific info
    if isinstance(config.model, _fafm_config.FAFMConfig):
        logging.info("=" * 70)
        logging.info("FAFM v3 Configuration:")
        logging.info(f"  Action dimension: {config.model.action_dim}")
        logging.info(f"  State dimension: {config.model.state_dim}")
        logging.info(f"  Action horizon (K): {config.model.action_horizon}")
        logging.info(f"  Chunk duration (T): {config.model.chunk_duration}s")
        logging.info(f"  Control frequency: {config.model.action_horizon / config.model.chunk_duration:.1f} Hz")
        logging.info(f"  Alpha_vel: {config.model.alpha_vel}")
        logging.info(f"  Time embed dim: {config.model.time_embed_dim}")
        logging.info(f"  LEITFAFM mode: {config.model.enable_leitfafm}")
        logging.info(f"  2B batch merge: {config.model.use_2b_batch_merge}")
        logging.info("=" * 70)

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update("jax_compilation_cache_dir", str(epath.Path("~/.cache/jax").expanduser()))

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_swanlab(config, resuming=resuming, enabled=config.swanlab_enabled)

    data_loader = _data_loader.create_data_loader(
        config,
        sharding=data_sharding,
        shuffle=True,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    logging.info(f"Initialized data loader:\n{training_utils.array_tree_to_info(batch)}")

    # Log images from first batch to sanity check.
    images_to_log = [
        swanlab.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
        for i in range(min(5, len(next(iter(batch[0].images.values())))))
    ]
    swanlab.log({"camera_views": images_to_log}, step=0)

    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    jax.block_until_ready(train_state)
    logging.info(f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params)}")

    if resuming:
        train_state = _checkpoints.restore_state(checkpoint_manager, train_state, data_loader)

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    # JIT-compile action_eval_step with the same FSDP sharding as ptrain_step.
    # Without this, the non-JIT call triggers unoptimized FSDP all-gathers on
    # every operation, causing OOM on each GPU (~16 GiB allocation attempts).
    paction_eval_step = jax.jit(
        functools.partial(action_eval_step, config),
        in_shardings=(train_state_sharding, replicated_sharding, data_sharding),
        out_shardings=replicated_sharding,
    )

    start_step = int(train_state.step)
    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    # Action-space eval: runs every N steps for all model types (Pi0 and FAFM).
    # Set to 0 to disable.
    action_eval_interval = 1000

    infos = []
    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))
            
            # Loss components are already included in reduced_info from train_step
            # No need to manually extract them
            
            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            swanlab.log(reduced_info, step=step)
            infos = []

        # Action-space evaluation — JIT-compiled with FSDP sharding (same as ptrain_step).
        # Skip step 0: random FM weights cause ODE blow-up that poisons the chart scale.
        if action_eval_interval > 0 and step % action_eval_interval == 0 and step > 0:
            try:
                eval_rng, train_rng = jax.random.split(train_rng)
                with sharding.set_mesh(mesh):
                    raw_metrics = paction_eval_step(train_state, eval_rng, batch)
                # Convert JAX arrays to Python floats outside JIT.
                action_metrics = {k: float(v) for k, v in jax.device_get(raw_metrics).items()}
                
                # Format output based on model type
                if 'action/velocity_mae' in action_metrics:
                    # FAFM v3: show both position and velocity metrics
                    # Compute adaptive threshold for display (matches action_eval_step)
                    T = config.model.chunk_duration
                    K = config.model.action_horizon
                    dt = T / K
                    threshold = 5.0 * dt
                    pbar.write(
                        f"Step {step} [action eval] "
                        f"pos_mae={action_metrics['action/position_mae']:.4f}  "
                        f"vel_mae={action_metrics['action/velocity_mae']:.4f}  "
                        f"pos_corr={action_metrics['action/position_corr']:.3f}  "
                        f"vel_corr={action_metrics['action/velocity_corr']:.3f}  "
                        f"explode={action_metrics['action/explode_ratio']:.2%}"
                        f"  (explode = NaN/Inf OR |delta|>={threshold:.2f})"
                    )
                else:
                    # Pi0: show position metrics only (backward compatible)
                    pbar.write(
                        f"Step {step} [action eval] "
                        f"mae={action_metrics.get('action/position_mae', action_metrics.get('action/mae', 0)):.4f}  "
                        f"mse={action_metrics.get('action/position_mse', action_metrics.get('action/mse', 0)):.4f}  "
                        f"corr={action_metrics.get('action/position_corr', action_metrics.get('action/corr', 0)):.3f}  "
                        f"explode={action_metrics['action/explode_ratio']:.2%}"
                        f"  (explode = NaN/Inf OR |delta|>=1.0)"
                    )
                swanlab.log(action_metrics, step=step)
            except Exception as exc:
                logging.warning(f"action eval failed at step {step}: {exc}")

        batch = next(data_iter)

        if (step % config.save_interval == 0 and step > start_step) or step == config.num_train_steps - 1:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
