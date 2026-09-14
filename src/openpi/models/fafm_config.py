"""Configuration for the FAFM model.

FAFM: Direct velocity field learning via Functional Flow Matching.

Three minimal changes to Pi0:
  1. Physical time encoding: tau_k = k * T_xi / K (seconds) replaces step index k
  2. State query input: A_curr added as runtime input at t=0
  3. Velocity supervision: direct MSE on ground-truth EEF velocities

Single-stage training: L = L_FM + alpha_vel * L_vel
No separate ODE network, no modulation vectors, no two-stage training.

Position supervision:
  - Set supervise_position_in_lfm=True to supervise integrated positions in L_FM
  - L_FM will compare predicted positions with ground-truth trajectory
  - Original default: supervise velocities/coefficients (supervise_position_in_lfm=False)

Note: For LIBERO, the action space is 6D end-effector (EEF) control + 1D gripper,
not 7 joint angles. LIBERO uses OSC_POSE (Operational Space Control) which operates
in Cartesian/EEF space.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Literal

import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.shared import array_typing as at

if TYPE_CHECKING:
    from openpi.models.fafm import FAFM


@dataclasses.dataclass(frozen=True)
class FAFMConfig(pi0_config.Pi0Config):
    """FAFM (Frequency-Aware Flow Matching) model configuration.
    
    The action expert learns a velocity field v(A, tau; z) where:
      - A: robot state (queried at execution time)
      - tau: physical time in seconds (tau_k = k * T_xi / K, NOT normalized)
      - z: VLM embedding (images + language)
    
    Training method selection (training_method):
      - 'flow_matching': Linear interpolation path, predicts vector field u = noise - clean
      - 'diffusion': Karras/EDM style with σ-based noise schedule, predicts clean sample
    
    Training uses combined loss: L = L_FM + alpha_vel * L_vel
      - L_FM: Flow Matching on DCT coefficient space
      - L_vel: Direct velocity supervision at t=0 with (A, tau) queries
    
    FM/Diffusion on DCT coefficients ĉ ∈ ℝ^((M+1)×d) with Fourier decoding
    """

    # loss结构是这样的？

    # --- Training method control ---
    training_method: Literal['flow_matching', 'diffusion'] = 'flow_matching'  # Training method selection
    
    # --- FAFM hyperparameters ---
    alpha_vel: float = 1.0  # velocity supervision loss weight
    time_embed_dim: int = 64  # physical time embedding dimension
    chunk_duration: float = 5.0  # T_ξ in physical seconds. LIBERO: 50 steps @ 10Hz = 5.0s.
                                  # Used as the absolute time base for TD encoding: tau_k = k*T/K (seconds).
                                  # Must reflect true physical duration; do NOT normalize away.
    
    # --- Diffusion-specific parameters (only used when training_method='diffusion') ---
    sigma_min: float = 0.05  # Minimum noise level for diffusion
    sigma_max: float = 10.0   # Maximum noise level for diffusion
    sigma_data: float = 5.3   # Karras scaling parameter (data standard deviation)
    use_karras_scaling: bool = True  # Whether to use Karras preconditioning
    diffusion_rho: float = 7.0  # EDM noise schedule parameter (exponential interpolation)
    
    # --- Fourier/DCT parameters ---
    fourier_modes: int = 16  # M: number of DCT modes
    energy_threshold: float = 0.95  # Energy threshold for estimating M
    sigma_noise: float = 1.0  # Noise scaling factor for frequency-weighted Gaussian
    
    # --- Position supervision control ---
    supervise_position_in_lfm: bool = False  # Whether to supervise position (pose) instead of velocity in L_FM
                                             # True: L_FM supervises integrated position trajectories
                                             # False: L_FM supervises velocities (default, original FAFM)
    alpha_pos: float = 1.0  # Position supervision loss weight (only used when supervise_position_in_lfm=True)
    
    # --- State query control ---
    mask_state_during_training: bool = True  # Whether to mask state during FM denoising (t>0)
                                             # True: FAFM default (learn observation→action)
                                             # False: Pi0-like (learn observation+state→action)
    
    use_state_in_inference: bool = False     # Whether to use state query during inference
                                             # True: Use state at t=0 (closed-loop inference)
                                             # False: Mask state during inference (open-loop, default)
                                             # Note: Only effective when mask_state_during_training=True
    
    use_query_velocity_execution: bool = False  # Whether to use query_velocity for closed-loop execution
                                                # True: Generate chunk once, query at each step with current state
                                                # False: Use standard sample_actions (default)
                                                # Note: Only effective when use_state_in_inference=True
    
    # --- Action normalization control ---
    use_identity_action_norm: bool = True  # Whether to use identity normalization for actions during inference
                                           # True: Use identity transform (q01=-3, q99=3) - STABLE, no file dependency
                                           # False: Use actual norm_stats from training - requires correct norm_stats file
                                           # Changed default to True to avoid norm_stats shape mismatch issues
    
    # --- Pi0.5 state injection control ---
    # suppress_pi05_state_query: When pi05=True, the state is already injected into the prefix
    # as discrete language tokens. Setting this to True prevents FAFM's state_query_proj from
    # injecting the state a second time into action tokens, avoiding redundant state information.
    #   - True: Only use pi05 discrete prefix state (no state_query_proj in suffix)
    #   - False: Both pi05 discrete prefix state AND state_query_proj (double injection, default)
    # Has no effect when pi05=False.
    suppress_pi05_state_query: bool = False

    # --- LEITFAFM configuration ---
    # enable_leitfafm: Enable per-token state query and full L_vel supervision
    #   - True: LEITFAFM mode (per-token state injection, supervise all K tokens)
    #   - False: FAFM mode (no state query during training, simplified L_vel)
    enable_leitfafm: bool = False  # Default: LEITFAFM not enabled
    
    # --- Training optimization ---
    # use_2b_batch_merge: Merge L_FM and L_vel forward passes into single 2B batch
    #   - True: One LLM forward with 2B batch (faster, ~1.8x speedup)
    #   - False: Two separate LLM forwards (easier to debug)
    use_2b_batch_merge: bool = False  # Default: disabled for stability
    
    # --- Robot configuration ---
    # For LIBERO with OSC_POSE controller:
    #   - action_dim = 7 (6D EEF delta + 1D gripper)
    #   - state_dim = 8 (3D EEF pos + 3D EEF ori + 2D gripper state)
    #
    # Note: Although gripper is binary control, it's included in the action
    # dimension and learned as part of the velocity field.
    state_dim: int = 8      # Full state dimension
    action_dim: int = 7     # Full action dimension (6D EEF + 1D gripper)
    # action_horizon: number of velocity tokens K (e.g., 50 for 10Hz at 5.0s)
    action_horizon: int = 50
    # max_token_len: Pi0 base -> 48, Pi0.5 base -> 200
    max_token_len: int = None  # type: ignore

    def __post_init__(self):
        super().__post_init__()
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.FAFM05 if self.pi05 else _model.ModelType.FAFM

    @override
    def create(self, rng: at.KeyArrayLike) -> "FAFM":
        from openpi.models.fafm import FAFM
        from flax import nnx
        return FAFM(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                # state dimension (8 for LIBERO: 3D EEF pos + 3D EEF ori + 2D gripper)
                state=jax.ShapeDtypeStruct([batch_size, self.state_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
                # state_trajectory is OPTIONAL (generated internally in compute_loss)
                # Kept in inputs_spec for backward compatibility with old data pipelines
                state_trajectory=jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.state_dim], jnp.float32),
            )
        # action = (B, K, d) -- velocity sequence
        action_spec = jax.ShapeDtypeStruct(
            [batch_size, self.action_horizon, self.action_dim], jnp.float32
        )
        return observation_spec, action_spec
