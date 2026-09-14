"""FAFM: Modulated ODE Trajectory with Integrated Flow.

FAFM v3: Direct velocity field learning via Functional Flow Matching.

Three minimal changes to Pi0:
  1. Physical time encoding: tau_k (seconds) replaces step index k
  2. State query input: A_curr added as runtime input at t=0
  3. Velocity supervision: direct MSE on ground-truth EEF velocities

Single-stage training: L = L_FM + alpha_vel * L_vel
  - L_FM operates on velocity sequences (not positions)
  - L_vel supervises velocity field queries at (A, tau) pairs

No separate ODE network, no modulation vectors, no two-stage training.
The action expert itself IS the velocity field.
"""

from __future__ import annotations

import logging

import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import fafm_config as _fafm_config
from openpi.models import pi0
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


# ---------------------------------------------------------------------------
# Velocity utility functions
# ---------------------------------------------------------------------------

def compute_velocities_from_positions(
    positions: jax.Array,
    dt: float,
) -> jax.Array:
    """Compute velocities from position sequence via finite difference.
    
    Args:
        positions: (..., T, d) position sequence
        dt: time step in seconds
    
    Returns:
        (..., T, d) velocity sequence (last velocity repeated)
    """
    velocities = jnp.zeros_like(positions)
    velocities = velocities.at[..., :-1, :].set(
        (positions[..., 1:, :] - positions[..., :-1, :]) / dt
    )
    velocities = velocities.at[..., -1, :].set(velocities[..., -2, :])
    return velocities


def integrate_velocities(
    velocities: jax.Array,
    A0: jax.Array,
    dt: float,
) -> jax.Array:
    """Integrate velocity sequence to positions via Euler method.
    
    Uses jax.lax.scan for efficient JIT compilation (avoids unrolling K iterations).
    
    Args:
        velocities: (B, K, d) velocity sequence
        A0: (B, d) initial state
        dt: time step in seconds
    
    Returns:
        (B, K+1, d) position trajectory (includes initial state)
    """
    B, K, d = velocities.shape
    
    def step_fn(pos, vel):
        """Single integration step: pos_{k+1} = pos_k + vel_k * dt"""
        next_pos = pos + vel * dt
        return next_pos, next_pos
    
    # Scan over time dimension (K steps)
    # velocities: (B, K, d) -> transpose to (K, B, d) for scanning
    velocities_t = jnp.transpose(velocities, (1, 0, 2))  # (K, B, d)
    
    # Scan: carry is current position, output is next position
    _, positions_t = jax.lax.scan(step_fn, A0, velocities_t)  # positions_t: (K, B, d)
    
    # Transpose back and prepend initial state
    positions = jnp.transpose(positions_t, (1, 0, 2))  # (B, K, d)
    positions = jnp.concatenate([A0[:, None, :], positions], axis=1)  # (B, K+1, d)
    
    return positions


# ---------------------------------------------------------------------------
# Physical time encoding (Change 1)
# ---------------------------------------------------------------------------

def physical_time_encode(tau: jax.Array, d_model: int) -> jax.Array:
    """Sinusoidal embedding of physical execution time in seconds.
    
    Frequency range: [1, 10000] Hz, covering sub-millisecond to second-level resolution.
    This matches the paper's formula: omega_i = exp(i * log(10^4) / m).
    
    Args:
        tau: (B, K) physical times in seconds
        d_model: embedding dimension (must be even)
    
    Returns:
        (B, K, d_model) time embeddings
    """
    if tau.ndim == 3:
        tau = tau.squeeze(-1)
    
    half = d_model // 2
    # Correct: frequencies from 1 to 10000 (positive exponent)
    freqs = jnp.exp(jnp.arange(half, dtype=jnp.float32) * (jnp.log(10000.0) / half))
    args = tau[..., None] * freqs[None, None, :]
    emb = jnp.concatenate([jnp.sin(args), jnp.cos(args)], axis=-1)
    return emb


def make_physical_times(K: int, T: float) -> jax.Array:
    """Generate K uniformly spaced physical times: tau_k = k*T/K."""
    return jnp.arange(K, dtype=jnp.float32) * T / K


class FAFMTimeEmbedding(nnx.Module):
    """Physical time embedding for FAFM."""
    
    def __init__(self, d_model: int, *, rngs: nnx.Rngs):
        self.d_model = d_model
        self.proj = nnx.Linear(d_model, d_model, rngs=rngs)
    
    def __call__(self, tau: jax.Array) -> jax.Array:
        """tau: (B, K) seconds -> (B, K, d_model) embeddings"""
        raw = physical_time_encode(tau, self.d_model)
        return self.proj(raw)


# ---------------------------------------------------------------------------
# FAFM Model
# ---------------------------------------------------------------------------

class FAFM(pi0.Pi0):
    """FAFM: Velocity field learning via Functional Flow Matching.
    
    Inherits from Pi0 and adds:
      1. Physical time embedding (replaces step-index positional encoding)
      2. State query projection (for closed-loop execution)
      3. Modified loss and inference to work with velocities
    """

    def __init__(self, config: _fafm_config.FAFMConfig, rngs: nnx.Rngs):
        super().__init__(config, rngs=rngs)
        self._fafm_config = config
        
        # Get action expert config
        from openpi.models import gemma as _gemma
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        
        # Physical time embedding (Change 1)
        self.action_time_embed = FAFMTimeEmbedding(
            d_model=action_expert_config.width,
            rngs=rngs,
        )
        
        # State query projection (Change 2)
        self.state_query_proj = nnx.Linear(
            config.action_dim,
            action_expert_config.width,
            rngs=rngs,
        )
        # Learned mask token for denoising phase
        self.state_mask_token = nnx.Param(
            jnp.zeros(action_expert_config.width, dtype=jnp.float32)
        )
        
        # Override state_proj to accept state_dim (not action_dim)
        # LIBERO state is 8-dim (7 joints + 1 gripper), but action_dim is 7
        if not config.pi05:
            self.state_proj = nnx.Linear(
                config.state_dim,
                action_expert_config.width,
                rngs=rngs,
            )
        
        # Override action_in_proj and action_out_proj for DCT coefficient space
        # Operates on DCT coefficients (M+1 tokens)
        # Input: DCT coefficients (M+1, d)
        # Each token is d-dimensional
        self.action_in_proj = nnx.Linear(
            config.action_dim,
            action_expert_config.width,
            rngs=rngs,
        )
        # Output: DCT coefficients (M+1, d)
        # Each token outputs d dimensions
        self.action_out_proj = nnx.Linear(
            action_expert_config.width,
            config.action_dim,
            rngs=rngs,
        )
        logger.info(
            f"FAFM: Using DCT coefficient space with M={config.fourier_modes}, "
            f"N_tokens={config.fourier_modes + 1}, output_dim_per_token={config.action_dim}"
        )
    
    def embed_action_suffix_fafm(
        self,
        observation: _model.Observation,
        noisy_velocities: jax.Array,
        tau: jax.Array,
        t_diffusion: jax.Array,
        A_query: jax.Array | None = None,
    ):
        """Embed action tokens with physical time and state query.
        
        Args:
            observation: observation
            noisy_velocities: (B, N, d) action sequence
                - N=M+1 (DCT coefficient tokens), d=action_dim
            tau: (B, N) physical times in seconds
                - N=M+1 frequency indices (conceptually)
            t_diffusion: (B,) FM denoising time
            A_query: (B, d) or (B, N, d) robot state (d=action_dim)
                - (B, d): single state, broadcast to all tokens (FAFM mode)
                - (B, N, d): per-token state (LEITFAFM mode)
                - None: use mask token during denoising
        
        Returns:
            tokens, input_mask, ar_mask, adarms_cond
        """
        input_mask = []
        ar_mask = []
        tokens = []
        
        # Add state token (Pi0 convention)
        if not self.pi05:
            state_token = self.state_proj(observation.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((observation.state.shape[0], 1), dtype=jnp.bool_))
            ar_mask += [True]
        
        # Project velocities to hidden dim
        action_tokens = self.action_in_proj(noisy_velocities)  # (B, K, hidden)
        
        # Add physical time embedding (Change 1)
        time_emb = self.action_time_embed(tau)  # (B, K, hidden)
        action_tokens = action_tokens + time_emb
        
        # Add state query embedding (Change 2: LEITFAFM per-token state injection)
        B, K = action_tokens.shape[:2]
        
        # When pi05=True and suppress_pi05_state_query=True, skip state_query_proj entirely:
        # state is already carried as discrete language tokens in the prefix, so we only
        # apply the mask token to keep the action token shape consistent.
        if self.pi05 and self._fafm_config.suppress_pi05_state_query:
            action_tokens = action_tokens + self.state_mask_token.value[None, None, :]
        else:
            # Determine whether to use state based on config
            use_state = False
            if A_query is not None:
                if self._fafm_config.mask_state_during_training:
                    # FAFM default: mask state during denoising (t>0), use at execution (t=0)
                    is_exec = (t_diffusion == 0.0).astype(jnp.float32)  # (B,)
                    use_state = True
                else:
                    # Pi0-like: always use state (no masking)
                    is_exec = jnp.ones_like(t_diffusion)  # (B,) all ones
                    use_state = True
            
            if use_state:
                # Support both (B, d) and (B, K, d) shapes
                if A_query.ndim == 2:
                    # (B, d) → broadcast to all K tokens (FAFM mode)
                    state_emb = self.state_query_proj(A_query)  # (B, hidden)
                    state_tok = (
                        is_exec[:, None] * state_emb
                        + (1.0 - is_exec[:, None]) * self.state_mask_token.value[None, :]
                    )
                    action_tokens = action_tokens + state_tok[:, None, :]  # broadcast
                elif A_query.ndim == 3:
                    # (B, K, d) → per-token injection (LEITFAFM mode)
                    B_q, K_q, d_q = A_query.shape
                    A_flat = A_query.reshape(B_q * K_q, d_q)
                    state_emb_flat = self.state_query_proj(A_flat)  # (B*K, hidden)
                    state_emb = state_emb_flat.reshape(B_q, K_q, -1)  # (B, K, hidden)
                    state_tok = (
                        is_exec[:, None, None] * state_emb
                        + (1.0 - is_exec[:, None, None]) * self.state_mask_token.value[None, None, :]
                    )
                    action_tokens = action_tokens + state_tok  # per-token, no broadcast
                else:
                    raise ValueError(f"A_query must be (B, d) or (B, K, d), got shape {A_query.shape}")
            else:
                # No state query provided, use mask token
                action_tokens = action_tokens + self.state_mask_token.value[None, None, :]
        
        # Add FM timestep embedding
        time_emb_fm = pi0.posemb_sincos(
            t_diffusion, self.action_in_proj.out_features,
            min_period=4e-3, max_period=4.0,
        )
        
        if self.pi05:
            time_emb_fm = self.time_mlp_in(time_emb_fm)
            time_emb_fm = nnx.swish(time_emb_fm)
            time_emb_fm = self.time_mlp_out(time_emb_fm)
            time_emb_fm = nnx.swish(time_emb_fm)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb_fm
        else:
            time_tokens = einops.repeat(time_emb_fm, "b emb -> b k emb", k=K)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        ar_mask += [True] + ([False] * (K - 1))
        
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        
        return tokens, input_mask, ar_mask, adarms_cond
    
    @override
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
    ) -> tuple[at.Float[at.Array, "*b ah"], dict[str, at.Array]]:
        """FAFM training loss: L_FM + alpha_vel * L_vel.
        
        FM on DCT coefficients ĉ ∈ ℝ^((M+1)×d) with Fourier-decoded L_vel.
        
        Args:
            actions: (B, K, d) ground-truth velocity sequence
                     d = action_dim (e.g., 7 for LIBERO: 6D EEF + 1D gripper)
        
        Returns:
            Tuple of:
            - (B, K) per-token loss (for compatibility with Pi0 interface)
            - dict of metrics for logging (e.g., loss_fm, loss_vel)
        """
        return self._compute_loss(rng, observation, actions, train=train)

    def _compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        actions: _model.Actions,
        *,
        train: bool = False,
    ) -> tuple[at.Float[at.Array, "*b ah"], dict[str, at.Array]]:
        """FAFM training loss: FM on DCT coefficients + Fourier-decoded L_vel.
        
        Training process:
          - FM operates on DCT coefficient space ℝ^((M+1)×d)
          - Noise is frequency-weighted Gaussian (low-frequency dominant)
          - L_vel compares Fourier-decoded velocities at K training points
          - Output is (B, K) for interface compatibility, but computed from M+1 coefficients
        """
        from openpi.models import fafm_dct_utils
        
        preprocess_rng, noise_rng, time_rng, step_rng = jax.random.split(rng, 4)
        observation = _model.preprocess_observation(preprocess_rng, observation, train=train)
        
        B, K, d = actions.shape
        expected_d = self._fafm_config.action_dim
        assert d == expected_d, (
            f"Expected action_dim={expected_d}, got {d}. "
            f"For LIBERO: action_dim=7 (6D EEF + 1D gripper)"
        )
        
        T = self._fafm_config.chunk_duration
        M = self._fafm_config.fourier_modes
        sigma = self._fafm_config.sigma_noise
        
        # Physical times for K training points
        tau = make_physical_times(K, T)
        tau_batch = jnp.broadcast_to(tau[None, :], (B, K))
        
        # --- Extract DCT coefficients from velocity sequences ---
        v_clean = actions  # (B, K, d)
        c_star = fafm_dct_utils.extract_dct_coeffs(v_clean, M)  # (B, M+1, d)
        
        # Physical times for M+1 coefficient tokens (conceptual, for embedding)
        # Note: tau represents frequency indices rather than physical time
        # But we still use physical time encoding for consistency
        tau_coeff = make_physical_times(M + 1, T)  # (M+1,)
        tau_coeff_batch = jnp.broadcast_to(tau_coeff[None, :], (B, M + 1))
        
        # --- L_FM: Flow Matching in coefficient space ---
        # Sample frequency-weighted noise
        c0 = fafm_dct_utils.sample_freq_weighted_noise(
            M, d, T, B, sigma, noise_rng
        )  # (B, M+1, d)
        
        # FM time sampling
        t_diffusion = jax.random.beta(time_rng, 1.5, 1, (B,)) * 0.999 + 0.001
        t_expanded = t_diffusion[:, None, None]
        
        # Linear interpolation in coefficient space
        c_t = (1.0 - t_expanded) * c_star + t_expanded * c0  # (B, M+1, d)
        u_star = c0 - c_star  # Target vector field
        
        # --- L_vel: Velocity supervision at t=0 ---
        t_zero = jnp.zeros(B, dtype=jnp.float32)
        
        # --- Forward pass: 2B batch merge to save memory ---
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        
        # Determine whether to pass state during training
        d_action = self._fafm_config.action_dim
        
        if self._fafm_config.mask_state_during_training:
            # FAFM default: mask state during FM denoising (t>0), use at vel query (t=0)
            # L_FM: t>0, no state (will be masked by embed_action_suffix_fafm)
            A_query_fm = None
            # L_vel: t=0, provide state (will be used by embed_action_suffix_fafm because t=0)
            A_query_vel = observation.state[:, :d_action]  # (B, d)
        else:
            # Pi0-like: always use state (both FM and vel)
            A_query_fm = observation.state[:, :d_action]  # (B, d)
            A_query_vel = observation.state[:, :d_action]  # (B, d)
        
        # Prepare FM suffix (noisy coefficients at t_diffusion)
        suffix_tokens_fm, suffix_mask_fm, suffix_ar_mask_fm, adarms_cond_fm = \
            self.embed_action_suffix_fafm(
                observation, c_t, tau_coeff_batch, t_diffusion, A_query=A_query_fm
            )
        
        # Prepare vel suffix (clean coefficients at t=0)
        suffix_tokens_vel, suffix_mask_vel, suffix_ar_mask_vel, adarms_cond_vel = \
            self.embed_action_suffix_fafm(
                observation, c_star, tau_coeff_batch, t_zero, A_query=A_query_vel
            )
        
        # Merge along batch dimension: (B, ...) → (2B, ...)
        prefix_tokens_2B = jnp.concatenate([prefix_tokens, prefix_tokens], axis=0)
        prefix_mask_2B = jnp.concatenate([prefix_mask, prefix_mask], axis=0)
        suffix_tokens_2B = jnp.concatenate([suffix_tokens_fm, suffix_tokens_vel], axis=0)
        suffix_mask_2B = jnp.concatenate([suffix_mask_fm, suffix_mask_vel], axis=0)
        
        if self.pi05:
            adarms_cond_2B = jnp.concatenate([adarms_cond_fm, adarms_cond_vel], axis=0)
        else:
            adarms_cond_2B = None
        
        # One forward pass with 2B batch
        input_mask_2B = jnp.concatenate([prefix_mask_2B, suffix_mask_2B], axis=1)
        ar_mask_2B = jnp.concatenate([prefix_ar_mask, suffix_ar_mask_fm], axis=0)
        attn_mask_2B = pi0.make_attn_mask(input_mask_2B, ar_mask_2B)
        positions_2B = jnp.cumsum(input_mask_2B, axis=1) - 1
        
        (_, suffix_out_2B), _ = self.PaliGemma.llm(
            [prefix_tokens_2B, suffix_tokens_2B],
            mask=attn_mask_2B,
            positions=positions_2B,
            adarms_cond=[None, adarms_cond_2B],
        )
        
        # Split outputs: (2B, M+1, hidden) → (B, M+1, hidden) each
        N_tokens = M + 1
        suffix_out_fm = suffix_out_2B[:B, :, :]
        suffix_out_vel = suffix_out_2B[B:, :, :]
        
        # Output: predicted vector field (B, M+1, d)
        u_pred = self.action_out_proj(suffix_out_fm[:, -N_tokens:])  # (B, M+1, d)
        
        # L_FM: MSE in coefficient space
        loss_fm_coeff = jnp.mean(jnp.square(u_pred - u_star), axis=-1)  # (B, M+1)
        
        # Predicted clean coefficients (B, M+1, d)
        c_hat_0 = self.action_out_proj(suffix_out_vel[:, -N_tokens:])  # (B, M+1, d)
        
        # Decode coefficients to velocities at K training points
        # Use vmap to handle batch dimension
        v_decoded = jax.vmap(
            lambda c, taus: fafm_dct_utils.decode_coeffs_to_velocity(c, taus, T, K)
        )(c_hat_0, tau_batch)  # (B, K, d)
        
        # L_vel: MSE between decoded and ground-truth velocities
        loss_vel_decoded = jnp.mean(jnp.square(v_decoded - v_clean), axis=-1)  # (B, K)
        
        # --- Combine losses ---
        alpha = self._fafm_config.alpha_vel
        
        # Broadcast L_FM from (B, M+1) to (B, K) for interface compatibility
        # Use mean over coefficient dimension
        loss_fm_mean = jnp.mean(loss_fm_coeff, axis=-1, keepdims=True)  # (B, 1)
        loss_fm_broadcast = jnp.broadcast_to(loss_fm_mean, (B, K))  # (B, K)
        
        total_loss = loss_fm_broadcast + alpha * loss_vel_decoded  # (B, K)
        
        # Return metrics for logging (JAX arrays, will be converted outside JIT)
        metrics = {
            'loss_fm': jnp.mean(loss_fm_coeff),  # Mean over batch and modes
            'loss_vel': jnp.mean(loss_vel_decoded),  # Mean over batch and time
        }
        
        return total_loss, metrics

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        """FAFM inference: FM denoising to obtain motion intention code.
        
        This implements Algorithm 1 lines 1-7 (denoising phase).
        For closed-loop execution with state queries (lines 8-17),
        use query_velocity() instead.
        
        Args:
            rng: random key
            observation: observation
            num_steps: number of FM denoising steps
            noise: optional initial noise
        
        Returns:
            (B, K, d) velocity sequence (motion intention code)
            d = action_dim (e.g., 7 for LIBERO: 6D EEF + 1D gripper)
        
        Notes:
            Denoises in coefficient space, decodes to velocities via Fourier
        """
        return self._sample_actions(rng, observation, num_steps=num_steps, noise=noise)
    
    def _sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        """FAFM inference: FM denoising in coefficient space + Fourier decoding.
        
        Inference process:
          - Denoises in DCT coefficient space ℝ^((M+1)×d)
          - Uses frequency-weighted noise initialization
          - Decodes coefficients to velocities via Fourier basis
          - Returns (B, K, d) velocities for compatibility
        """
        from openpi.models import fafm_dct_utils
        
        observation = _model.preprocess_observation(None, observation, train=False)
        
        B = observation.state.shape[0]
        K = self.action_horizon
        d = self.action_dim
        T = self._fafm_config.chunk_duration
        M = self._fafm_config.fourier_modes
        sigma = self._fafm_config.sigma_noise
        
        # Physical times for coefficient tokens
        tau_coeff = make_physical_times(M + 1, T)
        tau_coeff_batch = jnp.broadcast_to(tau_coeff[None, :], (B, M + 1))
        
        # Initialize noise in coefficient space (frequency-weighted)
        if noise is None:
            c_noisy = fafm_dct_utils.sample_freq_weighted_noise(
                M, d, T, B, sigma, rng
            )  # (B, M+1, d)
        else:
            # If noise provided, assume it's in velocity space and convert to coefficients
            # This is for compatibility with existing code that may pass velocity noise
            c_noisy = fafm_dct_utils.extract_dct_coeffs(noise, M)
        
        # FM denoising in coefficient space
        dt = -1.0 / num_steps
        c_t = c_noisy
        time = 1.0
        
        # Cache prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = pi0.make_attn_mask(prefix_mask, prefix_ar_mask)
        positions_prefix = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None],
            mask=prefix_attn_mask,
            positions=positions_prefix,
        )
        
        def step(carry):
            c_t, time = carry
            t_current = jnp.full((B,), time, dtype=jnp.float32)
            
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = \
                self.embed_action_suffix_fafm(
                    observation, c_t, tau_coeff_batch, t_current, A_query=None
                )
            
            suffix_attn_mask = pi0.make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask_suffix = einops.repeat(
                prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1]
            )
            full_attn_mask = jnp.concatenate(
                [prefix_attn_mask_suffix, suffix_attn_mask], axis=-1
            )
            positions_suffix = (
                jnp.sum(prefix_mask, axis=-1)[:, None]
                + jnp.cumsum(suffix_mask, axis=-1) - 1
            )
            
            (_, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions_suffix,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            
            # Flow in coefficient space (B, M+1, d)
            N_tokens = M + 1
            flow = self.action_out_proj(suffix_out[:, -N_tokens:])  # (B, M+1, d)
            return c_t + dt * flow, time + dt
        
        def cond(carry):
            _, time = carry
            return time >= -dt / 2
        
        # Denoise to get clean coefficients
        c_clean, _ = jax.lax.while_loop(cond, step, (c_t, time))  # (B, M+1, d)
        
        # Decode coefficients to velocities at K control points
        tau_query = make_physical_times(K, T)  # (K,)
        tau_query_batch = jnp.broadcast_to(tau_query[None, :], (B, K))
        
        v_clean = jax.vmap(
            lambda c, taus: fafm_dct_utils.decode_coeffs_to_velocity(c, taus, T, K)
        )(c_clean, tau_query_batch)  # (B, K, d)
        
        return v_clean
    
    def query_velocity(
        self,
        v_clean: jax.Array,
        observation: _model.Observation,
        A_query: jax.Array,
        tau_k: float | jax.Array,
    ) -> jax.Array:
        """Query velocity field at a single (A, tau) point.
        
        Core primitive for both open-loop and LEITFAFM execution.
        The difference between modes is WHERE A_query comes from:
          - FAFM open-loop:  A_query = Euler-integrated prediction
          - LEITFAFM:        A_query = robot.read_state() (EEF or joint)
        
        In both cases, A_query is a motion state (EEF pose or joint angles),
        which is the most reliable proprioceptive signal. The output is a
        velocity command that can be sent to the robot's controller:
          - Velocity control:   robot.send_velocity(v_k)
          - Position control:   robot.send_position(A + v_k * dt)
          - Impedance control:  convert via impedance law
        
        Args:
            v_clean: (B, K, d) motion intention code from sample_actions()
                     d = action_dim (e.g., 7 for LIBERO: 6D EEF + 1D gripper)
            observation: observation (for VLM prefix embedding)
            A_query: (B, d) current action state to query at
                Note: Even if model was trained with LEITFAFM (per-token state),
                inference uses single state broadcast, which is correct behavior
                for querying the velocity field at a specific (A, tau) point.
            tau_k: scalar or (B,) physical time in seconds
        
        Returns:
            (B, d) velocity at (A_query, tau_k)
            d = action_dim (e.g., 7 for LIBERO: 6D EEF + 1D gripper)
        """
        B, K, d = v_clean.shape
        T = self._fafm_config.chunk_duration
        M = self._fafm_config.fourier_modes
        
        # Convert velocity sequence to DCT coefficients for embedding
        # This ensures consistency with training where we embed coefficients
        from openpi.models import fafm_dct_utils
        
        # Extract DCT coefficients from velocity sequence
        c_clean = fafm_dct_utils.extract_dct_coeffs(v_clean, M)  # (B, M+1, d)
        
        # Use coefficient token times for embedding
        tau_coeff = make_physical_times(M + 1, T)  # (M+1,)
        tau_batch = jnp.broadcast_to(tau_coeff[None, :], (B, M + 1))
        action_tokens = c_clean
        
        t_zero = jnp.zeros(B, dtype=jnp.float32)
        
        # Embed and forward at t=0 (execution mode)
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = \
            self.embed_action_suffix_fafm(
                observation, action_tokens, tau_batch, t_zero, A_query=A_query
            )
        
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = pi0.make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        
        (_, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens],
            mask=attn_mask,
            positions=positions,
            adarms_cond=[None, adarms_cond],
        )
        
        # Output is DCT coefficients; decode to velocity at query time
        from openpi.models import fafm_dct_utils
        
        # Output: DCT coefficients (B, M+1, d)
        c_all = self.action_out_proj(suffix_out[:, -(M + 1):])  # (B, M+1, d)
        
        # Decode coefficient to velocity at query time tau_k
        # Use continuous DCT decoding for accurate interpolation
        if isinstance(tau_k, (int, float)):
            tau_k_arr = jnp.full((B,), tau_k, dtype=jnp.float32)
        else:
            tau_k_arr = tau_k  # (B,)
        
        # Decode coefficients to velocities at query times
        # Use vmap to handle batch dimension
        v_k = jax.vmap(
            lambda c, tau: fafm_dct_utils.decode_coeffs_to_velocity(
                c[None, :, :], tau[None], T, K
            )[0, :]  # Extract single query point
        )(c_all, tau_k_arr)  # (B, d)
        
        return v_k
    
    def build_prefix_cache(
        self,
        observation: _model.Observation,
    ) -> tuple[jax.Array, jax.Array]:
        """Build prefix KV cache for efficient closed-loop execution.
        
        This computes the VLM prefix embedding once and caches the KV states,
        which can be reused across multiple query_velocity calls in a control loop.
        This is critical for real-time performance (e.g., 50Hz control).
        
        Args:
            observation: observation with images and language
        
        Returns:
            kv_cache: cached key-value states from prefix forward
            prefix_mask: prefix attention mask for cache alignment
        
        Usage:
            # Once per chunk:
            kv_cache, prefix_mask = model.build_prefix_cache(observation)
            
            # In control loop (50Hz):
            for k in range(K):
                A_current = robot.read_state()
                v_k = model.query_velocity_with_cache(
                    v_clean, kv_cache, prefix_mask, A_current, tau_k
                )
                robot.send_velocity(v_k)
        """
        observation = _model.preprocess_observation(None, observation, train=False)
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = pi0.make_attn_mask(prefix_mask, prefix_ar_mask)
        positions_prefix = jnp.cumsum(prefix_mask, axis=1) - 1
        
        _, kv_cache = self.PaliGemma.llm(
            [prefix_tokens, None],
            mask=prefix_attn_mask,
            positions=positions_prefix,
        )
        
        return kv_cache, prefix_mask
    
    def query_velocity_with_cache(
        self,
        v_clean: jax.Array,
        kv_cache: jax.Array,
        prefix_mask: jax.Array,
        A_query: jax.Array,
        tau_k: float | jax.Array,
    ) -> jax.Array:
        """Query velocity field using cached prefix KV states.
        
        This is the efficient version of query_velocity for closed-loop execution.
        Prefix KV cache is computed once per chunk via build_prefix_cache(),
        then reused for all timesteps in the control loop.
        
        Args:
            v_clean: (B, K, d) motion intention code (d=action_dim)
            kv_cache: cached prefix KV states from build_prefix_cache()
            prefix_mask: prefix attention mask from build_prefix_cache()
            A_query: (B, d) current action state (d=action_dim)
            tau_k: scalar or (B,) physical time in seconds
        
        Returns:
            (B, d) velocity at (A_query, tau_k) (d=action_dim)
        """
        B, K, d = v_clean.shape
        T = self._fafm_config.chunk_duration
        
        tau = make_physical_times(K, T)
        tau_batch = jnp.broadcast_to(tau[None, :], (B, K))
        t_zero = jnp.zeros(B, dtype=jnp.float32)
        
        # Embed suffix only (prefix is cached)
        # Note: We need to reconstruct a dummy observation for embed_action_suffix_fafm
        # This is a limitation of the current API design
        # TODO: Refactor to separate observation-dependent and observation-independent parts
        raise NotImplementedError(
            "query_velocity_with_cache requires API refactoring to separate "
            "observation-dependent embedding from action suffix embedding. "
            "For now, use query_velocity() which recomputes prefix each time."
        )
