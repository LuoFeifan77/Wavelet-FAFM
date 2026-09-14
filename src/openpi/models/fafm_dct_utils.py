"""DCT-based Fourier coefficient utilities for FAFM.

This module provides tools for converting between velocity sequences and DCT coefficients,
enabling continuous-time velocity field representation via Fourier basis functions.

Key functions:
  - extract_dct_coeffs: Extract DCT-II coefficients from velocity sequences
  - decode_coeffs_to_velocity: Decode coefficients to velocities at arbitrary times
  - sample_freq_weighted_noise: Sample frequency-weighted Gaussian noise
  - estimate_M_from_dataset: Estimate optimal number of Fourier modes
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from typing import Any


def extract_dct_coeffs(
    v_star: jax.Array,
    M: int,
) -> jax.Array:
    """Extract DCT-II coefficients from velocity sequence.
    
    Converts a discrete velocity sequence to its frequency-domain representation
    using the Discrete Cosine Transform (Type II) with orthonormal normalization.
    
    This is typically called offline during data preprocessing and results are cached,
    or online during training (less efficient but simpler).
    
    IMPORTANT: This is a JAX-native implementation that works inside JIT-compiled functions.
    Uses manual DCT-II formula instead of scipy.fft.dct to avoid traced array conversion errors.
    
    Args:
        v_star: (B, K, d) or (K, d) velocity sequence
            - B: batch size (optional)
            - K: number of timesteps
            - d: action dimension (e.g., 7 for LIBERO)
        M: number of Fourier modes to retain (M < K)
            Typical values: 8-32 for 1-second chunks at 50Hz
    
    Returns:
        (B, M+1, d) or (M+1, d) DCT coefficients
        - Retains modes 0 through M (inclusive)
        - Mode 0 is the DC component (mean velocity)
        - Higher modes capture higher-frequency variations
    
    Notes:
        - Uses JAX-native DCT-II implementation for JIT compatibility
        - Orthonormal normalization: norm='ortho'
        - Energy preservation: ||coeffs||^2 ≈ ||v_star||^2
        - Invertible: decode(extract(v)) ≈ v (up to truncation at M)
    """
    # Handle both batched (B, K, d) and unbatched (K, d) inputs
    if v_star.ndim == 3:
        B, K, d = v_star.shape
        batched = True
    elif v_star.ndim == 2:
        K, d = v_star.shape
        batched = False
        v_star = v_star[None, :, :]  # Add batch dim
        B = 1
    else:
        raise ValueError(f"v_star must be 2D (K, d) or 3D (B, K, d), got shape {v_star.shape}")
    
    # DCT-II formula (orthonormal):
    # c_k = sqrt(2/K) * sum_j v_j * cos(pi*k*(j+0.5)/K)
    # with c_0 scaled by 1/sqrt(2)
    
    j = jnp.arange(K, dtype=jnp.float32)  # (K,)
    k = jnp.arange(M + 1, dtype=jnp.float32)  # (M+1,)
    
    # Cosine basis: cos(pi*k*(j+0.5)/K)
    # k: (M+1,), j: (K,) -> basis: (M+1, K)
    basis = jnp.cos(jnp.pi * k[:, None] * (j[None, :] + 0.5) / K)  # (M+1, K)
    
    # Normalization factors for orthonormal DCT-II
    norm = jnp.ones(M + 1, dtype=jnp.float32)
    norm = norm.at[0].set(1.0 / jnp.sqrt(2.0))  # DC component
    norm = norm * jnp.sqrt(2.0 / K)  # Overall scaling
    
    # Compute coefficients: c = norm * (basis @ v)
    # v_star: (B, K, d), basis: (M+1, K)
    # Result: (B, M+1, d)
    coeffs = jnp.einsum('mk,bkd->bmd', basis, v_star) * norm[:, None]  # (B, M+1, d)
    
    if not batched:
        coeffs = coeffs[0]  # Remove batch dim: (M+1, d)
    
    return coeffs


def decode_coeffs_to_velocity(
    coeffs: jax.Array,
    tau: jax.Array,
    T: float,
    K_original: int,
) -> jax.Array:
    """Decode DCT coefficients to velocities at arbitrary physical times.
    
    Reconstructs velocity values at specified time points using the continuous-domain
    Fourier basis functions. This provides C^∞ smooth interpolation between training points.
    
    Args:
        coeffs: (M+1, d) or (B, M+1, d) DCT coefficients
        tau: (N,) or (B, N) physical times in seconds
            - Can query at arbitrary times, not limited to training grid
            - Typical: uniform grid for control, but any τ ∈ [0, T] is valid
        T: chunk duration in seconds (e.g., 1.0)
        K_original: original sequence length used for normalization
            - Should match the K used in extract_dct_coeffs
            - Ensures correct amplitude scaling
    
    Returns:
        (N, d) or (B, N, d) decoded velocities at query times
    
    Notes:
        - Basis functions: φₖ(τ) = norm_k * cos(kπτ/T)
        - norm_k accounts for DCT-II orthonormal convention
        - Continuous in τ: can evaluate at any time, not just training points
        - Smooth: C^∞ due to cosine basis (all derivatives exist)
    """
    # Handle batched and unbatched inputs
    if coeffs.ndim == 3:
        B, M_plus_1, d = coeffs.shape
        batched = True
    else:
        M_plus_1, d = coeffs.shape
        batched = False
        coeffs = coeffs[None, :, :]  # Add batch dim
        B = 1
    
    M = M_plus_1 - 1
    
    # Frequency indices
    k = jnp.arange(0, M + 1, dtype=jnp.float32)  # (M+1,)
    
    # DCT-II orthonormal normalization factors
    # Mode 0 has factor 1/√2, others have 1
    norm = jnp.ones(M + 1, dtype=jnp.float32)
    norm = norm.at[0].set(1.0 / jnp.sqrt(2.0))
    # Overall scaling for DCT-II: √(2/K)
    norm = norm * jnp.sqrt(2.0 / K_original)  # (M+1,)
    
    # Handle tau dimensions
    if tau.ndim == 1:
        # (N,) -> (1, N) for broadcasting
        tau_query = tau[None, :]  # (1, N)
        N = tau.shape[0]
    elif tau.ndim == 2:
        # (B, N)
        tau_query = tau
        N = tau.shape[1]
    else:
        raise ValueError(f"tau must be 1D (N,) or 2D (B, N), got shape {tau.shape}")
    
    # Compute basis functions for IDCT-II
    # For orthonormal DCT-II, the inverse uses the same basis with normalization
    # IDCT-II formula: v[j] = Σₖ norm_k * c_k * cos(πk(j+0.5)/K)
    # 
    # For continuous-time decoding at arbitrary τ:
    # We map j → τ via: j = τ * K / T
    # So: cos(πk(j+0.5)/K) → cos(πk(τ*K/T + 0.5)/K) = cos(πkτ/T + πk/(2K))
    
    # Compute j indices corresponding to tau
    j_indices = tau_query * K_original / T  # (B, N)
    
    # Basis: cos(πk(j+0.5)/K_original)
    # k: (M+1,), j_indices: (B, N) -> basis: (B, M+1, N)
    basis = jnp.cos(jnp.pi * k[None, :, None] * (j_indices[:, None, :] + 0.5) / K_original)
    
    # Apply IDCT normalization
    # Mode 0 has factor 1/√2, others have 1
    norm_idct = jnp.ones(M + 1, dtype=jnp.float32)
    norm_idct = norm_idct.at[0].set(1.0 / jnp.sqrt(2.0))
    norm_idct = norm_idct * jnp.sqrt(2.0 / K_original)  # (M+1,)
    
    # Decode: v(τ) = Σₖ norm_k * cₖ * basis_k(τ)
    # coeffs: (B, M+1, d), basis: (B, M+1, N), norm_idct: (M+1,)
    v_decoded = jnp.einsum('bkd,bkn,k->bnd', coeffs, basis, norm_idct)
    
    if not batched:
        v_decoded = v_decoded[0]  # Remove batch dim: (N, d)
    
    return v_decoded


def sample_freq_weighted_noise(
    M: int,
    d: int,
    T: float,
    B: int,
    sigma: float = 1.0,
    key: jax.Array | None = None,
) -> jax.Array:
    """Sample frequency-weighted Gaussian noise for DCT coefficients.
    
    Generates noise with variance inversely proportional to frequency squared,
    matching the H¹ regularity of physical velocity functions. This provides
    a more physically realistic noise prior than standard Gaussian.
    
    Physical motivation:
        - Velocity trajectories are naturally low-frequency dominant
        - High-frequency oscillations require more energy (H¹ norm)
        - Noise should reflect this: low frequencies have higher variance
    
    Args:
        M: number of Fourier modes (excluding DC)
        d: action dimension
        T: chunk duration in seconds
        B: batch size
        sigma: overall noise scale (default: 1.0)
        key: JAX random key (if None, uses default RNG)
    
    Returns:
        (B, M+1, d) frequency-weighted noise samples
        - Mode k has std = sigma / sqrt(1 + (kπ/T)²)
        - Mode 0 (DC) has highest variance
        - High modes have progressively lower variance
    
    Notes:
        - Matches H¹ Sobolev space norm: ||f||²_H¹ = Σₖ (1 + ωₖ²)|cₖ|²
        - Ensures FM interpolation path stays in physically plausible space
        - Can fall back to standard Gaussian by setting sigma very large
    """
    if key is None:
        key = jax.random.PRNGKey(0)
    
    # Frequency indices
    k_idx = jnp.arange(0, M + 1, dtype=jnp.float32)  # (M+1,)
    
    # Angular frequencies: ωₖ = kπ/T
    omega_k = k_idx * jnp.pi / T  # (M+1,)
    
    # Frequency-dependent standard deviation: σ / √(1 + ωₖ²)
    std_k = sigma / jnp.sqrt(1.0 + omega_k ** 2)  # (M+1,)
    
    # Sample standard Gaussian
    xi = jax.random.normal(key, (B, M + 1, d))  # (B, M+1, d)
    
    # Scale by frequency-dependent std
    noise = xi * std_k[None, :, None]  # (B, M+1, d)
    
    return noise


def estimate_M_from_dataset(
    dataset: Any,
    energy_threshold: float = 0.95,
) -> int:
    """Estimate optimal number of Fourier modes from dataset.
    
    Analyzes velocity sequences in the dataset to determine the minimum M
    that captures `energy_threshold` fraction of the signal energy.
    
    This should be run once before training to set the `fourier_modes` config.
    
    NOTE: This is an offline analysis tool that uses numpy/scipy for efficiency.
    It is NOT called during training, so numpy usage is acceptable here.
    
    Args:
        dataset: iterable of velocity sequences
            - Each element should be (K, d) array
            - Or dict with 'actions' key containing velocities
        energy_threshold: fraction of energy to preserve (default: 0.95)
            - 0.95 means retain 95% of signal energy
            - Higher values need more modes (better accuracy, more params)
            - Lower values need fewer modes (faster, more compression)
    
    Returns:
        M: recommended number of Fourier modes
        - Typical values: 8-16 for smooth trajectories
        - 20-32 for trajectories with rapid direction changes
    
    Usage:
        >>> from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        >>> dataset = LeRobotDataset("your_dataset")
        >>> M = estimate_M_from_dataset(dataset, energy_threshold=0.95)
        >>> print(f"Recommended fourier_modes: {M}")
    
    Notes:
        - Computes cumulative energy spectrum for each trajectory
        - Averages across dataset to get robust estimate
        - Conservative: rounds up to ensure threshold is met
    """
    cumulative_energies = []
    
    for item in dataset:
        # Extract velocity sequence
        if isinstance(item, dict):
            if 'actions' in item:
                v_star = np.asarray(item['actions'])
            else:
                raise ValueError("Dataset items must have 'actions' key")
        else:
            v_star = np.asarray(item)
        
        # Compute full DCT
        if v_star.ndim == 2:
            K, d = v_star.shape
            v_t = np.transpose(v_star, (1, 0))  # (d, K)
            coeffs = np.fft.fft.dct(v_t, axis=-1, norm='ortho')  # (d, K)
        else:
            raise ValueError(f"Expected 2D velocity array, got shape {v_star.shape}")
        
        # Compute energy per mode: sum over action dimensions
        energy = (coeffs ** 2).sum(axis=0)  # (K,)
        
        # Cumulative energy fraction
        cumsum = np.cumsum(energy) / energy.sum()
        cumulative_energies.append(cumsum)
    
    # Average cumulative energy across dataset
    mean_cumsum = np.stack(cumulative_energies).mean(axis=0)
    
    # Find first M where cumulative energy exceeds threshold
    M = int(np.argmax(mean_cumsum >= energy_threshold))
    
    # Ensure at least M=1 (DC + one harmonic)
    M = max(1, M)
    
    return M


# ============================================================================
# Validation utilities (for testing)
# ============================================================================

def validate_dct_invertibility(
    v_star: jax.Array,
    M: int,
    T: float,
    tol: float = 0.1,
) -> tuple[bool, float]:
    """Validate that DCT extraction and decoding are approximately invertible.
    
    Tests: decode(extract(v)) ≈ v
    
    NOTE: This uses JAX-native DCT, so reconstruction won't be perfect for M < K.
    The tolerance is set higher (0.1) to account for truncation error.
    
    Args:
        v_star: (K, d) velocity sequence
        M: number of Fourier modes
        T: chunk duration
        tol: relative error tolerance (default: 0.1 for truncated DCT)
    
    Returns:
        (is_valid, relative_error)
    """
    K, d = v_star.shape
    
    # Extract coefficients (JAX-native)
    coeffs = extract_dct_coeffs(v_star, M)  # (M+1, d)
    
    # Decode back to original times
    tau = jnp.linspace(0, T, K, endpoint=False)
    v_reconstructed = decode_coeffs_to_velocity(coeffs, tau, T, K)  # (K, d)
    
    # Compute relative error
    error = jnp.linalg.norm(v_reconstructed - v_star)
    norm = jnp.linalg.norm(v_star)
    relative_error = float(error / (norm + 1e-8))
    
    is_valid = relative_error < tol
    
    return is_valid, relative_error


def validate_energy_preservation(
    v_star: jax.Array,
    M: int,
    tol: float = 0.1,
) -> tuple[bool, float]:
    """Validate that DCT preserves energy (Parseval's theorem).
    
    Tests: ||coeffs||² ≈ ||v_star||²
    
    Args:
        v_star: (K, d) velocity sequence
        M: number of Fourier modes
        tol: relative error tolerance
    
    Returns:
        (is_valid, relative_error)
    """
    # Extract coefficients
    coeffs = extract_dct_coeffs(v_star, M)  # (M+1, d)
    
    # Compute energies
    energy_coeffs = float(jnp.sum(coeffs ** 2))
    energy_velocity = float(jnp.sum(v_star ** 2))
    
    # Relative difference
    relative_error = abs(energy_coeffs - energy_velocity) / (energy_velocity + 1e-8)
    
    is_valid = relative_error < tol
    
    return is_valid, relative_error
