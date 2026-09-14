# FAFM: Frequency-Aware Flow Matching for Continuous Robot Control

**FAFM** (Frequency-Aware Flow Matching) is a method for learning continuous and temporally consistent robot actions through frequency-domain flow matching. This repository contains the implementation based on the OpenPI framework.

## Overview

FAFM introduces a frequency-aware approach to robot action generation:
- **DCT Coefficient Space**: Transforms discrete action sequences into frequency domain using Discrete Cosine Transform
- **Continuous Actions**: Enables query at arbitrary temporal resolutions via cosine basis expansion
- **Temporal Consistency**: Regularizes first-order derivatives to promote smooth, consistent actions
- **Mixed-Frequency Training**: Handles demonstrations collected at heterogeneous control frequencies

## Key Features

✅ **Frequency-Domain Flow Matching**: Performs flow matching in DCT coefficient space
✅ **H¹ Sobolev Regularization**: Frequency-weighted noise and derivative supervision
✅ **Continuous-Time Representation**: C^∞ smooth action trajectories
✅ **Mixed-Frequency Support**: Unified training from heterogeneous-fps demonstrations
✅ **Parameter-Free**: No additional networks or hyperparameters beyond standard flow matching

## Demonstrations

### Real-World Pick-and-Place

<div style="display: flex; gap: 0; align-items: flex-start;">
  <figure style="flex: 1; margin: 0; text-align: center;">
    <img src="video/baseline/task1_pi05_action_chunk.gif" width="100%" alt="pi_0.5 with action chunking on real-world pick-and-place">
    <figcaption><sub><strong>$\pi_{0.5}$ (Action Chunk)</strong></sub></figcaption>
  </figure>
  <figure style="flex: 1; margin: 0; text-align: center;">
    <img src="video/ours/task1_pi05_fafm.gif" width="100%" alt="pi_0.5 with FAFM on real-world pick-and-place">
    <figcaption><sub><strong>$\pi_{0.5}$ (FAFM)</strong></sub></figcaption>
  </figure>
</div>

### Real-World Obstacle Avoidance

<div style="display: flex; gap: 0; align-items: flex-start;">
  <figure style="flex: 1; margin: 0; text-align: center;">
    <img src="video/baseline/task2_pi05_action_chunk.gif" width="100%" alt="pi_0.5 with action chunking on real-world obstacle avoidance">
    <figcaption><sub><strong>$\pi_{0.5}$ (Action Chunk)</strong></sub></figcaption>
  </figure>
  <figure style="flex: 1; margin: 0; text-align: center;">
    <img src="video/ours/task2_pi05_fafm.gif" width="100%" alt="pi_0.5 with FAFM on real-world obstacle avoidance">
    <figcaption><sub><strong>$\pi_{0.5}$ (FAFM)</strong></sub></figcaption>
  </figure>
</div>

### Multi-Modal Obstacle Avoidance in Simulation

<div style="display: flex; gap: 0; align-items: flex-start;">
  <figure style="flex: 1; margin: 0; text-align: center;">
    <img src="video/baseline/obstacle_avoidance_dp.gif" width="100%" alt="Diffusion Policy on multi-modal obstacle avoidance in simulation">
    <figcaption><sub><strong>Diffusion Policy</strong></sub></figcaption>
  </figure>
  <figure style="flex: 1; margin: 0; text-align: center;">
    <img src="video/ours/obstacle_avoidance_ours.gif" width="100%" alt="Ours on multi-modal obstacle avoidance in simulation">
    <figcaption><sub><strong>Ours</strong></sub></figcaption>
  </figure>
</div>

### LapGym: Rope Threading

<div style="display: flex; gap: 0; align-items: flex-start;">
  <figure style="flex: 1; margin: 0; text-align: center;">
    <img src="video/baseline/rope_threading_dp.gif" width="100%" alt="Diffusion Policy on LapGym rope threading">
    <figcaption><sub><strong>Diffusion Policy</strong></sub></figcaption>
  </figure>
  <figure style="flex: 1; margin: 0; text-align: center;">
    <img src="video/ours/rope_threading_ours.gif" width="100%" alt="Ours on LapGym rope threading">
    <figcaption><sub><strong>Ours</strong></sub></figcaption>
  </figure>
</div>

### LapGym: Grasp-Lift-Touch

<div style="display: flex; gap: 0; align-items: flex-start;">
  <figure style="flex: 1; margin: 0; text-align: center;">
    <img src="video/baseline/grasp_lift_touch_dp.gif" width="100%" alt="Diffusion Policy on LapGym grasp-lift-touch">
    <figcaption><sub><strong>Diffusion Policy</strong></sub></figcaption>
  </figure>
  <figure style="flex: 1; margin: 0; text-align: center;">
    <img src="video/ours/grasp_lift_touch_ours.gif" width="100%" alt="Ours on LapGym grasp-lift-touch">
    <figcaption><sub><strong>Ours</strong></sub></figcaption>
  </figure>
</div>

### LapGym: Bimanual Tissue Manipulation

<div style="display: flex; gap: 0; align-items: flex-start;">
  <figure style="flex: 1; margin: 0; text-align: center;">
    <img src="video/baseline/bimanual_tissue_manipulation_dp.gif" width="100%" alt="Diffusion Policy on LapGym bimanual tissue manipulation">
    <figcaption><sub><strong>Diffusion Policy</strong></sub></figcaption>
  </figure>
  <figure style="flex: 1; margin: 0; text-align: center;">
    <img src="video/ours/bimanual_tissue_manipulation_ours.gif" width="100%" alt="Ours on LapGym bimanual tissue manipulation">
    <figcaption><sub><strong>Ours</strong></sub></figcaption>
  </figure>
</div>

### LapGym: Ligating Loop

<div style="display: flex; gap: 0; align-items: flex-start;">
  <figure style="flex: 1; margin: 0; text-align: center;">
    <img src="video/baseline/ligating_loop_dp.gif" width="100%" alt="Diffusion Policy on LapGym ligating loop">
    <figcaption><sub><strong>Diffusion Policy</strong></sub></figcaption>
  </figure>
  <figure style="flex: 1; margin: 0; text-align: center;">
    <img src="video/ours/ligating_loop_ours.gif" width="100%" alt="Ours on LapGym ligating loop">
    <figcaption><sub><strong>Ours</strong></sub></figcaption>
  </figure>
</div>

## Method Overview

### 1. DCT Transformation
Transform discrete action trajectories to DCT coefficients:
```
ĉⱼ = (2/K) Σₙ ξₙ cos(jπ(2n+1)/(2K)), j = 0, ..., M
```

### 2. Flow Matching in Coefficient Space
Perform flow matching on DCT coefficients instead of raw actions:
```
L_FM = 𝔼[(o,ĉ*),t,ε] ||vθ(ĉₜ, o, t) - (ĉ* - ε)||²
```

### 3. Velocity Supervision
Regularize decoded velocities for temporal consistency:
```
L_vel = 𝔼[(ξ,τ)] ||v̇̂(τ) - ξ̇(τ)||²
```

### 4. Combined Loss
```
L_FAFM = L_FM + λ·L_vel  (λ = 1.0)
```

## Installation

FAFM is built based on the OpenPI framework. For installation instructions, please refer to the [OpenPI repository](https://github.com/Physical-Intelligence/openpi).

## Training

### Standard LIBERO Training
```bash
python scripts/train.py fafm_libero --exp_name fafm_libero
```

### Mixed-Frequency Training
```bash
python scripts/train.py fafm_libero_dct --exp_name fafm_mixed_freq
```

## Inference

```python
from openpi.training import config as _config
from openpi.policies import policy_config

# Load FAFM configuration
config = _config.get_config("fafm_libero")

# Create policy
policy = policy_config.create_policy(
    config,
    checkpoint_dir="path/to/checkpoint"
)

# Generate actions (returns continuous velocity field)
actions = policy.sample_actions(observation)
```

## Model Architecture

FAFM extends the π₀ and π₀.₅ architecture with:
1. **DCT Encoding**: Transform velocity sequences to frequency coefficients
2. **Frequency-Weighted Noise**: Prior matching H¹ Sobolev space norm
3. **Continuous Decoding**: Cosine basis expansion for arbitrary-time queries
4. **Derivative Supervision**: First-order consistency regularization

## Acknowledgments

Built on top of the [OpenPI](https://github.com/Physical-Intelligence/openpi) framework by Physical Intelligence.
