import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_libero_example() -> dict:
    """Creates a random input example for the Libero policy."""
    return {
        "observation/state": np.random.rand(8),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class LiberoInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.

    For your own dataset, you can copy this class and modify the keys based on the comments below to pipe
    the correct elements of your dataset into the model.
    """

    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference.
        # Keep this for your own dataset, but if your dataset stores the images
        # in a different key than "observation/image" or "observation/wrist_image",
        # you should change it below.
        # Pi0 models support three image inputs at the moment: one third-person view,
        # and two wrist views (left and right). If your dataset does not have a particular type
        # of image, e.g. wrist images, you can comment it out here and replace it with zeros like we do for the
        # right wrist image below.
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                # Pad any non-existent images with zero-arrays of the appropriate shape.
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # We only mask padding images for pi0 model, not pi0-FAST. Do not change this for your own dataset.
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        # Pad actions to the model action dimension. Keep this for your own dataset.
        # Actions are only available during training.
        if "actions" in data:
            inputs["actions"] = data["actions"]

        # Pass the prompt (aka language instruction) to the model.
        # Keep this for your own dataset (but modify the key if the instruction is not
        # stored in "prompt"; the output dict always needs to have the key "prompt").
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class LiberoOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.

    For your own dataset, you can copy this class and modify the action dimension based on the comments below.
    """

    def __call__(self, data: dict) -> dict:
        # Only return the first N actions -- since we padded actions above to fit the model action
        # dimension, we need to now parse out the correct number of actions in the return dict.
        # For Libero, we only return the first 7 actions (since the rest is padding).
        # For your own dataset, replace `7` with the action dimension of your dataset.
        return {"actions": np.asarray(data["actions"][:, :7])}


@dataclasses.dataclass(frozen=True)
class PositionsToVelocities(transforms.DataTransformFn):
    """Convert position deltas to velocities via finite difference.
    
    FAFM v3 training requires velocity supervision.
    This transform converts the position-based deltas in LIBERO
    to velocity-based actions.
    
    Note: LIBERO uses OSC_POSE controller with 7D actions:
    - action[0:6] = 6D end-effector delta (3D position + 3D orientation)
    - action[6] = gripper control (binary: -1=open, +1=close)
    
    All 7 dimensions are converted to velocities and used for training.
    Although gripper is binary control, it's learned as part of the velocity field.
    
    State trajectory is now generated internally by the model
    during compute_loss() via integration of velocities. This eliminates
    redundant computation and ensures consistency between training and inference.
    
    IMPORTANT: For mixed-fps datasets, this transform should use the per-sample
    fps from "_dataset_fps" metadata (added by AttachDatasetFps transform).
    The fps parameter is used as a fallback for backward compatibility.
    """
    
    fps: float = 10.0  # Default frame rate (fallback if _dataset_fps not in data)
    chunk_len: int = 50  # Chunk length
    
    def __call__(self, data: dict) -> dict:
        """Convert position deltas to velocities.
        
        LIBERO actions are position deltas (displacement per timestep).
        To get velocities, we simply divide by dt:
            velocity[k] = delta[k] / dt
        
        This is NOT the same as finite difference of deltas, which would be:
            (delta[k+1] - delta[k]) / dt  # ❌ This is acceleration!
        """
        if "actions" not in data:
            return data
        
        actions = np.asarray(data["actions"])  # (K, 7): 6D EEF delta + 1D gripper
        
        # Use per-sample fps if available (for mixed-fps datasets)
        # Otherwise fall back to the configured fps
        if "_dataset_fps" in data:
            sample_fps = float(np.asarray(data["_dataset_fps"]))
        else:
            sample_fps = self.fps
        
        # Compute dt based on actual sample fps
        dt = 1.0 / sample_fps
        
        # Convert deltas to velocities
        # delta[k] is the displacement from t=k to t=k+1
        # velocity[k] = delta[k] / dt
        velocities = actions / dt
        
        # Store velocities (7D) as actions for model training
        data["actions"] = velocities  # (K, 7) - full velocity field
        
        # Clean up metadata to avoid leaking into training
        if "_dataset_fps" in data:
            del data["_dataset_fps"]
        
        # State trajectory is now generated internally by the model
        # No need to compute it here
        
        return data


@dataclasses.dataclass(frozen=True)
class FAFMLiberoOutputs(transforms.DataTransformFn):
    """Output transform for FAFM v3 on LIBERO.

    FAFM v3's sample_actions() returns velocities (B, K, 7) for LIBERO.
    
    Conversion logic:
    - Training: velocity[k] = delta[k] / dt, where dt = 1/fps = 0.1s
    - Inference: delta[k] = velocity[k] * dt
    
    Since training fps = inference control freq (both 10Hz), we have:
        delta = velocity * 0.1
    
    Note: LIBERO uses Operational Space Control (OSC_POSE) which expects:
    - action[0:6] = 6D end-effector delta (3D position + 3D orientation)
    - action[6] = gripper control (binary: -1=open, +1=close)
    """

    fps: float = 10.0  # Must match training fps

    def __call__(self, data: dict) -> dict:
        velocities = np.asarray(data["actions"])  # (B, K, 7) or (K, 7)
        
        # Convert velocities back to position deltas
        dt = 1.0 / self.fps  # 0.1s
        deltas = velocities * dt
        
        return {"actions": deltas}
