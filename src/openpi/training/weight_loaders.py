import dataclasses
import logging
import re
from typing import Protocol, runtime_checkable

import flax.traverse_util
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download

logger = logging.getLogger(__name__)


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "gs://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str
    extra_missing_regex: str | None = None

    def load(self, params: at.Params) -> at.Params:
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        missing = ".*lora.*"
        if self.extra_missing_regex:
            missing = f"({missing}|{self.extra_missing_regex})"
        return _merge_params(loaded_params, params, missing_regex=missing)


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        # Add all missing weights.
        return _merge_params(loaded_params, params, missing_regex=".*")


def _tuple_key_to_str(k: tuple) -> str:
    """Join a tuple key into a '/' separated string, converting int keys to str."""
    return "/".join(str(x) for x in k)


def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    """Merges the loaded parameters with the reference parameters.

    Uses tuple-based flatten_dict to preserve original key types (including
    integer keys from NNX list-indexed modules like MetaDynamicsNet.layers).

    Args:
        loaded_params: The parameters to merge.
        params: The reference parameters.
        missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.

    Returns:
        A new dictionary with the merged parameters.
    """
    flat_ref = flax.traverse_util.flatten_dict(params)
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params)

    result = {}
    skipped = []
    for k, v in flat_loaded.items():
        if k in flat_ref:
            ref_v = flat_ref[k]
            if hasattr(v, "shape") and hasattr(ref_v, "shape") and v.shape != ref_v.shape:
                skipped.append((_tuple_key_to_str(k), v.shape, ref_v.shape))
                continue
            result[k] = v.astype(ref_v.dtype) if v.dtype != ref_v.dtype else v

    if skipped:
        logger.info(
            "Skipped %d params with shape mismatch:\n  %s",
            len(skipped),
            "\n  ".join(f"{name}: loaded={ls} vs model={ms}" for name, ls, ms in skipped),
        )

    flat_loaded.clear()

    pattern = re.compile(missing_regex)
    for k in {k for k in flat_ref if pattern.fullmatch(_tuple_key_to_str(k))}:
        if k not in result:
            result[k] = flat_ref[k]

    return flax.traverse_util.unflatten_dict(result)
