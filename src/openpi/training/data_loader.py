from collections.abc import Iterator, Sequence
import json
import logging
import multiprocessing
import os
import pathlib
import typing
from typing import Literal, Protocol, SupportsIndex, TypeVar

import jax
import jax.numpy as jnp
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset
import numpy as np
import torch

import openpi.models.model as _model
import openpi.training.config as _config
from openpi.training.droid_rlds_dataset import DroidRldsDataset
import openpi.transforms as _transforms

T_co = TypeVar("T_co", covariant=True)


class Dataset(Protocol[T_co]):
    """Interface for a dataset with random access."""

    def __getitem__(self, index: SupportsIndex) -> T_co:
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class IterableDataset(Protocol[T_co]):
    """Interface for an iterable dataset."""

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of IterableDataset should implement __iter__.")

    def __len__(self) -> int:
        raise NotImplementedError("Subclasses of Dataset should implement __len__.")


class DataLoader(Protocol[T_co]):
    """Interface for a data loader."""

    def data_config(self) -> _config.DataConfig:
        """Get the data config for this data loader."""
        raise NotImplementedError("Subclasses of DataLoader should implement data_config.")

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError("Subclasses of DataLoader should implement __iter__.")


class TransformedDataset(Dataset[T_co]):
    def __init__(self, dataset: Dataset, transforms: Sequence[_transforms.DataTransformFn]):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)

    def __getitem__(self, index: SupportsIndex) -> T_co:
        return self._transform(self._dataset[index])

    def __len__(self) -> int:
        return len(self._dataset)


class IterableTransformedDataset(IterableDataset[T_co]):
    def __init__(
        self,
        dataset: IterableDataset,
        transforms: Sequence[_transforms.DataTransformFn],
        *,
        is_batched: bool = False,
    ):
        self._dataset = dataset
        self._transform = _transforms.compose(transforms)
        self._is_batched = is_batched

    def __iter__(self):
        for sample in self._dataset:
            if self._is_batched:
                # Transforms are designed to be applied to individual samples. So we need to split the batch into
                # individual samples and apply the transform to each sample individually.
                batch_size = next(v.shape[0] for v in sample.values())

                # Split batch into individual samples using tree_map
                individual_samples = [jax.tree.map(lambda x: x[i], sample) for i in range(batch_size)]  # noqa: B023

                # Transform each sample
                transformed = [self._transform(s) for s in individual_samples]

                # Recombine batch with tree_map
                yield jax.tree.map(lambda *x: np.stack(x, axis=0), *transformed)
            else:
                yield self._transform(sample)

    def __len__(self) -> int:
        return len(self._dataset)


class FakeDataset(Dataset):
    def __init__(self, model_config: _model.BaseModelConfig, num_samples: int):
        self._num_samples = num_samples
        self._observation_spec, self._action_spec = model_config.inputs_spec()

    def __getitem__(self, index: SupportsIndex) -> dict:
        rng = jax.random.key(index.__index__())

        def make_from_spec(spec: jax.ShapeDtypeStruct):
            nonlocal rng
            rng, data_rng = jax.random.split(rng)
            # Remove the batch dimension.
            shape = spec.shape[1:]
            if spec.dtype == jnp.float32:
                return jax.random.uniform(data_rng, shape=shape, minval=-1.0, maxval=1.0)
            if spec.dtype == jnp.int32:
                return jax.random.randint(data_rng, shape=shape, minval=0, maxval=2048)
            return jnp.zeros(shape=shape, dtype=spec.dtype)

        observation = jax.tree.map(make_from_spec, self._observation_spec)
        action = jax.tree.map(make_from_spec, self._action_spec)

        return {
            **observation.to_dict(),
            "actions": action,
        }

    def __len__(self) -> int:
        return self._num_samples


# FAFMStage2Dataset removed - v3 does not use Stage 1 theta vectors
# Data is processed directly from positions to velocities in transforms


def create_torch_dataset(
    data_config: _config.DataConfig, action_horizon: int, model_config: _model.BaseModelConfig
) -> Dataset:
    """Create a dataset for training.
    
    If data_config.use_dct_loading=True:
        - Loads full episode sequences (all available frames)
        - DCT extraction happens in transform pipeline
        - Decoding to target action_horizon happens in transform pipeline
        
    Otherwise (default):
        - Uses standard LeRobot loading with fixed fps
        - delta_timestamps limits to action_horizon frames
    """
    repo_id = data_config.repo_id
    if repo_id is None:
        raise ValueError("Repo ID is not set. Cannot create dataset.")
    if repo_id == "fake":
        return FakeDataset(model_config, num_samples=1024)

    dataset_meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    
    # DCT loading branch: load denser samples for fixed-duration chunks
    if data_config.use_dct_loading:
        # Validate model_config supports DCT
        if not hasattr(model_config, 'fourier_modes'):
            raise ValueError(
                "use_dct_loading=True requires model_config with 'fourier_modes' attribute "
                "(e.g., FAFMConfig). Current model_config does not support DCT loading."
            )
        if not hasattr(model_config, 'chunk_duration'):
            raise ValueError(
                "use_dct_loading=True requires model_config with 'chunk_duration' attribute"
            )
        
        # source_fps must equal dataset fps: LeRobot validates that every
        # delta_timestamp is a multiple of 1/dataset_fps, so dct_source_fps
        # cannot be independently higher.  dct_source_fps is kept for legacy
        # documentation purposes; at runtime we always use dataset_meta.fps.
        source_fps = dataset_meta.fps
        if data_config.dct_source_fps is not None and data_config.dct_source_fps != dataset_meta.fps:
            logging.warning(
                f"dct_source_fps={data_config.dct_source_fps} differs from "
                f"dataset fps={dataset_meta.fps}. LeRobot requires delta_timestamps "
                f"to be multiples of 1/dataset_fps, so dct_source_fps is ignored."
            )
        chunk_duration = model_config.chunk_duration

        # Number of frames covering exactly chunk_duration at this dataset's fps
        K_source = round(chunk_duration * source_fps)

        logging.info(
            f"DCT loading: chunk_duration={chunk_duration}s, "
            f"fps={source_fps}Hz, K_source={K_source}, "
            f"target_K={action_horizon}, M={model_config.fourier_modes}"
        )

        dataset = lerobot_dataset.LeRobotDataset(
            data_config.repo_id,
            delta_timestamps={
                key: [t / source_fps for t in range(K_source)]
                for key in data_config.action_sequence_keys
            },
        )
    else:
        # Standard loading path: load K_load frames at dataset fps.
        # When model_config has chunk_duration, derive K_load = round(T * fps) so
        # that the loaded window exactly matches the physical time span T.
        # Falls back to action_horizon for non-FAFM models.
        if hasattr(model_config, 'chunk_duration'):
            K_load = round(model_config.chunk_duration * dataset_meta.fps)
        else:
            K_load = action_horizon
        dataset = lerobot_dataset.LeRobotDataset(
            data_config.repo_id,
            delta_timestamps={
                key: [t / dataset_meta.fps for t in range(K_load)] for key in data_config.action_sequence_keys
            },
        )

    # Attach fps metadata to main dataset for proper velocity calculation
    # This is critical for mixed-fps datasets
    dataset_transforms = []
    if data_config.prompt_from_task:
        dataset_transforms.append(_transforms.PromptFromLeRobotTask(dataset_meta.tasks))
    dataset_transforms.append(_transforms.AttachDatasetFps(fps=dataset_meta.fps))
    dataset = TransformedDataset(dataset, dataset_transforms)

    # Concatenate extra datasets if specified (each with its own fps)
    if data_config.extra_repo_ids:
        extra_datasets = []
        for extra_repo_id in data_config.extra_repo_ids:
            extra_meta = lerobot_dataset.LeRobotDatasetMetadata(extra_repo_id)
            if data_config.use_dct_loading:
                extra_K = round(model_config.chunk_duration * extra_meta.fps)
                extra_ds = lerobot_dataset.LeRobotDataset(
                    extra_repo_id,
                    delta_timestamps={
                        key: [t / extra_meta.fps for t in range(extra_K)]
                        for key in data_config.action_sequence_keys
                    },
                )
            else:
                if hasattr(model_config, 'chunk_duration'):
                    extra_K = round(model_config.chunk_duration * extra_meta.fps)
                else:
                    extra_K = action_horizon
                extra_ds = lerobot_dataset.LeRobotDataset(
                    extra_repo_id,
                    delta_timestamps={
                        key: [t / extra_meta.fps for t in range(extra_K)]
                        for key in data_config.action_sequence_keys
                    },
                )
            # Attach fps metadata to each extra dataset
            extra_transforms = []
            if data_config.prompt_from_task:
                extra_transforms.append(_transforms.PromptFromLeRobotTask(extra_meta.tasks))
            extra_transforms.append(_transforms.AttachDatasetFps(fps=extra_meta.fps))
            extra_ds = TransformedDataset(extra_ds, extra_transforms)
            extra_datasets.append(extra_ds)
        import torch.utils.data as torch_data
        dataset = torch_data.ConcatDataset([dataset] + extra_datasets)
        logging.info(f"Concatenated {1 + len(extra_datasets)} datasets, total samples: {len(dataset)}")

    # FAFM v3: no Stage 1 theta wrapping needed
    # Velocity conversion happens in data transforms

    return dataset


def create_rlds_dataset(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    shuffle: bool = False,
) -> Dataset:
    # At the moment, we only support DROID for RLDS datasets.
    return DroidRldsDataset(
        data_dir=data_config.rlds_data_dir,
        batch_size=batch_size,
        shuffle=shuffle,
        action_chunk_size=action_horizon,
        action_space=data_config.action_space,
        datasets=data_config.datasets,
    )


def transform_dataset(
    dataset: Dataset, 
    data_config: _config.DataConfig, 
    *, 
    skip_norm_stats: bool = False,
    model_config: _model.BaseModelConfig | None = None,
) -> Dataset:
    """Transform the dataset by applying the data transforms.
    
    Args:
        dataset: The dataset to transform
        data_config: Data configuration
        skip_norm_stats: If True, skip normalization
        model_config: Model configuration (optional, required for DCT loading)
        
    If data_config.use_dct_loading=True and model_config is provided:
        - Applies ExtractDCTCoeffs (K_source → M+1 coefficients)
        - Applies DecodeDCTToActions (M+1 → action_horizon samples, resampling)
        - Applies Normalize in velocity space (same domain as standard path)
        
    Otherwise (default, backward compatible):
        - Standard transform chain: repack -> data -> normalize -> model
    """
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    # Build transform chain
    transforms_chain = [
        *data_config.repack_transforms.inputs,
        *data_config.data_transforms.inputs,
    ]
    
    # DCT loading branch
    if data_config.use_dct_loading:
        if model_config is None:
            raise ValueError(
                "use_dct_loading=True requires model_config to be provided to transform_dataset"
            )
        if not hasattr(model_config, 'fourier_modes'):
            raise ValueError(
                "use_dct_loading=True requires model_config with 'fourier_modes' attribute"
            )
        if not hasattr(model_config, 'chunk_duration'):
            raise ValueError(
                "use_dct_loading=True requires model_config with 'chunk_duration' attribute"
            )
        
        # DCT resampling: convert K_source frames (at source fps) to K=action_horizon
        # frames (at model fps) via the continuous DCT basis.
        # Step 1: compress K_source → M+1 frequency coefficients
        transforms_chain.append(
            _transforms.ExtractDCTCoeffs(
                M=model_config.fourier_modes,
                chunk_duration=model_config.chunk_duration,
            )
        )
        
        # Step 2: decode M+1 coefficients back to exactly action_horizon velocity samples
        # at uniform tau ∈ [0, T]. This is the resampling step: the source fps no longer
        # matters; the output is always (action_horizon, d) in velocity space.
        transforms_chain.append(
            _transforms.DecodeDCTToActions(
                target_K=model_config.action_horizon,
                chunk_duration=model_config.chunk_duration,
            )
        )
        
        # Step 3: normalize in velocity space (same domain as the standard path).
        # norm_stats are computed on decoded velocities, not on DCT coefficients.
        transforms_chain.append(
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm)
        )
    else:
        # Standard path (backward compatible, default)
        transforms_chain.append(
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm)
        )
    
    # Add model transforms
    transforms_chain.extend(data_config.model_transforms.inputs)

    return TransformedDataset(dataset, transforms_chain)


def transform_iterable_dataset(
    dataset: IterableDataset,
    data_config: _config.DataConfig,
    *,
    skip_norm_stats: bool = False,
    is_batched: bool = False,
) -> IterableDataset:
    """Transform the dataset by applying the data transforms."""
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    return IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        is_batched=is_batched,
    )


def create_data_loader(
    config: _config.TrainConfig,
    *,
    sharding: jax.sharding.Sharding | None = None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
    framework: Literal["jax", "pytorch"] = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        config: The training configuration.
        sharding: The sharding to use for the data loader (JAX only).
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return.
        skip_norm_stats: Whether to skip data normalization.
        framework: The framework to use ("jax" or "pytorch").
    """
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(f"data_config: {data_config}")

    if data_config.rlds_data_dir is not None:
        return create_rlds_data_loader(
            data_config,
            action_horizon=config.model.action_horizon,
            batch_size=config.batch_size,
            sharding=sharding,
            shuffle=shuffle,
            num_batches=num_batches,
            skip_norm_stats=skip_norm_stats,
            framework=framework,
        )
    return create_torch_data_loader(
        data_config,
        model_config=config.model,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=config.seed,
        skip_norm_stats=skip_norm_stats,
        framework=framework,
    )


def create_torch_data_loader(
    data_config: _config.DataConfig,
    model_config: _model.BaseModelConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create a data loader for training.

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
        num_workers: The number of worker processes to use. If zero, the data loader will
            execute in the main process.
        seed: The seed to use for shuffling the data.
    """
    dataset = create_torch_dataset(data_config, action_horizon, model_config)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats, model_config=model_config)

    # Use TorchDataLoader for both frameworks
    # For PyTorch DDP, create DistributedSampler and divide batch size by world size
    # For JAX, divide by process count
    sampler = None
    if framework == "pytorch":
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=torch.distributed.get_world_size(),
                rank=torch.distributed.get_rank(),
                shuffle=shuffle,
                drop_last=True,
            )
            local_batch_size = batch_size // torch.distributed.get_world_size()
        else:
            local_batch_size = batch_size
    else:
        local_batch_size = batch_size // jax.process_count()

    logging.info(f"local_batch_size: {local_batch_size}")
    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=None if framework == "pytorch" else sharding,
        shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
        sampler=sampler,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework=framework,
    )

    return DataLoaderImpl(data_config, data_loader)


def create_rlds_data_loader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    *,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    framework: str = "jax",
) -> DataLoader[tuple[_model.Observation, _model.Actions]]:
    """Create an RLDS data loader for training.

    Note: This data loader requires some extra dependencies -- see examples/droid/README_train.md

    Args:
        data_config: The data configuration.
        action_horizon: The action horizon.
        batch_size: The batch size.
        sharding: The sharding to use for the data loader. If None, the data loader will
            use a single device sharding.
        skip_norm_stats: Whether to skip data normalization.
        shuffle: Whether to shuffle the data.
        num_batches: Determines the number of batches to return. If the number exceeds the
            number of batches in the dataset, the data loader will loop over the dataset.
            If not provided, will iterate over the dataset indefinitely.
    """
    if framework == "pytorch":
        raise NotImplementedError("PyTorch RLDS data loader is not supported yet")
    dataset = create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=shuffle)
    dataset = transform_iterable_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats, is_batched=True)

    data_loader = RLDSDataLoader(
        dataset,
        sharding=sharding,
        num_batches=num_batches,
    )

    return DataLoaderImpl(data_config, data_loader)


class TorchDataLoader:
    """Torch data loader implementation."""

    def __init__(
        self,
        dataset,
        local_batch_size: int,
        *,
        sharding: jax.sharding.Sharding | None = None,
        shuffle: bool = False,
        sampler: torch.utils.data.Sampler | None = None,
        num_batches: int | None = None,
        num_workers: int = 0,
        seed: int = 0,
        framework: str = "jax",
    ):
        """Create a PyTorch data loader.

        Args:
            dataset: The dataset to load.
            local_batch_size: The local batch size for each process.
            sharding: The sharding to use for the data loader.
            shuffle: Whether to shuffle the data.
            num_batches: If provided, determines the number of returned batches. If the
                number is larger than the number of batches in the dataset, the data loader
                will loop over the dataset. If not provided, will iterate over the dataset
                indefinitely.
            num_workers: The number of worker processes to use. If zero, the data loader will
                execute in the main process.
            seed: The seed to use for shuffling the data.
        """
        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if len(dataset) < local_batch_size:
            raise ValueError(f"Local batch size ({local_batch_size}) is larger than the dataset size ({len(dataset)}).")

        # Store sharding - None for PyTorch, JAX sharding for JAX
        self._sharding = sharding
        if sharding is None and framework == "jax":
            # Use data parallel sharding by default for JAX only.
            self._sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )
        self._num_batches = num_batches

        mp_context = None
        if num_workers > 0:
            mp_context = multiprocessing.get_context("spawn")

        generator = torch.Generator()
        generator.manual_seed(seed)
        self._data_loader = torch.utils.data.DataLoader(
            typing.cast(torch.utils.data.Dataset, dataset),
            batch_size=local_batch_size,
            shuffle=(sampler is None and shuffle),  # Don't shuffle if using sampler
            sampler=sampler,
            num_workers=num_workers,
            multiprocessing_context=mp_context,
            persistent_workers=num_workers > 0,
            collate_fn=_collate_fn,
            worker_init_fn=_worker_init_fn,
            drop_last=True,
            generator=generator,
        )

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._data_loader

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._data_loader)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                # For JAX, convert to sharded arrays; for PyTorch, return torch tensors
                if self._sharding is not None:
                    yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)
                else:
                    yield jax.tree.map(torch.as_tensor, batch)


def _collate_fn(items):
    """Collate the batch elements into batched numpy arrays."""
    # Make sure to convert to numpy arrays before stacking since some of the incoming elements
    # may be JAX arrays.
    return jax.tree.map(lambda *xs: np.stack([np.asarray(x) for x in xs], axis=0), *items)


def _worker_init_fn(worker_id: int) -> None:
    """Tell JAX inside the worker process not to preallocate the GPU memory."""
    # NOTE: This is called after jax is imported inside the worker process. This
    # means that this approach will not work for selecting the backend.
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"


class RLDSDataLoader:
    """Shallow wrapper around the DROID data loader to make it compatible with openpi.

    All batching already happens in the DROID dataset, so we don't need to do anything here.
    """

    def __init__(
        self,
        dataset: DroidRldsDataset,
        *,
        sharding: jax.sharding.Sharding | None = None,
        num_batches: int | None = None,
    ):
        self._dataset = dataset
        self._num_batches = num_batches

        if jax.process_count() > 1:
            raise NotImplementedError("Data loading with multiple processes is not supported.")

        if sharding is None:
            # Use data parallel sharding by default.
            sharding = jax.sharding.NamedSharding(
                jax.sharding.Mesh(jax.devices(), ("B",)),
                jax.sharding.PartitionSpec("B"),
            )

        self._sharding = sharding
        self._num_batches = num_batches

    def __iter__(self):
        num_items = 0
        while True:
            data_iter = iter(self._dataset)
            while True:
                if self._num_batches is not None and num_items >= self._num_batches:
                    return
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break  # We've exhausted the dataset. Create a new iterator and start over.
                num_items += 1
                yield jax.tree.map(lambda x: jax.make_array_from_process_local_data(self._sharding, x), batch)


class DataLoaderImpl(DataLoader):
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader | RLDSDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            yield _model.Observation.from_dict(batch), batch["actions"]
