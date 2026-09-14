"""Compute normalization statistics for a config.

This script is used to compute the normalization statistics for a given config. It
will compute the mean and standard deviation of the data in the dataset and save it
to the config assets directory.
"""

import numpy as np
import tqdm
import tyro

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    
    # Create raw dataset
    raw_dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)
    
    # Build transform chain that matches training, INCLUDING DCT if enabled
    transform_chain = [
        *data_config.repack_transforms.inputs,
        *data_config.data_transforms.inputs,
    ]
    
    # Add DCT transforms if use_dct_loading is enabled (CRITICAL for mixed-FPS data)
    if data_config.use_dct_loading:
        if not hasattr(model_config, 'fourier_modes'):
            raise ValueError("use_dct_loading=True requires model_config with 'fourier_modes' attribute")
        if not hasattr(model_config, 'chunk_duration'):
            raise ValueError("use_dct_loading=True requires model_config with 'chunk_duration' attribute")
        
        # Apply DCT resampling BEFORE computing norm stats
        import openpi.transforms as _transforms
        transform_chain.extend([
            _transforms.ExtractDCTCoeffs(
                M=model_config.fourier_modes,
                chunk_duration=model_config.chunk_duration,
            ),
            _transforms.DecodeDCTToActions(
                target_K=model_config.action_horizon,
                chunk_duration=model_config.chunk_duration,
            ),
        ])
    
    # Remove strings (not supported by JAX)
    transform_chain.append(RemoveStrings())
    
    dataset = _data_loader.TransformedDataset(raw_dataset, transform_chain)
    
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        # NOTE: this length is currently hard-coded for DROID.
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def main(config_name: str, max_frames: int | None = None):
    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)

    if data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size, max_frames
        )
    else:
        data_loader, num_batches = create_torch_dataloader(
            data_config, config.model.action_horizon, config.batch_size, config.model, config.num_workers, max_frames
        )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    # Save norm_stats to main dataset
    output_path = config.assets_dirs / data_config.repo_id
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)
    
    # IMPORTANT: If extra_repo_ids are specified, save the SAME norm_stats to all of them
    # This ensures consistent normalization across all datasets during training
    if data_config.extra_repo_ids:
        print(f"\nAlso writing stats to {len(data_config.extra_repo_ids)} extra datasets:")
        for extra_repo_id in data_config.extra_repo_ids:
            extra_output_path = config.assets_dirs / extra_repo_id
            print(f"  - {extra_output_path}")
            normalize.save(extra_output_path, norm_stats)
        print(f"\nTotal: saved to {1 + len(data_config.extra_repo_ids)} dataset directories.")


if __name__ == "__main__":
    tyro.cli(main)
