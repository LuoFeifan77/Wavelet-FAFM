"""Check imports and spawned workers without datasets, credentials or downloads."""

import argparse
import importlib
import importlib.metadata
import multiprocessing


def check_imports():
    for name in ("cv2", "jax", "torch", "swanlab", "openpi_client", "openpi.training.data_loader"):
        importlib.import_module(name)
    from openpi.training import config
    from transformers.models.siglip import check

    assert config.get_config("fafm_libero").model.action_dim == 7
    assert check.check_whether_transformers_replace_is_installed_correctly()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true", help="Also execute a small operation on each CUDA GPU")
    parser.add_argument("--expected-gpus", type=int, default=1)
    args = parser.parse_args()

    check_imports()
    worker = multiprocessing.get_context("spawn").Process(target=check_imports)
    worker.start()
    worker.join(120)
    if worker.is_alive():
        worker.terminate()
        worker.join()
        raise RuntimeError("Spawned worker import check timed out")
    if worker.exitcode != 0:
        raise RuntimeError(f"Spawned worker failed: exit code {worker.exitcode}")

    for name in ("jax", "jaxlib", "torch", "flax", "numpy", "transformers", "swanlab"):
        print(f"{name}: {importlib.metadata.version(name)}", flush=True)
    print("PASS: training imports, Transformers patch, and spawned worker", flush=True)

    if args.gpu:
        import jax
        import numpy as np

        devices = jax.devices("gpu")
        if len(devices) != args.expected_gpus:
            raise RuntimeError(f"Expected {args.expected_gpus} GPUs, found {len(devices)}")
        for device in devices:
            x = jax.device_put(np.ones((16, 16), dtype=np.float32), device)
            result = (x @ x).block_until_ready()
            np.testing.assert_allclose(np.asarray(result), 16)
            print(f"PASS: {device}", flush=True)


if __name__ == "__main__":
    main()
