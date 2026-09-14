"""Apply the repository's Transformers 4.53.2 patch after a pip installation."""

import importlib.metadata
import importlib.util
from pathlib import Path
import shutil


def main():
    version = importlib.metadata.version("transformers")
    if version != "4.53.2":
        raise RuntimeError(f"This patch requires transformers==4.53.2; installed: {version}")
    spec = importlib.util.find_spec("transformers")
    if spec is None or spec.origin is None:
        raise RuntimeError("Cannot locate the installed transformers package")
    source = Path(__file__).resolve().parents[1] / "src/openpi/models_pytorch/transformers_replace"
    if not source.is_dir():
        raise FileNotFoundError(f"Keep this script with the repository: missing {source}")
    target = Path(spec.origin).parent
    shutil.copytree(source, target, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    print(f"Applied Transformers {version} patch to {target}")


if __name__ == "__main__":
    main()
