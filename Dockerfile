# syntax=docker/dockerfile:1
# Build from the repository root: docker build -t wavelet-fafm:train .
ARG CUDA_IMAGE=nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04@sha256:2d913b09e6be8387e1a10976933642c73c840c0b735f0bf3c28d97fc9bc422e0
FROM ${CUDA_IMAGE} AS base

ENV DEBIAN_FRONTEND=noninteractive \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/.venv \
    UV_PYTHON_INSTALL_DIR=/opt/python \
    VIRTUAL_ENV=/.venv \
    PATH=/.venv/bin:$PATH \
    PYTHONPATH=/app:/app/src:/app/packages/openpi-client/src \
    PYTHONUNBUFFERED=1 \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    OPENPI_DATA_HOME=/openpi_assets \
    IS_DOCKER=true

# OpenCV needs GL/GLib even on a headless server. flock is used by the launcher.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates git git-lfs libgl1 libglib2.0-0 libgomp1 util-linux \
    && rm -rf /var/lib/apt/lists/*

FROM base AS builder
COPY --from=ghcr.io/astral-sh/uv:0.5.1 /uv /uvx /usr/local/bin/
RUN apt-get update && apt-get install -y --no-install-recommends build-essential clang \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app

# Install the existing locked OpenPI baseline before adding training extras.
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY packages/openpi-client/ packages/openpi-client/
RUN uv venv --python 3.11.9 /.venv
RUN --mount=type=cache,target=/root/.cache/uv \
    GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --no-install-project --no-dev

# Versions taken from the working training container; do not re-resolve uv.lock.
COPY scripts/docker/requirements-training.txt /tmp/requirements-training.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /.venv/bin/python -r /tmp/requirements-training.txt

COPY src/ src/
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /.venv/bin/python --no-deps -e .
# Required by the repository's PyTorch implementation; apply after installation.
RUN /.venv/bin/python -c 'from pathlib import Path; import shutil, transformers; shutil.copytree("src/openpi/models_pytorch/transformers_replace", Path(transformers.__file__).parent, dirs_exist_ok=True)'

FROM base AS runtime
WORKDIR /app
COPY --from=builder /opt/python /opt/python
COPY --from=builder /.venv /.venv
COPY --from=builder /usr/local/bin/uv /usr/local/bin/uvx /usr/local/bin/
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY packages/openpi-client/ packages/openpi-client/
COPY src/ src/
COPY scripts/ scripts/
COPY run_fafm_libero.sh ./

# No GPU, dataset, login, or model downloads are needed for this build check.
RUN python scripts/docker/check_training_env.py

# docker exec sessions and training jobs are independent of this keepalive.
CMD ["sleep", "infinity"]
