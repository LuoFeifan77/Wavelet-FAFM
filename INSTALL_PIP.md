# 用 pip 重建训练镜像的 Python 环境

`requirements.txt` 导出自 `wavelet-fafm:train` 镜像（`df39157e7553`）的 `/.venv`，
包含 217 个 Python 包：215 项固定版本或 Git 提交，以及本仓库的两个可编辑安装项。
这是一份已安装依赖快照，不是虚拟环境二进制压缩包；安装时仍需要下载软件包。

目标环境为 **Linux x86_64、Python 3.11（原版本 3.11.9）、NVIDIA GPU**。
其中 JAX 0.5.3、PyTorch 2.8.0、Flax 0.10.2、Transformers 4.53.2、SwanLab 0.10.0
和 NVIDIA CUDA Python 依赖与镜像保持一致。此文件不适用于直接在 macOS、Windows 或 CPU-only 环境重建。

## 系统前提

宿主机需有可用的 NVIDIA 驱动；原服务器使用 580.65.06，`nvidia-smi` 应能正常运行。
Python 包中的 CUDA 库不包含宿主机驱动，也不替代驱动安装。

需要 Git/Git LFS、编译工具以及 OpenCV 所需的 GL/GLib 等系统库。
例如在 Ubuntu 上由管理员执行：

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates git git-lfs build-essential clang \
  libgl1 libglib2.0-0 libgomp1
```

另外准备 Python 3.11 和它的 `venv` 支持（也可使用已有的 Python 3.11 Conda 环境）。
在已准备好的环境内安装 Python 依赖不需要 sudo。
没有安装系统依赖的权限时，可以使用仓库 Dockerfile 构建出的环境。

## 安装

先克隆或复制**完整仓库**并进入根目录。不能只复制 `requirements.txt`：
`openpi` 和 `openpi-client` 来自本仓库源码；容器中的 `/app` 绝对路径已经改成了相对路径。
LeRobot 固定到原 Git commit，安装时会访问 GitHub。

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

GIT_LFS_SKIP_SMUDGE=1 python -m pip install -r requirements.txt

# 必需：恢复镜像中已经应用过的 Transformers 源码补丁
python scripts/install_transformers_patch.py

python -m pip check
python scripts/docker/check_training_env.py
```

在八卡服务器上进一步检查每张 GPU：

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python scripts/docker/check_training_env.py --gpu --expected-gpus 8
```

`requirements.txt` 不会自动复制手工修改的第三方源码，所以补丁步骤不能省略；
重新安装 Transformers 后也要重新应用补丁。

## 与镜像、数据的关系

- Dockerfile 仍以 `uv.lock` 加 `scripts/docker/requirements-training.txt` 构建；根目录的
  `requirements.txt` 是该镜像完整 Python 环境的 pip 安装入口。
- Python 解释器、Linux 系统库、驱动、模型权重、数据集、checkpoint 和 SwanLab 登录凭据不在 requirements 中。
- 直接在宿主机运行时，需要将训练配置的数据路径设置为宿主机实际路径，或提供与原配置相同的 `/data/libero_lerobot` 路径。
- 依赖文件已与镜像的安装版本逐项核对，镜像内依赖兼容性检查通过；这不等于已经在所有目标操作系统上完成全新 pip 安装验证。

依赖导出的机制参考 [pip freeze 文档](https://pip.pypa.io/en/stable/cli/pip_freeze/)。
