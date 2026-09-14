# 在另一台服务器重建 FAFM 训练环境

在**宿主机**的项目根目录执行以下命令。训练镜像使用根目录 `Dockerfile`；
原有 `serve_policy.Dockerfile` 和 `compose.yml` 继续用于推理。

## 环境和文件

- 新服务器需已安装 Docker（包含 BuildKit）、Docker Compose v2、NVIDIA 驱动和 NVIDIA Container Toolkit，并允许当前用户使用 Docker。
- 本环境基于当前 x86_64 / H100 训练环境。原服务器驱动为 580.65.06；构建过程不需要 GPU，运行 GPU 训练需要兼容驱动。
- 基础镜像固定为 CUDA 12.2.2 / cuDNN 8 / Ubuntu 22.04，并固定镜像摘要。
- Python 3.11.9；`uv.lock` 锁定 JAX 0.5.3、PyTorch 2.8.0、Flax 0.10.2、Transformers 4.53.2 等。
- `requirements-training.txt` 单独固定原训练容器中的 SwanLab 0.10.0 及其依赖版本，在 `uv sync` 后安装。
- PyTorch/JAX 的 Python 包还会安装各自需要的 CUDA 库；基础镜像的 CUDA 版本不代表所有 Python 包的 CUDA 版本。
- Dockerfile 通过多阶段构建，最终镜像不保留编译工具和下载缓存，不打包数据集、checkpoint、运行日志或账号凭据。GPU 依赖本身仍较大。

保持整个仓库一起迁移，不能只复制 Dockerfile。构建会访问 Docker Hub、GHCR、GitHub、PyPI 和 Ubuntu 软件源。

## 构建

```bash
git clone https://github.com/LuoFeifan77/Wavelet-FAFM.git
cd Wavelet-FAFM
docker build --progress=plain -t wavelet-fafm:train .
```

构建最后自动检查训练依赖导入、Transformers 补丁，以及 `spawn` 子进程导入。
成功时会显示 `PASS: training imports, Transformers patch, and spawned worker`。

## 准备数据与持久化文件

单独迁移原服务器的 **LeRobot 格式** LIBERO 数据集，包括 `meta/`、数据文件、图像/视频以及已有的 `norm_stats.json`。
原配置在容器内读取 `/data/libero_lerobot`。宿主路径可以任意指定：

```bash
export FAFM_DATASET_DIR=/你的绝对路径/libero_lerobot
export OPENPI_CACHE_DIR="$PWD/.cache/openpi"
mkdir -p "$OPENPI_CACHE_DIR"
test -f "$FAFM_DATASET_DIR/meta/info.json"
```

为减少下载，可将旧服务器的 OpenPI 缓存内容复制到 `$OPENPI_CACHE_DIR`。
从零训练所需的 `pi0_base` 权重和 tokenizer 不在镜像中，没有缓存时运行阶段需要联网下载。

如果要继续旧实验，还需将整个
`checkpoints/fafm_libero/fafm_libero_8gpu/` 复制到新仓库下的同名路径，
包括 checkpoint 子目录及 `swanlab_id.txt`。单纯克隆 Git 仓库不会带来这些被忽略的文件。

## 启动常驻容器

```bash
docker compose -f scripts/docker/compose.train.yml up -d --no-build
docker exec -it -w /app fafm-8gpu bash
```

`exit` 只退出交互终端，容器由 `sleep infinity` 保持运行。`restart: always` 设置容器自动重启。
宿主机仓库挂载到 `/app`，因此日志、checkpoint 和代码修改保留在宿主机。
数据集以可写方式挂载，以便首次计算归一化统计。16 GB 共享内存用于多进程数据加载。
Compose 使用所有可见 GPU，下面的训练命令按 **8 张 GPU** 配置。

在宿主机确认 GPU 和关键依赖：

```bash
docker exec fafm-8gpu nvidia-smi
docker exec fafm-8gpu python scripts/docker/check_training_env.py --gpu --expected-gpus 8
```

## SwanLab 和训练

默认使用 SwanLab 云端记录，首次在新容器中交互登录：

```bash
docker exec -it -w /app fafm-8gpu /.venv/bin/swanlab login
```

登录凭据保存在该容器内，不会写入构建镜像。删除并重建容器后需重新登录。
如果只需要本地记录，在 `compose up` **之前**设置 `export SWANLAB_MODE=offline`；离线模式不需要登录。

### 恢复迁移过来的实验

确认 checkpoint 已复制，然后在宿主机执行：

```bash
sh run_fafm_libero.sh
```

脚本自动检查重复训练，后台启动 `--resume` 并显示日志。
`Ctrl+C` 只退出日志查看，SSH 断线或退出交互终端不影响后台训练。
容器或服务器重启仍会终止训练；容器自动重启后需重新执行脚本，从最近 checkpoint 恢复。

### 新实验：没有 checkpoint 时

不能使用 `--resume`。如果数据集还没有 `norm_stats.json`，先执行：

```bash
docker exec -w /app fafm-8gpu python scripts/compute_norm_stats.py --config-name fafm_libero
```

仅启动一次新的训练：

```bash
docker exec -d -w /app fafm-8gpu bash -c '
  exec >> /app/train_fafm_8gpu.log 2>&1
  exec 9>/tmp/fafm_libero_8gpu.lock
  flock -n 9 || exit 1
  python -u scripts/train.py fafm_libero \
    --exp-name fafm_libero_8gpu \
    --batch-size 64 --fsdp-devices 8 --num-workers 8 9>&-
  train_exit_code=$?
  printf "\nTraining exited: code=%s time=%s\n" "$train_exit_code" "$(date -Is)"
  exit "$train_exit_code"
'
docker exec fafm-8gpu tail -n 30 -F /app/train_fafm_8gpu.log
```

首次运行会初始化模型并进行 JIT 编译，初期可能较慢。新实验保存过 checkpoint 后，可使用 `sh run_fafm_libero.sh` 恢复。

## 使用说明

- 进入容器后，`python` 已指向 `/.venv/bin/python`，无需手动激活环境。
- 使用 `python scripts/train.py ...`。不要再次执行 `uv sync` 或普通 `uv run`，它们可能按原始锁文件移除额外安装的 SwanLab 或恢复被覆盖的依赖。
- 镜像是训练环境的重建配方，不包括正在运行的进程、主机驱动、数据、模型权重和账号登录状态。
- 在当前旧服务器上不要再次运行这份 Compose：同名 `fafm-8gpu` 容器已存在。本流程用于另一台服务器。

构建方法参考 [uv Docker 指南](https://docs.astral.sh/uv/guides/integration/docker/) 和
[Docker 多阶段构建](https://docs.docker.com/build/building/multi-stage/)。
