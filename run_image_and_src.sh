#在宿主终端新建容器，让新增挂载生效：

  cd /data/nvme0/luofeifan/world-action-models/FAFM
  docker compose -f scripts/docker/compose.yml run --rm openpi_server bash

  #在容器内依次执行：

  # 进入已经运行的容器
  docker exec -it fafm-8gpu bash

  docker exec -it -w /app fafm-8gpu bash

  cd /app
  source /.venv/bin/activate

  # 新容器补齐项目和训练依赖
  # uv sync --frozen
  # uv pip install swanlab==0.10.0

  # 首次使用该数据集时计算归一化统计
  python scripts/compute_norm_stats.py --config-name fafm_libero

  # 上一步成功后再启动训练
  python scripts/train.py fafm_libero --exp_name fafm_libero


  # 多卡训练
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  python scripts/train.py fafm_libero \
    --exp-name fafm_libero_8gpu \
    --batch-size 64 \
    --fsdp-devices 8 \
    --num-workers 8 \
    --resume


  # 1. 设置容器自动重启并确保已启动

  #你的 fafm-8gpu 已使用 sleep infinity 保持运行，设置重启策略即可：

  docker update --restart=always fafm-8gpu
  docker start fafm-8gpu

  #重启策略立即生效。Docker 文档

  #2. 后台启动训练，只执行一次

  docker exec -d \
    -w /app \
    -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
    fafm-8gpu \
    bash -c '
      /.venv/bin/python -u scripts/train.py fafm_libero \
        --exp-name fafm_libero_8gpu \
        --batch-size 64 \
        --fsdp-devices 8 \
        --num-workers 8 \
        --resume \
        >> /app/train_fafm_8gpu.log 2>&1
      train_exit_code=$?
      printf "\nTraining exited: code=%s time=%s\n" \
        "$train_exit_code" "$(date -Is)" >> /app/train_fafm_8gpu.log
      exit "$train_exit_code"
    '