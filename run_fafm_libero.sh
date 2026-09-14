#!/bin/sh
set -eu

# Run on the host. Log in once if needed:
# docker exec -it -w /app fafm-8gpu /.venv/bin/swanlab login
container=fafm-8gpu
log_file=/app/train_fafm_8gpu.log

# Also detect training started by the older script, which did not take a lock.
processes=$(docker top "$container" -eo pid,comm,args)
training_pids=$(printf '%s\n' "$processes" | awk '
    $2 ~ /^python([0-9.]+)?$/ &&
    index($0, "scripts/train.py fafm_libero ") &&
    index($0, "--exp-name fafm_libero_8gpu ") { print $1 }
')

if [ -n "$training_pids" ]; then
    printf '检测到训练进程（宿主机 PID）：\n%s\n不重复启动，转为查看日志。\n' "$training_pids"
else
    docker exec -d \
        -w /app \
        -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
        "$container" \
        bash -c '
            exec >> /app/train_fafm_8gpu.log 2>&1
            exec 9>/tmp/fafm_libero_8gpu.lock
            if ! flock -n 9; then
                printf "已有启动任务持有训练锁，本次不重复启动。\n"
                exit 1
            fi
            printf "\nTraining starting: time=%s\n" "$(date -Is)"
            /.venv/bin/python -u scripts/train.py fafm_libero \
                --exp-name fafm_libero_8gpu \
                --batch-size 64 \
                --fsdp-devices 8 \
                --num-workers 8 \
                --resume 9>&-
            train_exit_code=$?
            printf "\nTraining exited: code=%s time=%s\n" \
                "$train_exit_code" "$(date -Is)"
            exit "$train_exit_code"
        '
    printf '后台启动请求已提交，请通过下面的新日志确认训练状态。\n'
fi

printf '日志：%s\n按 Ctrl+C 只退出日志查看，后台训练继续。\n' "$log_file"
exec docker exec "$container" tail -n 20 -F "$log_file"
