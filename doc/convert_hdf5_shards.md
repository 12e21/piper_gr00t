# convert_hdf5_shards.py - HDF5 并行转换工具

使用多进程并行将 HDF5 文件转换为 LeRobot Dataset，每个 worker 处理部分文件并生成独立的 shard。

### 基本用法

```bash
# 本地并行转换（使用 4 个 worker）
python convert_parallel/convert_hdf5_shards.py \
  --hdf5-root ./data \
  --all \
  --repo-id "your/repo" \
  --workers 4

# SLURM 集群并行转换
python convert_parallel/convert_hdf5_shards.py \
  --hdf5-root ./data \
  --all \
  --repo-id "your/repo" \
  --workers 100 \
  --slurm 1 \
  --partition cpu
```

### 参数说明

| 参数 | 说明 |
|------|------|
| `--hdf5-root` | HDF5 文件根目录 |
| `--all` | 处理目录中所有 HDF5 文件 |
| `--hdf5-files` | 指定要处理的文件（可多个）|
| `--repo-id` | HuggingFace 仓库 ID |
| `--robot-type` | 机器人类型（默认：bi_piper）|
| `--fps` | 视频帧率（默认：30）|
| `--workers` | 并行 worker 数量 |
| `--slurm` | 使用 SLURM（1=启用，0=本地）|
| `--partition` | SLURM 分区名称 |
| `--cpus-per-task` | 每个 task 的 CPU 数量 |
| `--mem-per-cpu` | 每个 CPU 的内存 |
| `--job-name` | 任务名称 |

### 输出结构

转换后会生成多个 shards：
```
your/repo_world_4_rank_0  # Worker 0 处理的文件
your/repo_world_4_rank_1  # Worker 1 处理的文件
your/repo_world_4_rank_2  # Worker 2 处理的文件
your/repo_world_4_rank_3  # Worker 3 处理的文件
```

### 注意事项

- 脚本使用 datatrove 框架进行任务管理，日志存放在 `./logs/<job-name>/`
- 重复运行时如果提示任务已完成，需要删除日志目录或使用新的 `--job-name`
- 更改 `--workers` 数量时，需要清空日志或使用新的 `--job-name`
