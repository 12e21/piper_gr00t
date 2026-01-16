# 使用Piper数据微调GR00T N1.6
## 1. 配置环境
配置hdf5转lerobot dataset v2.1的环境
```bash
git clone https://github.com/12e21/piper_gr00t.git
conda create -n hdf2lerobot python=3.10
conda activate hdf2lerobot
conda install ffmpeg=7 -c conda-forge
pip install uv
uv pip install lerobot==0.4.2 h5py opencv-python ipdb typer tqdm numpy datatrove
```

## 2. 使用脚本转换hdf5文件

```bash
python hdf2lerobotv21.py --all \ # 转换目录下所有hdf5文件
  --repo-id "your/repo" \ # 仓库id
  --hdf5-root "./data" \ # 存放hdf5文件的目录
  --push # 上传到huggingface
```
脚本支持以下数据格式：

- **动作**: 14 维关节位置（双臂各 7 个关节）
- **观测**: 14 维关节位置状态
- **图像**: 3 个视角（左手腕、中间、右手腕），480x640 分辨率


## 3. 配置GR00T N1.6环境
```bash
git clone --recurse-submodules https://github.com/NVIDIA/Isaac-GR00T
cd Isaac-GR00T
git submodule update --init --recursive

conda create -n gr00t16 python=3.10 -y && conda activate gr00t16
pip install uv

uv pip install -e .
uv pip install jsonlines lerobot
conda install ffmpeg=7 -c conda-forge

hf download nvidia/GR00T-N1.6-3B
```

## 4. 设置数据集参数
1. 将`config_for_gr00tn16`目录复制到GR00T的根目录下
2. 将`config_for_gr00tn16/modality.json`存放在lerobot数据集的`meta`目录下
  - 如果需要修改Action的绝对和相对表示，可以在这个文件中修改，默认采用arm为相对，gripper为绝对
3. 启动微调脚本（数据集存放位置，GPU数量，微调步数等设置请在脚本中修改）
```bash
bash config_for_gr00tn16/finetune_bi_piper.sh
```

## 测试
1. hdf5 转换 Lerobot Dataset转换成功，速度不是很快
2. 使用两个episode转换得到的数据集微调GR00T N1.6，使用两张A100 80G，Loss正常下降

# HDF5 Tools使用

## read_hdf5.py - HDF5 文件查看工具

快速查看 HDF5 文件的数据结构、属性和数据内容。

### 基本用法
```bash
# 基本查看文件结构
python hdf5_tools/read_hdf5.py data.h5

# 显示属性和数据预览
python hdf5_tools/read_hdf5.py data.h5 --attrs --preview

# 限制显示层级（适用于深层嵌套结构）
python hdf5_tools/read_hdf5.py data.h5 --max-level 2
```

### 交互式模式
```bash
python hdf5_tools/read_hdf5.py data.h5 --interactive
```

交互模式可用命令：
- `help` - 显示帮助
- `cd <name>` - 进入组（使用 `..` 返回上级）
- `ls` - 列出当前组内容
- `info <name>` - 显示数据集详细信息
- `preview <name>` - 预览数据集数据
- `pwd` - 显示当前位置
- `exit` 或 `quit` - 退出

---

## split_hdf5.py - HDF5 文件拆分工具

将包含多个 group 的 HDF5 文件拆分成多个单独的文件，每个 group 保存为独立的 HDF5 文件。

### 列出文件中的 groups
```bash
python hdf5_tools/split_hdf5.py list-hdf5-groups --input data.h5
```

### 拆分文件
```bash
# 拆分所有 groups
python hdf5_tools/split_hdf5.py split-hdf5-file \
  --input data.h5 \
  --output ./split_output

# 拆分指定的 groups
python hdf5_tools/split_hdf5.py split-hdf5-file \
  --input data.h5 \
  --output ./split_output \
  --groups episode_0 episode_1

# 添加文件名前缀
python hdf5_tools/split_hdf5.py split-hdf5-file \
  --input data.h5 \
  --output ./split_output \
  --prefix "piper_"

# 覆盖已存在的文件
python hdf5_tools/split_hdf5.py split-hdf5-file \
  --input data.h5 \
  --output ./split_output \
  --overwrite
```

### 短参数形式
```bash
python hdf5_tools/split_hdf5.py split-hdf5-file -i data.h5 -o ./split_output
```

---

## repack_hdf5.py - HDF5 文件重新打包工具

将目录中的多个 HDF5 文件重新划分成包含指定数量 episodes 的 HDF5 文件。

例如：输入目录包含 file1.hdf5 (10 episodes) 和 file2.hdf5 (15 episodes)，指定每文件 5 个 episodes，将生成 5 个输出文件，每个包含 5 个 episodes。

### 分析目录中的 episodes 分布
```bash
python convert_parallel/repack_hdf5.py analyze \
  --input ./data \
  --episodes-per-file 50

# 指定文件匹配模式
python convert_parallel/repack_hdf5.py analyze \
  --input ./data \
  --pattern "*.hdf5" \
  --episodes-per-file 100
```

### 预览重新打包结果
```bash
python convert_parallel/repack_hdf5.py repack \
  --input ./data \
  --output ./repacked \
  --episodes-per-file 50 \
  --dry-run
```

### 执行重新打包
```bash
# 基本用法
python convert_parallel/repack_hdf5.py repack \
  --input ./data \
  --output ./repacked \
  --episodes-per-file 50

# 自定义输出文件名前缀
python convert_parallel/repack_hdf5.py repack \
  --input ./data \
  --output ./repacked \
  --episodes-per-file 100 \
  --prefix "shard_"

# 覆盖已存在的文件
python convert_parallel/repack_hdf5.py repack \
  --input ./data \
  --output ./repacked \
  --episodes-per-file 50 \
  --overwrite

# 短参数形式
python convert_parallel/repack_hdf5.py repack \
  -i ./data \
  -o ./repacked \
  -e 50
```

---

## convert_hdf5_shards.py - HDF5 并行转换工具

使用多进程并行将 HDF5 文件转换为 LeRobot Dataset，每个 worker 处理部分文件并生成独立的 shard，最后可聚合为完整数据集。

### 安装依赖
```bash
pip install datatrove lerobot
```

### 本地并行转换（LocalPipelineExecutor）
```bash
# 基本用法：使用 4 个 worker 并行处理所有 HDF5 文件
python convert_parallel/convert_hdf5_shards.py \
  --hdf5-root ./data \
  --all \
  --repo-id "12e21/bi_piper_parallel" \
  --workers 4

# 指定要处理的文件
python convert_parallel/convert_hdf5_shards.py \
  --hdf5-root ./data \
  --hdf5-files file1.hdf5 file2.hdf5 file3.hdf5 \
  --repo-id "12e21/bi_piper_parallel" \
  --workers 8

# 自定义参数
python convert_parallel/convert_hdf5_shards.py \
  --hdf5-root ./data \
  --all \
  --repo-id "12e21/bi_piper_parallel" \
  --robot-type bi_piper \
  --fps 30 \
  --workers 8 \
  --job-name "convert_bi_piper"
```

### SLURM 集群并行转换
```bash
python convert_parallel/convert_hdf5_shards.py \
  --hdf5-root ./data \
  --all \
  --repo-id "12e21/bi_piper_parallel" \
  --workers 100 \
  --slurm 1 \
  --partition cpu \
  --cpus-per-task 8 \
  --mem-per-cpu 4000M \
  --job-name "convert_bi_piper"
```

### 输出结构
转换后会生成多个 shards：
```
12e21/bi_piper_parallel_world_4_rank_0  # Worker 0 处理的文件
12e21/bi_piper_parallel_world_4_rank_1  # Worker 1 处理的文件
12e21/bi_piper_parallel_world_4_rank_2  # Worker 2 处理的文件
12e21/bi_piper_parallel_world_4_rank_3  # Worker 3 处理的文件
```

### 聚合 Shards
```bash
# 将所有 shards 聚合为一个完整数据集
python convert_parallel/aggregate_hdf5_shards.py \
  --repo-id "12e21/bi_piper_parallel" \
  --num-shards 4
```

### datatrove 日志系统

`convert_hdf5_shards.py` 使用 datatrove 框架进行并行处理，会自动在 `./logs/` 目录下创建日志文件用于**断点续传**和**任务追踪**：

```
logs/convert_hdf5/
├── executor.json          # 执行器配置和状态（workers, tasks, pipeline等）
├── completions/           # 任务完成标记（00000, 00001, ...）
├── logs/                  # 每个 worker 的详细运行日志
└── stats/                 # 处理速度、数据量等统计信息
```

#### 重复运行的问题

如果看到以下提示：
```
Not doing anything as all X tasks have already been completed.
```

说明 datatrove 检测到之前的任务已全部完成。解决方法：

**方法 1：清空日志目录**
```bash
rm -rf ./logs/convert_hdf5/
```

**方法 2：使用不同的 job 名称**
```bash
python convert_parallel/convert_hdf5_shards.py \
  --hdf5-root ./data \
  --all \
  --repo-id "12e21/bi_piper_parallel" \
  --workers 4 \
  --job-name "convert_run2"  # 新日志目录：logs/convert_run2/
```

#### 注意事项
- ✅ 优点：支持断点续传，任务中断后可继续
- ⚠️ 注意：更改 `--workers` 数量时，需要清空日志或使用新的 `--job-name`
- 📁 日志位置：默认在 `./logs/<job-name>/`

---

## aggregate_hdf5_shards.py - Shards 聚合工具

将 `convert_hdf5_shards.py` 生成的多个 shards 聚合为一个完整的 LeRobot Dataset。

### 基本用法
```bash
python convert_parallel/aggregate_hdf5_shards.py \
  --repo-id "12e21/bi_piper_parallel" \
  --num-shards 4
```

### 指定输出数据集名称
```bash
python convert_parallel/aggregate_hdf5_shards.py \
  --repo-id "12e21/bi_piper_parallel" \
  --num-shards 4 \
  --output-repo-id "12e21/bi_piper_final"
```

### SLURM 集群运行
```bash
python convert_parallel/aggregate_hdf5_shards.py \
  --repo-id "12e21/bi_piper_parallel" \
  --num-shards 100 \
  --slurm 1 \
  --partition cpu \
  --job-name "aggregate_bi_piper"
```

### 参数说明
| 参数 | 说明 |
|------|------|
| `--repo-id` | 基础 repo ID（不含 _world_X_rank_Y 后缀）|
| `--num-shards` | Shard 数量（应等于 convert 时的 --workers）|
| `--output-repo-id` | 输出数据集名称（可选，默认使用 --repo-id）|
| `--slurm` | 使用 SLURM（1=启用，0=本地）|
| `--workers` | Worker 数量（聚合应设为 1）|
| `--partition` | SLURM 分区名称 |
| `--cpus-per-task` | 每个 task 的 CPU 数量 |
| `--mem-per-cpu` | 每个 CPU 的内存 |
| `--logs-dir` | 日志目录（默认：./logs）|
| `--job-name` | 任务名称 |

### 完整工作流示例
```bash
# Step 1: 并行转换（100 个 workers）
python convert_parallel/convert_hdf5_shards.py \
  --hdf5-root ./data \
  --all \
  --repo-id "12e21/bi_piper_parallel" \
  --workers 100

# Step 2: 聚合所有 shards
python convert_parallel/aggregate_hdf5_shards.py \
  --repo-id "12e21/bi_piper_parallel" \
  --num-shards 100

# 现在可以使用完整数据集 "12e21/bi_piper_parallel"
```

---

### convert_hdf5_shards.py 参数说明
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
| `--logs-dir` | 日志目录（默认：./logs）|
| `--job-name` | 任务名称 |

---

## lerobot_v30_to_v21.py - LeRobot 数据集版本转换工具

将 LeRobot 数据集从 codebase 版本 v3.0 转换回 v2.1 格式。

### 环境要求
- LeRobot 版本至少需要在 commit `f55c6e89f` 之后
- 已在 LeRobot 版本 0.4.0 (commit: f25ac02) 上测试通过

### 转换说明
脚本会将本地的 v3.0 数据集转换为 v2.1 格式：
- 原 v3.0 路径会被 v2.1 路径覆盖
- 原始 v3.0 数据集会备份到带 `_v30` 后缀的文件夹中

### 基本用法
```bash
# 转换 HuggingFace 上的数据集
python convert_parallel/lerobot_v30_to_v21.py \
  --repo-id "lerobot/pusht"

# 转换本地数据集
python convert_parallel/lerobot_v30_to_v21.py \
  --repo-id "lerobot/pusht" \
  --root "/path/to/datasets"

# 强制重新下载并转换
python convert_parallel/lerobot_v30_to_v21.py \
  --repo-id "lerobot/pusht" \
  --force-conversion
```

### 参数说明
| 参数 | 说明 |
|------|------|
| `--repo-id` | HuggingFace 仓库标识符（必需，例如 `lerobot/pusht`）|
| `--root` | 本地目录，用于存储数据集（可选）|
| `--force-conversion` | 忽略现有本地快照，从 Hub 重新下载（标志位）|
