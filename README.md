# 使用Piper数据微调GR00T N1.6
## 1. 配置环境
配置hdf5转lerobot dataset v2.1的环境
```bash
git clone https://github.com/12e21/piper_gr00t.git
conda create -n hdf2lerobot python=3.10
conda activate hdf2lerobot
conda install ffmpeg=7 -c conda-forge
pip install uv
uv pip install git+https://github.com/huggingface/lerobot # 请安装lerobot>=0.4.3, 0.4.2版本的aggregate函数存在bug
uv pip install h5py opencv-python ipdb typer tqdm numpy datatrove
```

## 2. 使用脚本转换hdf5文件

```bash
# 转换目录下所有hdf5文件
python hdf2lerobotv21.py --all \
  --repo-id "your/repo" \
  --hdf5-root "./data" \
  --push  # 上传到huggingface
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

---

# 并行转换数据集

适用于需要转换大量 HDF5 文件的场景，通过多进程并行处理提高转换速度。

## 转换流程

```bash
# Step 1: 重新打包 HDF5 文件（可选）
# 目的：保证 HDF5 文件数量大于并行 worker 数量
python convert_parallel/repack_hdf5.py repack \
  --input ./data \
  --output ./repacked \
  --episodes-per-file 50

# Step 2: 并行转换为 LeRobot Dataset shards
python convert_parallel/convert_hdf5_shards.py \
  --hdf5-root ./repacked \
  --all \
  --repo-id "your/repo" \
  --workers 100

# Step 3: 聚合 shards 为完整数据集
# 注意：并行转换会产生大量数据集碎片，aggregate 不会删除这些碎片
python convert_parallel/aggregate_hdf5_shards.py \
  --repo-id "your/repo" \
  --num-shards 100

# Step 4: 转换数据集版本（可选）
# 如果需要将 LeRobot v3.0 数据集转换为 v2.1 格式
python convert_parallel/lerobot_v30_to_v21.py --repo-id "your/repo"
```

## 工具说明

| 工具 | 用途 |
|------|------|
| `repack_hdf5.py` | 重新打包 HDF5 文件，便于并行处理 |
| `convert_hdf5_shards.py` | 多进程并行转换为 LeRobot Dataset shards |
| `aggregate_hdf5_shards.py` | 聚合 shards 为完整数据集 |

## 性能测试

测试数据：10 个 HDF5 文件（100 episodes）

| 步骤 | 配置 | 用时 |
|------|------|------|
| Repack 10×10 → 2×50 episodes | - | 12 min |
| Convert shards (50 workers) | 55核CPU, 500GB内存 | ~9 min |
| Aggregate shards | 55核CPU, 500GB内存 | ~1 min |
---

# 详细文档

- [HDF5 基础工具](doc/hdf5_tools.md) - `read_hdf5.py`、`split_hdf5.py`
- [HDF5 重新打包工具](doc/repack_hdf5.md) - `repack_hdf5.py`
- [HDF5 并行转换工具](doc/convert_hdf5_shards.md) - `convert_hdf5_shards.py`
- [Shards 聚合工具](doc/aggregate_hdf5_shards.md) - `aggregate_hdf5_shards.py`
- [LeRobot 版本转换工具](doc/lerobot_version_converter.md) - `lerobot_v30_to_v21.py`
