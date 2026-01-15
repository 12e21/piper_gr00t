# 使用Piper数据微调GR00T N1.6
## 1. 配置环境
配置hdf5转lerobot dataset v2.1的环境
```bash
git clone https://github.com/12e21/piper_gr00t.git
conda create -n hdf2lerobot python=3.10
conda activate hdf2lerobot
pip install uv
uv pip install lerobot==0.3.3 h5py opencv-python ipdb typer tqdm numpy
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
