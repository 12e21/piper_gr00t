# Piper Gr00t

Adapt piper data to finetune on gr00t n1.6

## HDF5 文件查看工具

### 使用方法

```bash
# 基本查看
python read_hdf5.py <your_file.h5>

# 显示属性和数据预览
python read_hdf5.py <your_file.h5> --attrs --preview

# 限制显示层级
python read_hdf5.py <your_file.h5> --max-level 2

# 交互式模式（推荐）
python read_hdf5.py <your_file.h5> --interactive
```

### 交互式命令

- `help` - 显示帮助
- `cd <name>` - 进入组（`..` 返回上级）
- `ls` - 列出内容
- `info <name>` - 数据集详情
- `preview <name>` - 预览数据
- `pwd` - 显示当前位置
- `exit` - 退出

## HDF5 转 LeRobot 数据集

`hdf2lerobotv21.py` 用于将双臂 Piper 机器人的 HDF5 数据转换为 LeRobot 数据集格式。

### 安装依赖

```bash
uv pip install typer lerobot opencv-python h5py numpy tqdm
```

### 使用方法

```bash
# 处理所有 HDF5 文件
python hdf2lerobotv21.py --all

# 处理指定文件
python hdf2lerobotv21.py --hdf5-files 0000.hdf5 --hdf5-files 0001.hdf5

# 使用自定义参数
python hdf2lerobotv21.py --all \
  --repo-id "your/repo" \
  --robot-type "bi_piper" \
  --fps 30 \
  --hdf5-root "./data"

# 处理并推送到 HuggingFace Hub
python hdf2lerobotv21.py --all --push
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--repo-id` | HuggingFace 仓库 ID | `12e21/bi_piper_subset` |
| `--robot-type` | 机器人类型 | `bi_piper` |
| `--fps` | 视频帧率 | `30` |
| `--hdf5-root` | HDF5 文件根目录 | `./data` |
| `--hdf5-files` | 指定要处理的文件（可多次使用） | - |
| `--all` | 处理根目录下所有 .hdf5 文件 | `false` |
| `--push` | 推送数据集到 HuggingFace Hub | `false` |

### 数据格式

脚本支持以下数据格式：

- **动作**: 14 维关节位置（双臂各 7 个关节）
- **观测**: 14 维关节位置状态
- **图像**: 3 个视角（左手腕、中间、右手腕），480x640 分辨率

## 环境

使用 conda 和 uv，可直接用 `python` 运行脚本。
