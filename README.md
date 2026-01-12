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
