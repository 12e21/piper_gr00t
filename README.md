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

## 环境

使用 conda 和 uv，可直接用 `python` 运行脚本。
