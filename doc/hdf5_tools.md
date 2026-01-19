# HDF5 基础工具

## read_hdf5.py - HDF5 文件查看工具

快速查看 HDF5 文件的数据结构、属性和数据内容。

### 基本用法

```bash
# 查看文件结构
python hdf5_tools/read_hdf5.py data.h5

# 显示属性和数据预览
python hdf5_tools/read_hdf5.py data.h5 --attrs --preview
```

### 交互式模式

```bash
python hdf5_tools/read_hdf5.py data.h5 --interactive
```

可用命令：`cd`、`ls`、`info`、`preview`、`pwd`、`exit`

---

## split_hdf5.py - HDF5 文件拆分工具

将包含多个 group 的 HDF5 文件拆分成多个单独的文件。

### 基本用法

```bash
# 列出文件中的 groups
python hdf5_tools/split_hdf5.py list-hdf5-groups --input data.h5

# 拆分所有 groups
python hdf5_tools/split_hdf5.py split-hdf5-file --input data.h5 --output ./split_output

# 拆分指定的 groups
python hdf5_tools/split_hdf5.py split-hdf5-file --input data.h5 --output ./split_output --groups episode_0 episode_1
```

### 注意事项

- 使用 `--overwrite` 覆盖已存在的文件
- 使用 `--prefix` 添加文件名前缀
