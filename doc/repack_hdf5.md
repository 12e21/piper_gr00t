# repack_hdf5.py - HDF5 文件重新打包工具

将目录中的多个 HDF5 文件重新划分成包含指定数量 episodes 的 HDF5 文件。

**用途**：将零散的 HDF5 文件按指定 episodes 数量重新打包，便于后续并行转换。

### 基本用法

```bash
# 分析目录中的 episodes 分布
python convert_parallel/repack_hdf5.py analyze --input ./data --episodes-per-file 50

# 预览重新打包结果
python convert_parallel/repack_hdf5.py repack --input ./data --output ./repacked --episodes-per-file 50 --dry-run

# 执行重新打包
python convert_parallel/repack_hdf5.py repack --input ./data --output ./repacked --episodes-per-file 50
```

### 参数说明

| 参数 | 说明 |
|------|------|
| `--input` | 输入目录 |
| `--output` | 输出目录 |
| `--episodes-per-file` | 每个输出文件包含的 episodes 数量 |
| `--pattern` | 文件匹配模式（默认：*.hdf5）|
| `--prefix` | 输出文件名前缀 |
| `--overwrite` | 覆盖已存在的文件 |

### 注意事项

- `--episodes-per-file` 的选择应考虑并行转换时的 worker 数量，建议每个 worker 处理多个文件
