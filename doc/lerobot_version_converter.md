# lerobot_v30_to_v21.py - LeRobot 数据集版本转换工具

将 LeRobot 数据集从 codebase 版本 v3.0 转换回 v2.1 格式。

**环境要求**：LeRobot 版本至少需要在 commit `f55c6e89f` 之后（已在 0.4.0 测试通过）

### 基本用法

```bash
# 转换 HuggingFace 上的数据集
python convert_parallel/lerobot_v30_to_v21.py --repo-id "lerobot/pusht"

# 转换本地数据集
python convert_parallel/lerobot_v30_to_v21.py --repo-id "lerobot/pusht" --root "/path/to/datasets"

# 强制重新下载并转换
python convert_parallel/lerobot_v30_to_v21.py --repo-id "lerobot/pusht" --force-conversion
```

### 参数说明

| 参数 | 说明 |
|------|------|
| `--repo-id` | HuggingFace 仓库标识符（必需）|
| `--root` | 本地目录，用于存储数据集（可选）|
| `--force-conversion` | 忽略现有本地快照，从 Hub 重新下载（标志位）|

### 注意事项

- 原 v3.0 路径会被 v2.1 路径覆盖
- 原始 v3.0 数据集会备份到带 `_v30` 后缀的文件夹中
