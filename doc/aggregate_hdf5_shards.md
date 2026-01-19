# aggregate_hdf5_shards.py - Shards 聚合工具

将 `convert_hdf5_shards.py` 生成的多个 shards 聚合为一个完整的 LeRobot Dataset。

### 基本用法

```bash
# 聚合 shards（输出 repo_id 默认与 --repo-id 相同）
python convert_parallel/aggregate_hdf5_shards.py \
  --repo-id "your/repo" \
  --num-shards 4

# 指定输出数据集名称
python convert_parallel/aggregate_hdf5_shards.py \
  --repo-id "your/repo" \
  --num-shards 4 \
  --output-repo-id "your/final_repo"
```

### 参数说明

| 参数 | 说明 |
|------|------|
| `--repo-id` | 基础 repo ID（不含 _world_X_rank_Y 后缀）|
| `--num-shards` | Shard 数量（应等于 convert 时的 --workers）|
| `--output-repo-id` | 输出数据集名称（可选，默认使用 --repo-id）|

### 完整工作流示例

```bash
# Step 1: 并行转换（100 个 workers）
python convert_parallel/convert_hdf5_shards.py \
  --hdf5-root ./data \
  --all \
  --repo-id "your/repo" \
  --workers 100

# Step 2: 聚合所有 shards
python convert_parallel/aggregate_hdf5_shards.py \
  --repo-id "your/repo" \
  --num-shards 100

# 现在可以使用完整数据集 "your/repo"
```
