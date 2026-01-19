"""
聚合 HDF5 并行转换生成的 shards 成一个完整的 LeRobot Dataset
"""

import argparse
import logging

from lerobot.datasets.aggregate import aggregate_datasets
from lerobot.utils.utils import init_logging


def main():
    parser = argparse.ArgumentParser(
        description="聚合 HDF5 并行转换生成的 shards 成一个完整的 LeRobot Dataset"
    )

    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="基础 repository ID（不包含 _world_X_rank_Y 后缀）",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        required=True,
        help="Shard 的数量（应该等于 convert_hdf5_shards.py 中的 --workers 数量）",
    )
    parser.add_argument(
        "--output-repo-id",
        type=str,
        default=None,
        help="输出数据集的 repo_id（默认使用 --repo-id 的值）",
    )

    args = parser.parse_args()

    # 初始化日志
    init_logging()

    # 构造 shard repo_ids
    repo_ids = [f"{args.repo_id}_world_{args.num_shards}_rank_{rank}" for rank in range(args.num_shards)]

    # 确定输出 repo_id
    output_repo_id = args.output_repo_id if args.output_repo_id else args.repo_id

    # 打印信息
    print(f"📊 Aggregation Configuration:")
    print(f"   Base repo ID: {args.repo_id}")
    print(f"   Number of shards: {args.num_shards}")
    print(f"   Output repo ID: {output_repo_id}")
    print()
    print(f"📁 Shards to aggregate:")
    for repo_id in repo_ids:
        print(f"   - {repo_id}")
    print()

    # 执行聚合
    logging.info(f"Starting aggregation of {len(repo_ids)} datasets into {output_repo_id}")
    aggregate_datasets(repo_ids, output_repo_id)

    print(f"\n✨ Aggregation complete!")
    print(f"Aggregated dataset: {output_repo_id}")
    return 0


if __name__ == "__main__":
    exit(main())
