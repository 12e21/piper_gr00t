#!/usr/bin/env python3

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
聚合 HDF5 并行转换生成的 shards 成一个完整的 LeRobot Dataset
"""

import argparse
import logging
from pathlib import Path

from datatrove.executor import LocalPipelineExecutor
from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.base import PipelineStep

from lerobot.datasets.aggregate import aggregate_datasets
from lerobot.utils.utils import init_logging


class AggregateHDF5Datasets(PipelineStep):
    """
    聚合 HDF5 并行转换生成的所有 shards 成一个完整的 dataset
    """

    def __init__(
        self,
        repo_ids: list[str],
        aggregated_repo_id: str,
    ):
        super().__init__()
        self.repo_ids = repo_ids
        self.aggregated_repo_id = aggregated_repo_id

    def run(self, data=None, rank: int = 0, world_size: int = 1):
        init_logging()

        # Since aggregate_datasets already handles parallel processing internally,
        # we only need one worker to run the entire aggregation
        if rank == 0:
            logging.info(f"Starting aggregation of {len(self.repo_ids)} datasets into {self.aggregated_repo_id}")
            logging.info(f"Shards to aggregate:")
            for repo_id in self.repo_ids:
                logging.info(f"  - {repo_id}")

            aggregate_datasets(self.repo_ids, self.aggregated_repo_id)

            logging.info("Aggregation complete!")
            logging.info(f"Aggregated dataset saved as: {self.aggregated_repo_id}")
        else:
            logging.info(f"Worker {rank} skipping - only worker 0 performs aggregation")


def make_aggregate_executor(
    repo_ids, repo_id, job_name, logs_dir, workers, partition, cpus_per_task, mem_per_cpu, slurm=True
):
    """创建聚合 executor"""
    kwargs = {
        "pipeline": [
            AggregateHDF5Datasets(repo_ids, repo_id),
        ],
        "logging_dir": str(logs_dir / job_name),
    }

    if slurm:
        # For aggregation, we only need 1 task since aggregate_datasets handles everything
        kwargs.update(
            {
                "job_name": job_name,
                "tasks": 1,  # Only need 1 task for aggregation
                "workers": 1,  # Only need 1 worker
                "time": "08:00:00",
                "partition": partition,
                "cpus_per_task": cpus_per_task,
                "sbatch_args": {"mem-per-cpu": mem_per_cpu},
            }
        )
        executor = SlurmPipelineExecutor(**kwargs)
    else:
        kwargs.update(
            {
                "tasks": 1,
                "workers": 1,
            }
        )
        executor = LocalPipelineExecutor(**kwargs)

    return executor


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
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("./logs"),
        help="Path to logs directory for datatrove",
    )
    parser.add_argument(
        "--job-name",
        type=str,
        default="aggregate_hdf5",
        help="Job name used in slurm/logs",
    )
    parser.add_argument(
        "--slurm",
        type=int,
        default=0,
        help="Launch over slurm. Use --slurm 1 to enable, --slurm 0 for local",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of slurm workers. For aggregation, this should be 1.",
    )
    parser.add_argument(
        "--partition",
        type=str,
        help="Slurm partition (e.g., 'cpu')",
    )
    parser.add_argument(
        "--cpus-per-task",
        type=int,
        default=8,
        help="Number of CPUs per task",
    )
    parser.add_argument(
        "--mem-per-cpu",
        type=str,
        default="4000M",
        help="Memory per CPU (e.g., '4000M')",
    )

    args = parser.parse_args()

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

    # Create logs directory
    args.logs_dir.mkdir(parents=True, exist_ok=True)

    # Create and run executor
    kwargs = {
        "repo_id": output_repo_id,
        "job_name": args.job_name,
        "logs_dir": args.logs_dir,
        "workers": args.workers,
        "partition": args.partition,
        "cpus_per_task": args.cpus_per_task,
        "mem_per_cpu": args.mem_per_cpu,
        "slurm": args.slurm == 1,
    }

    executor = make_aggregate_executor(repo_ids, **kwargs)
    executor.run()

    print(f"\n✨ Aggregation complete!")
    print(f"Aggregated dataset: {output_repo_id}")
    return 0


if __name__ == "__main__":
    exit(main())
