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
并行转换 HDF5 文件到 LeRobot Dataset
每个 worker 处理分配给它的 HDF5 文件，生成独立的 shard
"""

import argparse
import h5py
import numpy as np
import os
import cv2
from pathlib import Path
from typing import Optional, List

from datatrove.executor import LocalPipelineExecutor
from datatrove.executor.slurm import SlurmPipelineExecutor
from datatrove.pipeline.base import PipelineStep

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.utils import init_logging


# feature definition for bi-arm piper data
BI_PIPER_FEATURES = {
    "action": {
        "dtype": "float32",
        "shape": (14,),
        "names": [
            "left_joint_1.pos",
            "left_joint_2.pos",
            "left_joint_3.pos",
            "left_joint_4.pos",
            "left_joint_5.pos",
            "left_joint_6.pos",
            "left_joint_7.pos",
            "right_joint_1.pos",
            "right_joint_2.pos",
            "right_joint_3.pos",
            "right_joint_4.pos",
            "right_joint_5.pos",
            "right_joint_6.pos",
            "right_joint_7.pos",
        ],
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (14,),
        "names": [
            "left_joint_1.pos",
            "left_joint_2.pos",
            "left_joint_3.pos",
            "left_joint_4.pos",
            "left_joint_5.pos",
            "left_joint_6.pos",
            "left_joint_7.pos",
            "right_joint_1.pos",
            "right_joint_2.pos",
            "right_joint_3.pos",
            "right_joint_4.pos",
            "right_joint_5.pos",
            "right_joint_6.pos",
            "right_joint_7.pos",
        ],
    },
    "observation.images.left_wrist": {
        "dtype": "video",
        "shape": [480, 640, 3],
        "names": ["height", "width", "channels"],
        "video_info": {
            "video.height": 480,
            "video.width": 640,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": 30.0,
            "video.channels": 3,
            "has_audio": False,
        },
    },
    "observation.images.mid": {
        "dtype": "video",
        "shape": [480, 640, 3],
        "names": ["height", "width", "channels"],
        "video_info": {
            "video.height": 480,
            "video.width": 640,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": 30.0,
            "video.channels": 3,
            "has_audio": False,
        },
    },
    "observation.images.right_wrist": {
        "dtype": "video",
        "shape": [480, 640, 3],
        "names": ["height", "width", "channels"],
        "video_info": {
            "video.height": 480,
            "video.width": 640,
            "video.codec": "av1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": 30.0,
            "video.channels": 3,
            "has_audio": False,
        },
    },
}


def process_data(dataset: LeRobotDataset, episode_group: h5py.Group, episode_name: str) -> bool:
    """处理单个 episode 的数据"""
    import logging

    episode_instruction = episode_group.attrs.get("instruction")
    episode_frame_length = episode_group.attrs.get("length")

    action = np.array(episode_group["action"])
    state = np.array(episode_group["state"])
    image_left_bytes = episode_group["image_left"][()]
    image_mid_bytes = episode_group["image_mid"][()]
    image_right_bytes = episode_group["image_right"][()]

    for frame_index in range(episode_frame_length):
        image_left = cv2.imdecode(np.frombuffer(image_left_bytes[frame_index], np.uint8), cv2.IMREAD_COLOR)
        image_mid = cv2.imdecode(np.frombuffer(image_mid_bytes[frame_index], np.uint8), cv2.IMREAD_COLOR)
        image_right = cv2.imdecode(np.frombuffer(image_right_bytes[frame_index], np.uint8), cv2.IMREAD_COLOR)

        frame = {
            "action": action[frame_index],
            "observation.state": state[frame_index],
            "observation.images.left_wrist": image_left,
            "observation.images.mid": image_mid,
            "observation.images.right_wrist": image_right,
            "task": episode_instruction,
        }
        dataset.add_frame(frame=frame)

    logging.info(f"Processed episode '{episode_name}' with {episode_frame_length} frames")
    return True


class ConvertHDF5Shards(PipelineStep):
    """
    并行转换 HDF5 文件到 LeRobot Dataset
    每个 worker 处理分配给它的 HDF5 文件
    """

    def __init__(
        self,
        hdf5_files: List[str],
        repo_id: str,
        robot_type: str = "bi_piper",
        fps: int = 30,
    ):
        super().__init__()
        self.hdf5_files = sorted(hdf5_files)
        self.repo_id = repo_id
        self.robot_type = robot_type
        self.fps = fps

    def _allocate_files_by_rank(self, rank: int, world_size: int) -> List[str]:
        """
        根据 rank 分配 HDF5 文件
        使用轮询分配确保负载均衡
        """
        return [f for i, f in enumerate(self.hdf5_files) if i % world_size == rank]

    def run(self, data=None, rank: int = 0, world_size: int = 1):
        import logging

        from datasets.utils.tqdm import disable_progress_bars

        init_logging()
        disable_progress_bars()

        # 生成 shard repo_id
        shard_repo_id = f"{self.repo_id}_world_{world_size}_rank_{rank}"

        logging.info(f"Worker {rank}/{world_size}: Processing shard '{shard_repo_id}'")

        # 分配文件
        files_to_process = self._allocate_files_by_rank(rank, world_size)
        logging.info(f"Worker {rank}: Assigned {len(files_to_process)} files")

        if not files_to_process:
            logging.warning(f"Worker {rank}: No files assigned, skipping")
            return

        # 创建 shard dataset
        dataset = LeRobotDataset.create(
            repo_id=shard_repo_id,
            fps=self.fps,
            robot_type=self.robot_type,
            features=BI_PIPER_FEATURES,
        )

        # 处理分配的文件
        total_episodes = 0
        for hdf5_file in files_to_process:
            logging.info(f"Worker {rank}: Processing {hdf5_file}")
            with h5py.File(hdf5_file, "r") as f:
                for episode_name in f.keys():
                    episode_group = f[episode_name]
                    process_data(dataset, episode_group, episode_name)
                    dataset.save_episode()
                    total_episodes += 1

        logging.info(f"Worker {rank}: Completed processing {total_episodes} episodes from {len(files_to_process)} files")


def make_convert_executor(
    hdf5_files,
    repo_id,
    robot_type,
    fps,
    job_name,
    logs_dir,
    workers,
    partition,
    cpus_per_task,
    mem_per_cpu,
    slurm=True,
):
    """创建并行转换 executor"""
    kwargs = {
        "pipeline": [
            ConvertHDF5Shards(
                hdf5_files=hdf5_files,
                repo_id=repo_id,
                robot_type=robot_type,
                fps=fps,
            ),
        ],
        "logging_dir": str(logs_dir / job_name),
    }

    if slurm:
        kwargs.update(
            {
                "job_name": job_name,
                "tasks": workers,  # 每个 worker 一个 task
                "workers": workers,
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
                "tasks": workers,
                "workers": workers,
            }
        )
        executor = LocalPipelineExecutor(**kwargs)

    return executor


def main():
    parser = argparse.ArgumentParser(
        description="并行转换 HDF5 文件到 LeRobot Dataset"
    )

    parser.add_argument(
        "--hdf5-root",
        type=Path,
        required=True,
        help="Root directory containing HDF5 files",
    )
    parser.add_argument(
        "--hdf5-files",
        nargs="*",
        help="Specific HDF5 files to process (relative to hdf5-root)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all HDF5 files in the root directory",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Repository identifier on Hugging Face (e.g., 'username/dataset_name')",
    )
    parser.add_argument(
        "--robot-type",
        type=str,
        default="bi_piper",
        help="Robot type for the dataset",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second for video data",
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
        default="convert_hdf5",
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
        default=4,
        help="Number of parallel workers",
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

    # Handle file selection
    hdf5_root = Path(args.hdf5_root)
    if args.all:
        hdf5_files = sorted([str(f) for f in hdf5_root.glob("*.hdf5")])
        print(f"Found {len(hdf5_files)} HDF5 files in {hdf5_root}")
    elif args.hdf5_files:
        hdf5_files = [str(hdf5_root / f) for f in args.hdf5_files]
    else:
        print("Error: Please specify --hdf5-files or use --all to process all files")
        parser.print_help()
        return 1

    if not hdf5_files:
        print("Error: No HDF5 files found to process")
        return 1

    # Create logs directory
    args.logs_dir.mkdir(parents=True, exist_ok=True)

    # Create and run executor
    kwargs = {
        "hdf5_files": hdf5_files,
        "repo_id": args.repo_id,
        "robot_type": args.robot_type,
        "fps": args.fps,
        "job_name": args.job_name,
        "logs_dir": args.logs_dir,
        "workers": args.workers,
        "partition": args.partition,
        "cpus_per_task": args.cpus_per_task,
        "mem_per_cpu": args.mem_per_cpu,
        "slurm": args.slurm == 1,
    }

    executor = make_convert_executor(**kwargs)
    executor.run()

    print(f"\n✨ Conversion complete!")
    print(f"Generated {args.workers} shards:")
    for i in range(args.workers):
        print(f"  - {args.repo_id}_world_{args.workers}_rank_{i}")
    print(f"\nNext step: Aggregate shards using aggregate_hdf5_shards.py")
    print(f"Example: python convert_parallel/aggregate_hdf5_shards.py --repo-id {args.repo_id} --num-shards {args.workers}")

    return 0


if __name__ == "__main__":
    exit(main())
