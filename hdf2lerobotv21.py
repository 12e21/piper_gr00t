import h5py
import numpy as np
import os
import cv2
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm
import typer
from pathlib import Path
from typing import Optional, List

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
    episode_instruction = episode_group.attrs.get("instruction")
    episode_frame_length = episode_group.attrs.get("length")

    action = np.array(episode_group["action"])
    state = np.array(episode_group["state"])
    image_left_bytes = episode_group["image_left"][()]
    image_mid_bytes = episode_group["image_mid"][()]
    image_right_bytes = episode_group["image_right"][()]
    

    decode = lambda x: x if isinstance(x, np.ndarray) and x.ndim == 3 else cv2.imdecode(np.frombuffer(x, np.uint8), cv2.IMREAD_COLOR)
    for frame_index in range(episode_frame_length):
        image_left = decode(image_left_bytes[frame_index])
        image_mid = decode(image_mid_bytes[frame_index])
        image_right = decode(image_right_bytes[frame_index])

        frame = {
            "action": action[frame_index],
            "observation.state": state[frame_index],
            "observation.images.left_wrist": image_left,
            "observation.images.mid": image_mid,
            "observation.images.right_wrist": image_right,
            "task": episode_instruction,
        }
        dataset.add_frame(frame=frame)

    return True
            

def convert_hdf5_to_lerobot(
    repo_id: str = typer.Option("12e21/bi_piper_subset", help="HuggingFace repository ID"),
    robot_type: str = typer.Option("bi_piper", help="Robot type"),
    fps: int = typer.Option(30, help="Frames per second"),
    hdf5_root: str = typer.Option("./data", help="Root directory for HDF5 files"),
    hdf5_files: Optional[List[str]] = typer.Option(None, help="HDF5 files to process (can be specified multiple times)"),
    all_files: bool = typer.Option(False, "--all", help="Process all HDF5 files in the root directory"),
    push_to_hub: bool = typer.Option(False, "--push", help="Push dataset to HuggingFace Hub"),
) -> None:
    """
    Convert HDF5 files to LeRobot dataset format.
    """
    # Handle file selection
    if all_files:
        hdf5_files = sorted([str(f) for f in Path(hdf5_root).glob("*.hdf5")])
        typer.echo(f"Found {len(hdf5_files)} HDF5 files in {hdf5_root}")
    elif hdf5_files:
        hdf5_files = [os.path.join(hdf5_root, file) for file in hdf5_files]
    else:
        typer.echo("Error: Please specify --hdf5-files or use --all to process all files", err=True)
        raise typer.Exit(1)
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        robot_type=robot_type,
        features=BI_PIPER_FEATURES,
    )

    for hdf5_file in tqdm(hdf5_files, desc="Processing HDF5 files"):
        with h5py.File(hdf5_file, "r") as f:
            for episode_name in tqdm(f.keys(), desc="Processing episodes"):
                episode_group = f[episode_name]
                process_data(dataset, episode_group, episode_name)
                dataset.save_episode()

    if push_to_hub:
        dataset.push_to_hub()

if __name__ == "__main__":
    typer.run(convert_hdf5_to_lerobot)