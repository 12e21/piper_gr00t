import h5py
import numpy as np
import os
import cv2
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm

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
        }
        dataset.add_frame(frame=frame, task=episode_instruction)

    return True
            

def convert_hdf5_to_lerobot():
    repo_id = "12e21/bi_piper_subset"
    robot_type = "bi_piper"
    fps = 30
    hdf5_root = "./data"
    hdf5_files = ["0000.hdf5", "0001.hdf5"]
    push_to_hub = False

    hdf5_files = [os.path.join(hdf5_root, file) for file in hdf5_files]
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
    convert_hdf5_to_lerobot()