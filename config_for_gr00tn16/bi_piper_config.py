from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


bi_piper_config = {
    "video": ModalityConfig(
        delta_indices=[0], # 只使用当前帧
        modality_keys=["left_wrist", "mid", "right_wrist"], # 与modality.json中保持一致
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[ # 只使用当前帧
            "left_arm",
            "left_gripper",
            "right_arm",
            "right_gripper",
        ],
    ),
    "action": ModalityConfig(
        # Action Chunk
        delta_indices=[
            0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
        ],
        modality_keys=[
            "left_arm",
            "left_gripper",
            "right_arm",
            "right_gripper"
        ],
        action_configs=[
            # arm 使用相对action表示，gripper 使用绝对action表示
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            )
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(bi_piper_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)

