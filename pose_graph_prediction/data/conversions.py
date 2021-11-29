from pose_graph_prediction.helpers.human36m_definitions import JOINT_MAPPING_FROM_GT_TO_ESTIMATION, ACTION_IDS_DICT

from typing import List, Tuple


def convert_estimated_pose_frame_to_gt_format(estimated_pose: List[Tuple[float, float, float]]):
    converted_pose = []
    for gt_joint_id, estimation_joint_id in JOINT_MAPPING_FROM_GT_TO_ESTIMATION:
        converted_pose.append(estimated_pose[estimation_joint_id])
    return converted_pose


def convert_estimated_pose_sequence_to_gt_format(estimated_pose_sequence: List[List[Tuple[float, float, float]]]):
    for frame_id, frame in enumerate(estimated_pose_sequence):
        estimated_pose_sequence[frame_id] = convert_estimated_pose_frame_to_gt_format(frame)


def convert_action_label_to_action_id(action_label: str) -> int:
    if action_label not in ACTION_IDS_DICT.keys():
        raise ValueError("Unrecognized action: %s" % action_label)
    else:
        return ACTION_IDS_DICT[action_label]
