from os import listdir
from os.path import exists, join

from pose_graph_tracking.data.conversions import convert_estimated_pose_sequence_to_gt_format, \
    convert_action_label_to_action_id

import rosbag

from typing import Any, Dict, List, Tuple, Union


Joint_Position = Tuple[float, float, float]
Pose = List[Joint_Position]
Frame = Dict[str, Union[float, List[Pose]]]
Sequence = List[Frame]


class ROSBagDataLoader(object):
    """
    Load estimated pose data from all ros bags within a directory to a list of sequences.
    Each sequence frame consists of a time_stamp and a list of estimated poses - ideally one for each visible human.

    Sequence : List[Frame] : len() is number of frames in the sequence
    # TODO: how do I want to treat frames without visible humans - probably it's best to save them as well and handle
    #  those cases later, because this could be an important information as well

    Frame : dict{
        "time_stamp" : float : time passed since first frame in seconds
        "poses" : List[Pose] : len() is number of humans a pose could be estimated for }

    Pose : List[Joint_Position] : len() is currently 17 as there are 17 joints positions in the Human 3.6M format

    Joint_Position : List[float, float, float] : 3D position of the joint

    :param path_to_bag_directory: path to the directory the bag files are saved in.
    """
    def __init__(self,
                 path_to_bag_directory: str,
                 score_threshold: float = 0.01):
        assert exists(path_to_bag_directory), "Path to bag directory does not exist."
        assert score_threshold >= 0.0, "Score threshold has to be at least zero."

        self.sequences = []
        self.number_of_incomplete_poses = 0

        # Making sure path ends with a separator
        self.path_to_input_data_root_dir = join(path_to_bag_directory, "")
        self.score_threshold = score_threshold

        self.list_of_bag_file_names = self._get_all_bag_file_paths_from_directory()

        self._load_data()

    def _load_data(self):
        """
        Loads the sequence data from the bag files and stores the sequences in the member self.sequences.
        """
        for path_to_current_bag_file in self.list_of_bag_file_names:
            self._load_sequences_from_file(path_to_current_bag_file)

    def _get_all_bag_file_paths_from_directory(self) -> List[str]:
        list_of_bag_file_names = []
        for file_name in listdir(self.path_to_input_data_root_dir):
            if file_name.endswith(".bag"):
                list_of_bag_file_names.append(join(self.path_to_input_data_root_dir, file_name))

        return list_of_bag_file_names

    def _load_sequences_from_file(self,
                                  path_to_bag_file: str):
        sequence = []
        for topic, msg, t in rosbag.Bag(path_to_bag_file).read_messages():
            if topic == "/human_pose_estimation/persons3d_fused":
                self._incorporate_frame_into_sequence(msg, sequence)
        self.sequences.append(sequence)

    def _incorporate_frame_into_sequence(self,
                                         poses_message: Any,
                                         sequence: Sequence):
        # TODO: continue from here
        frame = {"time_stamp": self.ros_stamp_to_float_stamp(poses_message.header.stamp),
                 "poses": []}

        number_of_humans = len(poses_message.persons)
        for current_human_id in range(number_of_humans):
            pose = []
            joint_missing = False
            for joint_id, joint in enumerate(poses_message.persons[current_human_id].keypoints):
                if joint.score > self.score_threshold:
                    joint_position = [joint.joint.x,
                                      joint.joint.y,
                                      joint.joint.z]
                    pose.append(joint_position)
                elif joint_id < 19:
                    joint_missing = True
                    print(str(frame["time_stamp"]) + " : joint with id " + str(len(pose)) + " missing")
                    break

            # FIXME: Currently we only use poses without missing joints
            if joint_missing:
                self.number_of_incomplete_poses += 1
                break

            frame["poses"].append(pose)

        # TODO: convert arrangement of joints to previously used?

        sequence.append(frame)

    def ros_stamp_to_float_stamp(self, ros_stamp) -> float:
        return ros_stamp.secs + ros_stamp.nsecs / 1000000000.0

    def _extract_estimated_pose_sequence_from_sequence_data(self,
                                                            sequence_data: List[dict]):
        estimated_poses = [frame["poses_3d_filter"] for frame in sequence_data
                           if frame["poses_3d_filter"] is not None]
        number_of_missing_frames = len(sequence_data) - len(estimated_poses)
        if number_of_missing_frames > 0:
            print('{} estimated frames are missing/None!'.format(number_of_missing_frames))
        return estimated_poses

    def is_any_estimated_joint_missing(self,
                                       estimated_pose_sequence: List[List[Tuple[float, float, float]]]) -> bool:
        for frame in estimated_pose_sequence:
            for joint_position in frame:
                if joint_position[0] == 0.0 and joint_position[1] == 0.0 and joint_position[2] == 0.0:
                    print("Sequence contains missing estimated joint -> sequence is skipped.")
                    return True
        return False


if __name__ == '__main__':
    # TODO: visualize data 
    data_loader = ROSBagDataLoader("/home/razlaw/bags/human_pose_tracking")
    print("number of frames ", len(data_loader.sequences[0]))
    print("skipped frames ", data_loader.number_of_incomplete_poses)
    print("first frame ", data_loader.sequences[0][0])
