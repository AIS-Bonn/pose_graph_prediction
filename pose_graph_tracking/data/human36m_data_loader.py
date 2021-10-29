from json import load as load_json_file

from os.path import exists, join

from pose_graph_tracking.data.conversions import convert_estimated_pose_sequence_to_gt_format, \
    convert_action_label_to_action_id

from typing import List, Tuple, Union


class Human36MDataLoader(object):
    """
    Load Human 3.6M data from files to a list of sequences.

    Each sequence consists of a dict with three entries - the action label, a list of estimated poses and a list of
    ground truth poses.

    sequence : dict
        "action_id" : int
        "estimated_poses" : List[Pose] : len() is number of frames in sequence
        "ground_truth_poses" : List[Pose] : len() is number of frames in sequence

    Pose : List[Joint_Position] : len() is currently 17 as there are 17 joints positions in the Human 3.6M format

    Joint_Position : List[float, float, float] : 3D position of the joint

    :param path_to_data_root_directory: path to the directory the data files are saved in.
    :param ids_of_subjects_to_load: a list containing a combination of the subject ids [1, 5, 6, 7, 8, 9, 11] or None to
     load all subjects.
    :param specifically_requested_action_label: use only sequences with this action [Direction, Discuss, Eating, Greet,
     Phone, Pose, Purchase, Sitting, Sitting Down, Smoke, Photo, Wait, Walk, Walk Dog, Walk Together]
    :param frames_to_skip_at_sequence_start: start sequence after skipping that many frames.
    """
    def __init__(self,
                 path_to_data_root_directory: str,
                 ids_of_subjects_to_load: Union[List[int], None] = None,
                 specifically_requested_action_label: Union[str, None] = None,
                 frames_to_skip_at_sequence_start: int = 0):
        # List the loaded sequences are stored in
        self.sequences = []

        if ids_of_subjects_to_load is not None:
            self.ids_of_subjects_to_load = ids_of_subjects_to_load
        else:
            self.ids_of_subjects_to_load = [1, 5, 6, 7, 8, 9, 11]

        self.specifically_requested_action_label = specifically_requested_action_label
        self.frames_to_skip_at_sequence_start = frames_to_skip_at_sequence_start

        # Making sure path ends with a separator
        self.path_to_input_data_root_dir = join(path_to_data_root_directory, "")

        self._load_data()

    def _load_data(self):
        """
        Loads the sequence data for the specified subject ids and stores the sequences in the member self.sequences.
        """
        for subject_id in self.ids_of_subjects_to_load:
            filename_of_current_subject = "keypoints_s" + str(subject_id) + "_h36m.json"
            path_to_current_subject_file = self.path_to_input_data_root_dir + filename_of_current_subject

            self._load_sequences_from_file(path_to_current_subject_file)

    def _load_sequences_from_file(self,
                                  path_to_subject_file: str):
        if exists(path_to_subject_file):
            with open(path_to_subject_file) as json_file:
                subject_data = load_json_file(json_file)
                self._incorporate_data_into_sequences(subject_data)
        else:
            print("Cannot load file: " + str(path_to_subject_file) + "\nContinuing with next file.")

    def _incorporate_data_into_sequences(self,
                                         subject_data: dict):
        number_of_sequences = len(subject_data["action_labels"])
        for current_sequence_id in range(number_of_sequences):
            current_action_label = subject_data["action_labels"][current_sequence_id]

            if self.specifically_requested_action_label is not None and \
                    current_action_label != self.specifically_requested_action_label:
                continue

            current_sequence_data = subject_data["sequences"][current_sequence_id]

            action_id = convert_action_label_to_action_id(current_action_label)

            estimated_pose_sequence = self._extract_estimated_pose_sequence_from_sequence_data(current_sequence_data)
            convert_estimated_pose_sequence_to_gt_format(estimated_pose_sequence)

            if self.is_any_estimated_joint_missing(estimated_pose_sequence):
                continue

            ground_truth_pose_sequence = [frame["labels"]["poses_3d"]
                                          for frame in current_sequence_data[self.frames_to_skip_at_sequence_start:]]

            self.sequences.append({"action_id": action_id,
                                   "estimated_poses": estimated_pose_sequence,
                                   "ground_truth_poses": ground_truth_pose_sequence})

    def _extract_estimated_pose_sequence_from_sequence_data(self,
                                                            sequence_data: List[dict]):
        estimated_poses = [frame["poses_3d_filter"] for frame in sequence_data[self.frames_to_skip_at_sequence_start:]
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

