from json import load as load_json_file

from os.path import exists, join

from typing import List, Union


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

    :param ids_of_subjects_to_load: a list containing a combination of the subject ids [1, 5, 6, 7, 8, 9, 11] or None to
     load all subjects
    """
    def __init__(self,
                 path_to_data_root_directory: str,
                 ids_of_subjects_to_load: Union[List[int], None] = None):
        self.sequences = []

        if ids_of_subjects_to_load is not None:
            self.ids_of_subjects_to_load = ids_of_subjects_to_load
        else:
            self.ids_of_subjects_to_load = [1, 5, 6, 7, 8, 9, 11]

        # Make sure path ends with a separator
        self.path_to_input_data_root_dir = join(path_to_data_root_directory, "")

        self._load_data()

    def _load_data(self):
        """
        Loads the sequence data for the specified subject ids and fuses the sequences in a member list.
        """
        for subject_id in self.ids_of_subjects_to_load:
            filename_of_current_subject = "keypoints_s" + str(subject_id) + "_h36m.json"
            path_to_current_subject = self.path_to_input_data_root_dir + filename_of_current_subject

            self._load_sequences(path_to_current_subject)

    def _load_sequences(self,
                        path_to_current_subject: str):
        if exists(path_to_current_subject):
            with open(path_to_current_subject) as json_file:
                subject_data = load_json_file(json_file)
                self._incorporate_data_into_sequences(subject_data)
        else:
            print("Cannot load file: " + str(path_to_current_subject) + "\nContinuing with next file.")

    def _incorporate_data_into_sequences(self,
                                         subject_data: dict):
        self._add_action_labels_to_sequences(subject_data)
        for sequence in subject_data:
            self.sequences.append(sequence)

    def _add_action_labels_to_sequences(self,
                                        sequences_data: dict):
        pass
