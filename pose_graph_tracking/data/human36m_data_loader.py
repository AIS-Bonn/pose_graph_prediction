from json import load as load_json_file

from os.path import exists, join

from typing import List, Union


class Human36MDataLoader(object):
    """
    Load Human 3.6M data from files to a list of sequences.

    TODO: describe how sequences are defined
    """
    def __init__(self,
                 path_to_data_root_directory: str,
                 ids_of_subjects_to_load: Union[List[int], None] = None):
        self.sequences = []

        # Make sure path ends with a separator
        self.path_to_input_data_root_dir = join(path_to_data_root_directory, "")

        self._load_data_from_files(ids_of_subjects_to_load)

    def _load_data_from_files(self,
                              ids_of_subjects_to_load: Union[List[int], None] = None):
        """
        Loads the sequence data for the specified subject ids and fuses the sequences in a member list.

        :param ids_of_subjects_to_load: a list containing a combination of the subject ids [1 ,5, 6, 7, 8, 9, 11]
        """
        for subject_id in ids_of_subjects_to_load:
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
