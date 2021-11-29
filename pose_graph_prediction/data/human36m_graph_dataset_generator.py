from json import load as load_json_file, dump as save_json_file

from numpy import copy

from os.path import exists, join

from pose_graph_prediction.data.dataset_generator_utils import convert_samples_to_graph_data
from pose_graph_prediction.data.human36m_data_loader import Human36MDataLoader

from pose_graph_prediction.helpers.defaults import PATH_TO_DATA_DIRECTORY

import torch

from torch_geometric.data import Dataset

from tqdm import tqdm

from typing import List, Union


class Human36MDataset(Dataset):
    def __init__(self,
                 data_save_directory: str,
                 path_to_data_root_directory: str = PATH_TO_DATA_DIRECTORY + 'original/',
                 ids_of_subjects_to_load: Union[List[int], None] = None,
                 sample_sequence_length: int = 3):
        self.graphs_filenames = None

        # Making sure path ends with a separator
        self.data_save_directory = join(data_save_directory, "")

        self.path_to_data_root_directory = path_to_data_root_directory
        self.ids_of_subjects_to_load = ids_of_subjects_to_load

        self.sample_sequence_lenght = sample_sequence_length

        # If there is a dataset description file in the save directory, a dataset already exists - regenerate the
        # graphs_filenames to load the data samples
        self.path_to_dataset_description_file = self.data_save_directory + "dataset_description.json"
        if exists(self.path_to_dataset_description_file):
            with open(self.path_to_dataset_description_file) as json_file:
                self.dataset_description = load_json_file(json_file)
                self.graphs_filenames = ['data_{}.pt'.format(i) for i in
                                         range(self.dataset_description["number_of_samples"])]

        super(Human36MDataset, self).__init__(self.data_save_directory,
                                              transform=None,
                                              pre_transform=None)

    @property
    def raw_file_names(self) -> List[str]:
        return []

    @property
    def processed_file_names(self) -> List[str]:
        if self.graphs_filenames is None:
            return []
        else:
            return self.graphs_filenames

    def process(self):
        i = 0
        data_loader = Human36MDataLoader(self.path_to_data_root_directory,
                                         self.ids_of_subjects_to_load)

        number_of_sequences = len(data_loader.sequences)
        sequence_ids_progress_bar = tqdm(range(number_of_sequences))
        sequence_ids_progress_bar.set_description("Progress")
        for sequence_id in sequence_ids_progress_bar:
            sequence = data_loader.sequences[sequence_id]

            last_start_index_for_sampling = len(sequence["estimated_poses"]) - self.sample_sequence_lenght + 1
            for frame in range(last_start_index_for_sampling):
                estimated_poses_sample = copy(sequence["estimated_poses"][frame: frame + self.sample_sequence_lenght])
                ground_truth_sample = copy(sequence["ground_truth_poses"][frame: frame + self.sample_sequence_lenght])

                data = convert_samples_to_graph_data(estimated_poses_sample,
                                                     ground_truth_sample,
                                                     sequence["action_id"])

                torch.save(data, join(self.processed_dir, 'data_{}.pt'.format(i)))
                i += 1

        dataset_description = {"number_of_samples": i,
                               "frames_in_a_sample": self.sample_sequence_lenght,
                               "subject_ids": self.ids_of_subjects_to_load}
        self.graphs_filenames = ['data_{}.pt'.format(i) for i in range(dataset_description["number_of_samples"])]
        with open(self.path_to_dataset_description_file, "w") as outfile:
            save_json_file(dataset_description, outfile, indent=2)

    def len(self):
        return len(self.processed_file_names)

    def get(self,
            idx: int):
        data = torch.load(join(self.processed_dir, 'data_{}.pt'.format(idx)))
        return data
