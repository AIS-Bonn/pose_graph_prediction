from json import load as load_json_file, dump as save_json_file

from numpy import copy

from os.path import exists, join

from pose_graph_tracking.data.dataset_generator_utils import convert_samples_to_graph_data
from pose_graph_tracking.data.human36m_data_loader import Human36MDataLoader

from pose_graph_tracking.helpers.defaults import PATH_TO_DATA_DIRECTORY

import torch

from torch_geometric.data import Data, InMemoryDataset

from tqdm import tqdm

from typing import Callable, List, Optional, Union


class Human36MDataset(InMemoryDataset):
    def __init__(self,
                 data_save_directory: str,
                 path_to_data_root_directory: str = PATH_TO_DATA_DIRECTORY + 'original/',
                 ids_of_subjects_to_load: Union[List[int], None] = None,
                 sample_sequence_length: int = 3,
                 transform_method: Optional[Callable] = None):
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

        super(Human36MDataset, self).__init__(self.data_save_directory,
                                              transform=transform_method,
                                              pre_transform=None)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        return []

    @property
    def processed_file_names(self) -> List[str]:
        return ['data.pt']

    def process(self):
        i = 0
        data_loader = Human36MDataLoader(self.path_to_data_root_directory,
                                         self.ids_of_subjects_to_load)

        data_list = []
        number_of_sequences = len(data_loader.sequences)
        sequence_ids_progress_bar = tqdm(range(number_of_sequences))
        sequence_ids_progress_bar.set_description("Progress")
        for sequence_id in sequence_ids_progress_bar:
            sequence = data_loader.sequences[sequence_id]

            last_start_index_for_sampling = len(sequence["estimated_poses"]) - self.sample_sequence_lenght + 1
            for frame in range(last_start_index_for_sampling):
                self._process_sample(sequence, frame, data_list)
                i += 1

        self._save_dataset_description_to_file(i)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def _save_dataset_description_to_file(self,
                                          number_of_samples: int):
        dataset_description = {"number_of_samples": number_of_samples,
                               "frames_in_a_sample": self.sample_sequence_lenght,
                               "subject_ids": self.ids_of_subjects_to_load}
        with open(self.path_to_dataset_description_file, "w") as outfile:
            save_json_file(dataset_description, outfile, indent=2)

    def _process_sample(self,
                        sequence,
                        start_frame_id: int,
                        data_list: List[Data]):
        estimated_poses_sample = copy(sequence["estimated_poses"][start_frame_id:
                                                                  start_frame_id + self.sample_sequence_lenght])
        ground_truth_sample = copy(sequence["ground_truth_poses"][start_frame_id:
                                                                  start_frame_id + self.sample_sequence_lenght])

        data = convert_samples_to_graph_data(estimated_poses_sample,
                                             ground_truth_sample,
                                             sequence["action_id"])
        data_list.append(data)
