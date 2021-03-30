from numpy import array, copy

from pose_graph_tracking.data.dataset_generator_utils import convert_poses_to_graph_data
from pose_graph_tracking.data.human36m_graph_in_memory_dataset_generator import Human36MDataLoader, Human36MDataset

from pose_graph_tracking.helpers.defaults import PATH_TO_DATA_DIRECTORY
from pose_graph_tracking.helpers.human36m_definitions import CONNECTED_JOINTS_PAIRS_FOR_HUMAN36M_GROUND_TRUTH

import torch

from torch_geometric.data import Data

from tqdm import tqdm

from typing import List, Union


class AugmentingHuman36MDataset(Human36MDataset):
    """
    Generates an in memory dataset that augments each requested data sample by applying noise to the joint positions.
    """
    def __init__(self,
                 data_save_directory: str,
                 path_to_data_root_directory: str = PATH_TO_DATA_DIRECTORY + 'original/',
                 ids_of_subjects_to_load: Union[List[int], None] = None,
                 sample_sequence_length: int = 3,
                 percentage_of_link_length_as_noise_limit: float = 0.05):
        self.percentage_of_link_length_as_noise_limit = percentage_of_link_length_as_noise_limit
        super(AugmentingHuman36MDataset, self).__init__(data_save_directory,
                                                        path_to_data_root_directory,
                                                        ids_of_subjects_to_load,
                                                        sample_sequence_length,
                                                        self._transform_data)

    def process(self):
        """
        Is being called if no dataset exists in the specified directory.
        Loads the data and converts each sample sequence into graph data.
        Saves the dataset in the specified directory.
        """
        sample_counter = 0
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
                sample_counter += 1

        self._save_dataset_description_to_file(sample_counter)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    def _process_sample(self,
                        sequence,
                        start_frame_id: int,
                        data_list: List[Data]):
        """
        Converts the necessary sequence sample into a Data object and appends it to the data_list.

        :param sequence: Whole sequence containing all frames.
        :param start_frame_id: The id of the first frame in the sequence sample.
        :param data_list: List storing all converted data objects.
        """
        previous_pose = copy(sequence["estimated_poses"][start_frame_id])
        current_pose = copy(sequence["estimated_poses"][start_frame_id + 1])
        last_frame_id_in_sample = start_frame_id + self.sample_sequence_lenght - 1
        ground_truth_pose = copy(sequence["ground_truth_poses"][last_frame_id_in_sample])

        data = Data(previous_pose=torch.FloatTensor(array(previous_pose)),
                    current_pose=torch.FloatTensor(array(current_pose)),
                    ground_truth_pose=torch.FloatTensor(array(ground_truth_pose)))
        data_list.append(data)

    def _transform_data(self,
                        input_data: Data) -> Data:
        """
        Gets a sequence sample as a Data object, augments it by applying noise and converts it to graph data.
        """
        # Clone data to prevent accumulation if noise, because noise is applied in place
        current_pose = input_data.current_pose.clone()
        self._apply_noise_to_current_pose(current_pose)

        data = convert_poses_to_graph_data(input_data.previous_pose,
                                           current_pose,
                                           input_data.ground_truth_pose)
        return data

    def _apply_noise_to_current_pose(self,
                                     current_pose: torch.Tensor):
        noise = self._compute_noise_tensor(current_pose)
        for joint_id in range(current_pose.shape[0]):
            current_pose[joint_id] = current_pose[joint_id] + noise[joint_id]

    def _compute_noise_tensor(self,
                              current_pose: torch.Tensor) -> torch.Tensor:
        """
        Applies noise to each joint position.
        The amount of noise depends on the length of the link to the preceding joint (the one closer to the root joint
        in the pose graph) and a scaling factor.
        For the root joint the length to the lower back joint is used.

        :param current_pose: Joint positions.
        :return: Noise per joint and axis.
        """
        link_lengths_per_joint = self.compute_preceding_link_lengths_per_joints(current_pose)

        noise = torch.zeros_like(current_pose)
        for joint_id in range(len(link_lengths_per_joint)):
            random_values_between_minus_one_and_one = torch.rand(3) * 2 - 1
            noise[joint_id] = random_values_between_minus_one_and_one * \
                              link_lengths_per_joint[joint_id] * self.percentage_of_link_length_as_noise_limit

        return noise

    def compute_preceding_link_lengths_per_joints(self,
                                                  current_pose: torch.Tensor) -> List[float]:
        # Noise for root joint (mid hip) has to be computed separately, because mid hip is never target_joint_id
        mid_hip_id = 0
        lower_back_id = 7
        length_root_to_lower_back_joint = torch.dist(current_pose[mid_hip_id], current_pose[lower_back_id], 2).item()

        link_lengths_per_joint = [length_root_to_lower_back_joint]
        for joint_id_pairs_per_link in CONNECTED_JOINTS_PAIRS_FOR_HUMAN36M_GROUND_TRUTH:
            source_joint_id = joint_id_pairs_per_link[0]
            target_joint_id = joint_id_pairs_per_link[1]

            distance = torch.dist(current_pose[source_joint_id], current_pose[target_joint_id], 2).item()
            link_lengths_per_joint.append(distance)
        return link_lengths_per_joint
