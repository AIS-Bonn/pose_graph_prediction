from json import load as load_json_file, dump as save_json_file

from numpy import array, copy, ndarray

from os.path import exists, join

from pose_graph_tracking.data.normalization import PoseSequenceNormalizer
from pose_graph_tracking.data.human36m_data_loader import Human36MDataLoader

from pose_graph_tracking.helpers.defaults import PATH_TO_DATA_DIRECTORY

import torch

from torch_geometric.data import Dataset, Data

from tqdm import tqdm

from typing import List, Tuple, Union


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
        normalizer = PoseSequenceNormalizer()

        number_of_sequences = len(data_loader.sequences)
        sequence_ids_progress_bar = tqdm(range(number_of_sequences))
        sequence_ids_progress_bar.set_description("Progress")
        for sequence_id in sequence_ids_progress_bar:
            sequence = data_loader.sequences[sequence_id]

            last_start_index_for_sampling = len(sequence["estimated_poses"]) - self.sample_sequence_lenght + 1
            for frame in range(last_start_index_for_sampling):
                estimated_poses_sample = copy(sequence["estimated_poses"][frame: frame + self.sample_sequence_lenght])
                ground_truth_sample = copy(sequence["ground_truth_poses"][frame: frame + self.sample_sequence_lenght])
                normalizer.compute_normalization_parameters(estimated_poses_sample)
                normalizer.normalize_pose_sequence(estimated_poses_sample)
                normalizer.normalize_pose_sequence(ground_truth_sample)

                data = self.convert_samples_to_graph_data(estimated_poses_sample,
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

    def convert_samples_to_graph_data(self,
                                      estimated_poses_sample: Union[List[List[Tuple[float, float, float]]], ndarray],
                                      ground_truth_sample: Union[List[List[Tuple[float, float, float]]], ndarray],
                                      action_id: int):
        if len(estimated_poses_sample) != 3:
            print("Data conversion is currently implemented just for a sample length of 3. Exiting.")
            exit(-1)

        number_of_joints = len(estimated_poses_sample[0])
        mean_joint_id = number_of_joints / 2

        # Convert each joint from the latest time step to a node
        features_of_nodes = []
        for joint_id, joint in enumerate(estimated_poses_sample[1]):
            # Normalize joint_id to range from -1 to 1
            normalized_joint_id = (joint_id - mean_joint_id) / mean_joint_id
            node_features = [normalized_joint_id, joint[0], joint[1], joint[2]]
            features_of_nodes.append(node_features)
        features_of_nodes = torch.FloatTensor(array(features_of_nodes))

        # Compute the features of the edges between each node pair
        features_of_edges = []
        source_node_ids_of_edges = []
        target_node_ids_of_edges = []
        for source_joint_id, source_joint in enumerate(estimated_poses_sample[0]):
            for target_joint_id, target_joint in enumerate(estimated_poses_sample[1]):
                source_node_ids_of_edges.append(source_joint_id)
                target_node_ids_of_edges.append(target_joint_id)

                x_difference = target_joint[0] - source_joint[0]
                y_difference = target_joint[1] - source_joint[1]
                z_difference = target_joint[2] - source_joint[2]
                edge_feature = [x_difference, y_difference, z_difference]
                features_of_edges.append(edge_feature)
        features_of_edges = torch.FloatTensor(array(features_of_edges))
        node_ids_connected_by_edges = torch.tensor([source_node_ids_of_edges,
                                                    target_node_ids_of_edges], dtype=torch.long)

        # Convert the ground truth - the states of the joints in the next time step - to the format of the network's
        # output
        ground_truth_node_positions = torch.FloatTensor(array(ground_truth_sample[-1]))

        # TODO: Remove later on
        action_id_tensor = torch.IntTensor(array([action_id]))

        data = Data(x=features_of_nodes,
                    features_of_edges=features_of_edges,
                    node_ids_connected_by_edges=node_ids_connected_by_edges,
                    action_id=action_id_tensor,
                    ground_truth=ground_truth_node_positions)

        print("data ", data)

        print("Not tested yet")
        exit(-1)

        return data
