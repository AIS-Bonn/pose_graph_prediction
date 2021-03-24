from numpy import array, ndarray

from pose_graph_tracking.data.normalization import PoseSequenceNormalizer

import torch

from torch_geometric.data import Data

from typing import List, Optional, Tuple, Union


PoseType = List[Tuple[float, float, float]]
PoseSequenceType = List[PoseType]


def get_features_of_nodes(estimated_pose: Union[PoseType, ndarray]) -> torch.FloatTensor:
    number_of_joints = len(estimated_pose)
    mean_joint_id = number_of_joints / 2

    # Convert each joint from the latest time step to a node
    features_of_nodes = []
    for joint_id, joint in enumerate(estimated_pose):
        # Normalize joint_id to range from -1 to 1
        normalized_joint_id = (joint_id - mean_joint_id) / mean_joint_id
        node_features = [normalized_joint_id, joint[0], joint[1], joint[2]]
        features_of_nodes.append(node_features)
    return torch.FloatTensor(array(features_of_nodes))


def get_features_of_edges(estimated_poses_sample: Union[PoseSequenceType, ndarray]) -> torch.FloatTensor:
    features_of_edges = []
    for source_joint in estimated_poses_sample[0]:
        for target_joint in estimated_poses_sample[1]:
            x_difference = target_joint[0] - source_joint[0]
            y_difference = target_joint[1] - source_joint[1]
            z_difference = target_joint[2] - source_joint[2]
            edge_feature = [x_difference, y_difference, z_difference]
            features_of_edges.append(edge_feature)
    return torch.FloatTensor(array(features_of_edges))


def get_node_ids_connected_by_edges(estimated_poses_sample: Union[PoseSequenceType, ndarray]) -> torch.Tensor:
    source_node_ids_of_edges = []
    target_node_ids_of_edges = []
    for source_joint_id in range(len(estimated_poses_sample[0])):
        for target_joint_id in range(len(estimated_poses_sample[1])):
            source_node_ids_of_edges.append(source_joint_id)
            target_node_ids_of_edges.append(target_joint_id)

    return torch.tensor([source_node_ids_of_edges,
                         target_node_ids_of_edges], dtype=torch.long)


def convert_samples_to_graph_data(estimated_poses_sample: Union[PoseSequenceType, ndarray],
                                  ground_truth_sample: Union[PoseSequenceType, ndarray],
                                  action_id: Optional[int]) -> Data:
    if len(estimated_poses_sample) != 3:
        print("Data conversion is currently implemented just for a sample length of 3. Exiting.")
        exit(-1)

    normalizer = PoseSequenceNormalizer()
    normalizer.compute_normalization_parameters(estimated_poses_sample[0])
    normalizer.normalize_pose_sequence(estimated_poses_sample)
    normalizer.normalize_pose_sequence(ground_truth_sample)

    features_of_nodes = get_features_of_nodes(estimated_poses_sample[1])
    features_of_edges = get_features_of_edges(estimated_poses_sample)
    node_ids_connected_by_edges = get_node_ids_connected_by_edges(estimated_poses_sample)

    # Convert the ground truth - the states of the joints in the next time step - to the format of the network's
    # output
    ground_truth_node_positions = torch.FloatTensor(array(ground_truth_sample[-1]))

    data = Data(x=features_of_nodes,
                features_of_edges=features_of_edges,
                node_indexes_connected_by_edges=node_ids_connected_by_edges,
                # name within Data has to include 'index' in order for collate() to work properly..
                ground_truth=ground_truth_node_positions,
                normalization_offset=torch.FloatTensor(normalizer.offset),
                normalization_scale=torch.FloatTensor(array([normalizer.scale_factor])),
                normalization_rotation_matrix=torch.FloatTensor(normalizer.orientation_normalization_matrix))

    if action_id is not None:
        data["action_id"] = torch.IntTensor(array([action_id]))

    return data


def convert_poses_to_graph_data(previous_estimated_pose: Union[PoseType, ndarray],
                                current_estimated_pose: Union[PoseType, ndarray],
                                ground_truth_next_pose: Union[PoseType, ndarray, None] = None,
                                action_id: Optional[int] = None) -> Data:
    normalizer = PoseSequenceNormalizer()
    normalizer.compute_normalization_parameters(previous_estimated_pose)
    previous_estimated_pose = normalizer.normalize_pose(previous_estimated_pose)
    current_estimated_pose = normalizer.normalize_pose(current_estimated_pose)

    features_of_nodes = get_features_of_nodes(current_estimated_pose)
    features_of_edges = get_features_of_edges([previous_estimated_pose, current_estimated_pose])
    node_ids_connected_by_edges = get_node_ids_connected_by_edges([previous_estimated_pose, current_estimated_pose])

    data = Data(x=features_of_nodes,
                features_of_edges=features_of_edges,
                node_indexes_connected_by_edges=node_ids_connected_by_edges,
                # name within Data has to include 'index' in order for collate() to work properly..
                normalization_offset=torch.FloatTensor(normalizer.offset),
                normalization_scale=torch.FloatTensor(array([normalizer.scale_factor])),
                normalization_rotation_matrix=torch.FloatTensor(normalizer.orientation_normalization_matrix))

    if ground_truth_next_pose is not None:
        ground_truth_next_pose = normalizer.normalize_pose(ground_truth_next_pose)
        data["ground_truth"] = torch.FloatTensor(array(ground_truth_next_pose))

    if action_id is not None:
        data["action_id"] = torch.IntTensor(array([action_id]))

    return data


