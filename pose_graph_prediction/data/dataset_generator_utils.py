from numpy import array, ndarray

from pose_graph_prediction.data.normalization import PoseSequenceNormalizer

import torch

from torch_geometric.data import Data

from typing import List, Optional, Tuple, Union


PoseType = List[Tuple[float, float, float]]
PoseSequenceType = List[PoseType]


def get_features_of_nodes(estimated_poses_sample: Union[PoseSequenceType, ndarray]) -> torch.FloatTensor:
    # Convert each joint from the latest time step to a node
    previous_estimated_pose = estimated_poses_sample[0]
    current_estimated_pose = estimated_poses_sample[1]
    # This node feature setup requires estimated poses to have an estimate for every joint in every time step
    if len(previous_estimated_pose) != len(current_estimated_pose):
        print("Pose misses joints")

    features_of_nodes = []
    number_of_joints = len(previous_estimated_pose)
    for joint_id in range(number_of_joints):
        # One-hot encode joint id
        node_features = [0.0] * number_of_joints
        node_features[joint_id] = 1.0
        # Extend with xyz-positions of joint from previous and current time step
        node_features.extend([previous_estimated_pose[joint_id][0],
                              previous_estimated_pose[joint_id][1],
                              previous_estimated_pose[joint_id][2],
                              current_estimated_pose[joint_id][0],
                              current_estimated_pose[joint_id][1],
                              current_estimated_pose[joint_id][2]])
        features_of_nodes.append(node_features)
    return torch.FloatTensor(array(features_of_nodes))


def get_features_of_edges(estimated_poses_sample: Union[PoseSequenceType, ndarray]) -> torch.FloatTensor:
    features_of_edges = []
    number_of_joints_in_previous_pose = len(estimated_poses_sample[0])
    number_of_joints_in_current_pose = len(estimated_poses_sample[1])
    one_hot_encoding_lenght = number_of_joints_in_previous_pose * number_of_joints_in_current_pose
    for source_joint_id in range(number_of_joints_in_previous_pose):
        for target_joint_id in range(number_of_joints_in_current_pose):
            # One-hot encode joint id combinations
            edge_feature = [0.0] * one_hot_encoding_lenght
            id_of_joint_combination = target_joint_id * number_of_joints_in_previous_pose + source_joint_id
            edge_feature[id_of_joint_combination] = 1.0

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

    features_of_nodes = get_features_of_nodes(estimated_poses_sample)
    node_ids_connected_by_edges = get_node_ids_connected_by_edges(estimated_poses_sample)

    # Convert the ground truth - the states of the joints in the next time step - to the format of the network's
    # output
    ground_truth_node_positions = torch.FloatTensor(array(ground_truth_sample[-1]))

    data = Data(x=features_of_nodes,
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

    features_of_nodes = get_features_of_nodes([previous_estimated_pose, current_estimated_pose])
    node_ids_connected_by_edges = get_node_ids_connected_by_edges([previous_estimated_pose, current_estimated_pose])

    data = Data(x=features_of_nodes,
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


