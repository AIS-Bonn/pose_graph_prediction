from math import degrees

import numpy as np

from pose_graph_tracking.data.utils import get_angle_2d, get_angle_3d, get_rotation_matrix_around_z_axis

from typing import List, Tuple


class PoseSequenceNormalizer(object):
    def normalize_poses(self,
                        pose_sequence: List[List[Tuple[float, float, float]]]):
        """
        We normalize the pose sequence by transforming it into a local coordinate system.
        This system is centered at the mid hip position of the first frame.
        The z axis points into the opposite direction of the gravitational force.
        The y axis points into the direction from the mid hip point to the left hip point.
        The x axis is directed accordingly, roughly into the direction the front of the hip is pointing to when the
        person is standing.
        Additionally the poses are scaled by dividing by the estimated height of the person - assuming a small person
        (e.g. a child) moves roughly as a large person.
        """
        if len(pose_sequence) == 0:
            print("Sequence does not contain any poses.")
            exit(-1)

        first_mid_hip_position = np.array(pose_sequence[0][0])
        person_height = self._estimate_person_height(pose_sequence[0])
        rotation_matrix_around_z_axis = self._compute_normalization_rotation_matrix(pose_sequence[0])

        for frame_id, current_pose in enumerate(pose_sequence):
            pose_centered_at_mid_hip = np.array(current_pose) - first_mid_hip_position
            pose_centered_at_mid_hip = pose_centered_at_mid_hip / person_height
            normalized_pose = np.matmul(rotation_matrix_around_z_axis, pose_centered_at_mid_hip.transpose()).transpose()
            pose_sequence[frame_id] = normalized_pose

    def _compute_normalization_rotation_matrix(self,
                                               reference_pose: List[Tuple[float, float, float]]) -> np.ndarray:
        """
        Computes the rotation around the z-axis in order to rotate the reference pose in such a way that:
          (- The z axis points into the opposite direction of the gravitational force.)
          - The y axis points into the direction from the mid hip point to the left hip point when the person is
            standing or laying on the back or front and pointing from the mid hip point to the belly when the person is
            laying on its side.
          - The x axis is directed accordingly - roughly into the direction the body of the person is facing to, except
            when laying on the ground, because then the x axis and the z axis would overlap. That's why it faces to the
            persons feet when it's laying on its back and to its head when laying on its front.

        FIXME: If the person is laying on its back it gets aligned with its feet towards the x-axis (when laying on
         its face it gets aligned with its head to the x-axis - which is probably useful for the network). But when
         the person is laying on its side it gets aligned with the y-axis facing towards the x-axis. There is a hard
         switch when a person moves from laying on its back/front to laying on its side.. This is probably tricky to
         learn for a network.

        :param reference_pose: Pose of the person used to compute the rotation matrix.
        :return: Rotation matrix for normalization of the person orientation.
        """
        first_mid_hip_position = np.array(reference_pose[0])
        first_left_hip_position = np.array(reference_pose[4])
        if self._is_person_laying_on_its_side(first_mid_hip_position, first_left_hip_position):
            return self._get_normalization_matrix_for_person_laying_on_its_side(reference_pose)
        else:
            return self._get_normalisation_matrix_for_person_not_laying_on_its_side(first_left_hip_position,
                                                                                    first_mid_hip_position)

    def _is_person_laying_on_its_side(self,
                                      mid_hip_position: np.ndarray,
                                      left_hip_position: np.ndarray) -> bool:
        """
        Check whether the angle between the hips and the z-axis is smaller than 30 degrees.
        If that's the case the person is either laying on its side or on the best way getting there.
        """
        vector_mid_to_left_hip = left_hip_position - mid_hip_position
        angle_from_left_hip_to_z_axis = degrees(get_angle_3d(vector_mid_to_left_hip, np.array([0, 0, 1])))
        if angle_from_left_hip_to_z_axis < 30.0 or angle_from_left_hip_to_z_axis > 150.0:
            return True
        else:
            return False

    def _get_normalization_matrix_for_person_laying_on_its_side(self,
                                                                pose: List[Tuple[float, float, float]]) -> np.ndarray:
        """
        Calculate the angle to rotate the laying person around the z axis in order for it to face into the direction of
        the x-axis.
        If person is laying on its left side, compute angle between vector from midhip to belly and y-axis.
        Else compute the angle between the vector from the mid hip to the belly and the negative y-axis.

        :param pose: Pose of the person used to compute the rotation matrix.
        :return: Rotation matrix for normalization of the person orientation.
        """
        mid_hip_position_xy = np.array(pose[0][0:2])
        belly_position_xy = np.array(pose[7][0:2])
        vector_mid_hip_to_belly_xy = belly_position_xy - mid_hip_position_xy
        y_axis_vector_xy = np.array([0.0, 1.0])
        if self._is_person_laying_on_left_side(pose):
            angle_from_left_hip_to_y_axis = get_angle_2d(vector_mid_hip_to_belly_xy, y_axis_vector_xy)
            return get_rotation_matrix_around_z_axis(angle_from_left_hip_to_y_axis)
        else:
            angle_from_left_hip_to_neg_y_axis = get_angle_2d(vector_mid_hip_to_belly_xy, -y_axis_vector_xy)
            return get_rotation_matrix_around_z_axis(angle_from_left_hip_to_neg_y_axis)

    def _get_normalisation_matrix_for_person_not_laying_on_its_side(self,
                                                                    first_left_hip_position: Tuple[float, float, float],
                                                                    first_mid_hip_position: Tuple[float, float, float]
                                                                    ) -> np.ndarray:
        mid_hip_position_xy = np.array(first_mid_hip_position[0:2])
        left_hip_position_xy = np.array(first_left_hip_position[0:2])
        vector_mid_to_left_hip_xy = left_hip_position_xy - mid_hip_position_xy
        y_axis_vector_xy = np.array([0.0, 1.0])
        angle_from_left_hip_to_y_axis = get_angle_2d(vector_mid_to_left_hip_xy, y_axis_vector_xy)
        return get_rotation_matrix_around_z_axis(angle_from_left_hip_to_y_axis)

    def _is_person_laying_on_left_side(self,
                                       pose: List[Tuple[float, float, float]]):
        mid_hip_position = pose[0]
        left_hip_position = pose[4]
        if left_hip_position[2] < mid_hip_position[2]:
            return True
        return False

    def _estimate_person_height(self,
                                pose: List[Tuple[float, float, float]]) -> float:
        """
        Estimate the height of a person by adding the "bone" lengths between then joints.

        :param pose: List of joint positions.
        :return: The estimated height of the person.
        """
        np_pose = np.array(pose)
        shin_length = np.linalg.norm(np_pose[5] - np_pose[6])
        thigh_length = np.linalg.norm(np_pose[4] - np_pose[5])
        lower_spine_length = np.linalg.norm(np_pose[7] - np_pose[0])
        upper_spine_length = np.linalg.norm(np_pose[8] - np_pose[7])
        neck_length = np.linalg.norm(np_pose[9] - np_pose[8])
        head_length = np.linalg.norm(np_pose[10] - np_pose[9])
        return (shin_length + thigh_length + lower_spine_length + upper_spine_length + neck_length + head_length) / 1000
