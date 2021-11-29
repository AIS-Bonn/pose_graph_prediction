from math import degrees

import numpy as np

from numpy.linalg import inv as invert_matrix

from pose_graph_prediction.data.utils import get_angle_2d, get_angle_3d, get_rotation_matrix_around_z_axis

from typing import List, Tuple, Union


class PoseSequenceNormalizer(object):
    def __init__(self):
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
        self.offset = None
        self.scale_factor = None
        self.orientation_normalization_matrix = None

    def set_normalization_parameters(self,
                                     offset: np.ndarray,
                                     scale_factor: float,
                                     orientation_normalization_matrix: np.ndarray):
        self.set_normalization_offset(offset)
        self.set_normalization_scale_factor(scale_factor)
        self.set_normalization_rotation_matrix(orientation_normalization_matrix)

    def set_normalization_offset(self,
                                 offset: np.ndarray):
        self.offset = offset

    def set_normalization_scale_factor(self,
                                       scale_factor: float):
        self.scale_factor = scale_factor

    def set_normalization_rotation_matrix(self,
                                          orientation_normalization_matrix: np.ndarray):
        self.orientation_normalization_matrix = orientation_normalization_matrix

    def compute_normalization_parameters(self,
                                         pose: Union[List[Tuple[float, float, float]], np.ndarray]):
        """
        Compute the normalization parameters wrt. the pose and save them in the member variables.
        """
        first_mid_hip_position = np.array(pose[0])
        person_height = self._estimate_person_height(pose)
        rotation_matrix_around_z_axis = self._compute_normalization_rotation_matrix(pose)

        self.offset = first_mid_hip_position
        self.scale_factor = person_height
        self.orientation_normalization_matrix = rotation_matrix_around_z_axis

    def normalize_pose_sequence(self,
                                pose_sequence: Union[List[List[Tuple[float, float, float]]], np.ndarray]):
        """
        Apply the computed normalization to the provided pose_sequence.
        """
        for frame_id, current_pose in enumerate(pose_sequence):
            pose_sequence[frame_id] = self.normalize_pose(current_pose)

    def normalize_pose(self,
                       pose: Union[List[Tuple[float, float, float]], np.ndarray]):
        """
        Apply the computed normalization to the provided pose.
        """
        self._assert_all_normalization_parameters_are_set()

        pose_centered_at_mid_hip = np.array(pose) - self.offset
        scaled_pose_centered_at_mid_hip = pose_centered_at_mid_hip / self.scale_factor
        normalized_pose = np.matmul(self.orientation_normalization_matrix,
                                    scaled_pose_centered_at_mid_hip.transpose()).transpose()
        return normalized_pose

    def _assert_all_normalization_parameters_are_set(self):
        assert self.offset is not None, "Normalization offset is not set."
        assert self.scale_factor is not None, "Normalization scale factor is not set."
        assert self.orientation_normalization_matrix is not None, "Normalization orientation matrix is not set."

    def denormalize_pose(self,
                         pose: Union[List[Tuple[float, float, float]], np.ndarray]):
        """
        Denormalize the provided pose.
        """
        self._assert_all_normalization_parameters_are_set()

        normalized_pose = np.array(pose)
        derotated_pose = np.matmul(invert_matrix(self.orientation_normalization_matrix),
                                   normalized_pose.transpose()).transpose()
        descaled_pose = derotated_pose * self.scale_factor
        denormalized_pose = descaled_pose + self.offset
        return denormalized_pose

    def _compute_normalization_rotation_matrix(self,
                                               reference_pose: Union[List[Tuple[float, float, float]], np.ndarray]
                                               ) -> np.ndarray:
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
                                                                    first_left_hip_position: np.ndarray,
                                                                    first_mid_hip_position: np.ndarray) -> np.ndarray:
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
        return shin_length + thigh_length + lower_spine_length + upper_spine_length + neck_length + head_length
