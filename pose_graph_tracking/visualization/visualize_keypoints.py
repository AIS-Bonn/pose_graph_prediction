from argparse import ArgumentParser

from json import load as load_json_file

from math import atan2, acos, cos, degrees, sin, pi

from matplotlib.pyplot import figure, show as show_animation
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D

from mpl_toolkits.mplot3d.axes3d import Axes3D

import numpy as np

from os.path import exists

from pose_graph_tracking.data.conversions import convert_estimated_pose_sequence_to_gt_format

from pose_graph_tracking.helpers.defaults import PACKAGE_ROOT_PATH
from pose_graph_tracking.helpers.human36m_definitions import COCO_COLORS, \
    CONNECTED_JOINTS_PAIRS_FOR_HUMAN36M_GROUND_TRUTH

from typing import Any, List, Tuple, Union


def get_angle_2d(vector_a_2d: Union[np.ndarray, List[float]],
                 vector_b_2d: Union[np.ndarray, List[float]]) -> float:
    angle = atan2(vector_b_2d[1], vector_b_2d[0]) - atan2(vector_a_2d[1], vector_a_2d[0])
    while angle < 0.0:
        angle += 2 * pi
    return angle


def get_angle_3d(vector_a_3d: np.ndarray,
                 vector_b_3d: np.ndarray) -> float:
    vector_a_length = np.linalg.norm(vector_a_3d)
    vector_b_length = np.linalg.norm(vector_b_3d)
    dot_product = np.dot(vector_a_3d, vector_b_3d)
    return acos(dot_product/(vector_a_length * vector_b_length))


def get_rotation_matrix_around_z_axis(angle):
    return np.array([[cos(angle), -sin(angle), 0],
                     [sin(angle),  cos(angle), 0],
                     [         0,           0, 1]])


def get_rotation_matrix_around_x_axis(angle):
    return np.array([[1,          0,           0],
                     [0, cos(angle), -sin(angle)],
                     [0, sin(angle),  cos(angle)]])


def get_rotation_matrix_around_y_axis(angle):
    return np.array([[ cos(angle), 0, sin(angle)],
                     [          0, 1,          0],
                     [-sin(angle), 0, cos(angle)]])


def update_lines_using_pose(lines: List[Line2D],
                            pose: List[Tuple[float, float, float]],
                            connected_joint_pairs: List[Tuple[int, int]]):
    for link_id, linked_joint_ids in enumerate(connected_joint_pairs):
        link_start_keypoint = pose[linked_joint_ids[0]]
        link_end_keypoint = pose[linked_joint_ids[1]]

        # Restructure data to provide it to the line setters later on
        link_x_values = np.array([link_start_keypoint[0], link_end_keypoint[0]])
        link_y_values = np.array([link_start_keypoint[1], link_end_keypoint[1]])
        link_z_values = np.array([link_start_keypoint[2], link_end_keypoint[2]])
        # Convert from millimeters to meters
        link_x_values = link_x_values / 1000.0
        link_y_values = link_y_values / 1000.0
        link_z_values = link_z_values / 1000.0

        data_array = np.array([link_x_values, link_y_values])
        lines[link_id].set_data(data_array)
        lines[link_id].set_3d_properties(link_z_values)


def update_lines_using_pose_sequence(frame_id: int,
                                     lines: List[Line2D],
                                     pose_sequence: List[List[Tuple[float, float, float]]],
                                     connected_joint_pairs: List[Tuple[int, int]]) -> List[Line2D]:
    current_pose = pose_sequence[frame_id]
    update_lines_using_pose(lines,
                            current_pose,
                            connected_joint_pairs)

    return lines


class PoseGraphVisualizer(object):
    def __init__(self):
        self.visualize_estimated_poses = True
        self.print_number_of_missing_estimated_joints = False
        self.connected_joint_pairs = CONNECTED_JOINTS_PAIRS_FOR_HUMAN36M_GROUND_TRUTH

        self.visualizer_confidence_threshold = 0.3
        self.duration_between_frames_in_ms = 100  # 100ms = 10Hz (original playback speed)

        args = self.parse_arguments()
        self.config = None
        self.load_visualizer_config(args.config_file_path)

        self.pose_sequence = None
        self.action_label = None
        self.load_poses()

        self.plot3d = None
        self.lines = None

        self.print_infos_about_visualization()

    def parse_arguments(self) -> Any:
        parser = ArgumentParser()
        parser.add_argument('--config_file_path',
                            type=str,
                            default=PACKAGE_ROOT_PATH + "config/visualizer_config.json",
                            help='path to .json file containing visualizer config')
        args = parser.parse_args()
        return args

    def load_visualizer_config(self,
                               path_to_config_file: str):
        if exists(path_to_config_file):
            with open(path_to_config_file) as json_file:
                self.config = load_json_file(json_file)
        else:
            self.load_default_config()

    def load_default_config(self):
        self.config = {"filename": "data/original/keypoints_s1_h36m.json",
                       "subject_id": 1,  # [1, 5, 6, 7, 8, 9, 11]
                       "sequence_id": 0,  # [0-29]
                       "camera_id": 0,  # [0-3]
                       "speedup_factor": 5
                       }

    def load_poses(self):
        with open(PACKAGE_ROOT_PATH + self.config["filename"]) as json_file:
            data = load_json_file(json_file)

        sequence = data['sequences'][self.config["sequence_id"]]
        self.extract_poses_from_sequence(sequence)

        if self.print_number_of_missing_estimated_joints:
            print("Number of missing estimated joints is ", self.get_number_of_missing_estimated_joints())

        self.normalize_poses()

        self.action_label = data['action_labels'][self.config["sequence_id"]]

    def extract_poses_from_sequence(self,
                                    sequence: List[dict]):
        if self.visualize_estimated_poses:
            self.pose_sequence = [frame["poses_3d_filter"] for frame in sequence
                                  if frame["poses_3d_filter"] is not None]
            number_of_missing_frames = len(sequence) - len(self.pose_sequence)
            if number_of_missing_frames > 0:
                print('{} estimated frames are missing/None!'.format(number_of_missing_frames))
            convert_estimated_pose_sequence_to_gt_format(self.pose_sequence)
        else:
            self.pose_sequence = [frame["labels"]["poses_3d"] for frame in sequence]

    def get_number_of_missing_estimated_joints(self) -> int:
        missing_joint_counter = 0
        for frame in self.pose_sequence:
            for joint_position in frame:
                if joint_position[0] == 0.0 and joint_position[1] == 0.0 and joint_position[2] == 0.0:
                    missing_joint_counter += 1
        return missing_joint_counter

    def normalize_poses(self):
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
        if len(self.pose_sequence) == 0:
            print("Sequence does not contain any poses.")
            exit(-1)

        first_mid_hip_position = np.array(self.pose_sequence[0][0])
        person_height = self.estimate_person_height(self.pose_sequence[0])
        rotation_matrix_around_z_axis = self.compute_normalization_rotation_matrix(self.pose_sequence[0])

        for frame_id, current_pose in enumerate(self.pose_sequence):
            pose_centered_at_mid_hip = np.array(current_pose) - first_mid_hip_position
            pose_centered_at_mid_hip = pose_centered_at_mid_hip / person_height
            normalized_pose = np.matmul(rotation_matrix_around_z_axis, pose_centered_at_mid_hip.transpose()).transpose()
            self.pose_sequence[frame_id] = normalized_pose

    def compute_normalization_rotation_matrix(self,
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
        if self.is_person_laying_on_its_side(first_mid_hip_position, first_left_hip_position):
            return self.get_normalization_matrix_for_person_laying_on_its_side(reference_pose)
        else:
            return self.get_normalisation_matrix_for_person_not_laying_on_its_side(first_left_hip_position,
                                                                                   first_mid_hip_position)

    def is_person_laying_on_its_side(self,
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

    def get_normalization_matrix_for_person_laying_on_its_side(self,
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
        if self.is_person_laying_on_left_side(pose):
            angle_from_left_hip_to_y_axis = get_angle_2d(vector_mid_hip_to_belly_xy, y_axis_vector_xy)
            return get_rotation_matrix_around_z_axis(angle_from_left_hip_to_y_axis)
        else:
            angle_from_left_hip_to_neg_y_axis = get_angle_2d(vector_mid_hip_to_belly_xy, -y_axis_vector_xy)
            return get_rotation_matrix_around_z_axis(angle_from_left_hip_to_neg_y_axis)

    def get_normalisation_matrix_for_person_not_laying_on_its_side(self,
                                                                   first_left_hip_position: Tuple[float, float, float],
                                                                   first_mid_hip_position: Tuple[float, float, float]
                                                                   ) -> np.ndarray:
        mid_hip_position_xy = np.array(first_mid_hip_position[0:2])
        left_hip_position_xy = np.array(first_left_hip_position[0:2])
        vector_mid_to_left_hip_xy = left_hip_position_xy - mid_hip_position_xy
        y_axis_vector_xy = np.array([0.0, 1.0])
        angle_from_left_hip_to_y_axis = get_angle_2d(vector_mid_to_left_hip_xy, y_axis_vector_xy)
        return get_rotation_matrix_around_z_axis(angle_from_left_hip_to_y_axis)

    def is_person_laying_on_left_side(self,
                                      pose: List[Tuple[float, float, float]]):
        mid_hip_position = pose[0]
        left_hip_position = pose[4]
        if left_hip_position[2] < mid_hip_position[2]:
            return True
        return False

    def estimate_person_height(self,
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

    def print_infos_about_visualization(self):
        if self.visualize_estimated_poses:
            print('Visualizing estimated poses')
        else:
            print('Visualizing ground truth poses')
        print('Subject: {}'.format(self.config["subject_id"]))
        print('Sequence: {}'.format(self.config["sequence_id"]))
        print('Action type: {}'.format(self.action_label))

    def visualize(self):
        plot = figure()
        plot.canvas.set_window_title("Pose Graph Visualization")
        self.plot3d = Axes3D(plot)
        self.set_plot_title()
        self.create_lines()
        self.set_axes_limits()

        def draw_xy_coordinate_frame():
            return [self.plot3d.plot([-1, 1], [0, 0], [0, 0])[0], self.plot3d.plot([0, 0], [-1, 1], [0, 0])[0]]

        # Create the animation
        sequence_length = len(self.pose_sequence)
        _ = FuncAnimation(plot,
                          update_lines_using_pose_sequence,
                          frames=sequence_length,
                          init_func=draw_xy_coordinate_frame,
                          fargs=(self.lines,
                                 self.pose_sequence,
                                 self.connected_joint_pairs),
                          interval=self.duration_between_frames_in_ms,
                          blit=False)

        show_animation()

    def set_plot_title(self):
        window_name = 'Pose Visualization: Sequence {}, Camera {} - Action: {}'.format(self.config["sequence_id"],
                                                                                       self.config["camera_id"],
                                                                                       self.action_label)
        self.plot3d.set_title(window_name)

    def create_lines(self):
        number_of_lines = len(self.connected_joint_pairs)
        # Create a list of empty line objects within the plot3d - .plot returns a list of lines to be plotted, we use
        #  the first for each joint pair to set a different color to each line
        self.lines = [self.plot3d.plot([], [], [])[0] for _ in range(number_of_lines)]
        self.set_line_appearances()

    def set_line_appearances(self):
        for line_id, line in enumerate(self.lines):
            line_color = np.array(COCO_COLORS[line_id]) / 255.
            line.set_color(line_color)
            line.set_linewidth(3)

    def set_axes_limits(self):
        min_axes_limits, max_axes_limits = self.compute_axes_limits()

        self.plot3d.set_xlim3d([min_axes_limits[0] / 1000.0, max_axes_limits[0] / 1000.0])
        self.plot3d.set_xlabel('X')

        self.plot3d.set_ylim3d([min_axes_limits[1] / 1000.0, max_axes_limits[1] / 1000.0])
        self.plot3d.set_ylabel('Y')

        self.plot3d.set_zlim3d([min_axes_limits[2] / 1000.0, max_axes_limits[2] / 1000.0])
        self.plot3d.set_zlabel('Z')

    def compute_axes_limits(self) -> Tuple[List[float], List[float]]:
        min_pose_keypoint_values = [float("inf")] * 3
        max_pose_keypoint_values = [-float("inf")] * 3
        for frame in self.pose_sequence:
            current_minimum = np.minimum.reduce(frame)
            min_pose_keypoint_values = np.minimum(min_pose_keypoint_values, current_minimum)

            current_maximum = np.maximum.reduce(frame)
            max_pose_keypoint_values = np.maximum(max_pose_keypoint_values, current_maximum)

        return min_pose_keypoint_values, max_pose_keypoint_values


if __name__ == '__main__':
    visualizer = PoseGraphVisualizer()
    visualizer.visualize()
