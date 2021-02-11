from argparse import ArgumentParser

from json import load as load_json_file

from matplotlib.pyplot import figure, show as show_animation
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D

from mpl_toolkits.mplot3d.axes3d import Axes3D

import numpy as np

from os.path import exists

from pose_graph_tracking.data.conversions import convert_estimated_pose_sequence_to_gt_format
from pose_graph_tracking.data.normalization import PoseSequenceNormalizer

from pose_graph_tracking.helpers.defaults import PACKAGE_ROOT_PATH
from pose_graph_tracking.helpers.human36m_definitions import COCO_COLORS, \
    CONNECTED_JOINTS_PAIRS_FOR_HUMAN36M_GROUND_TRUTH

from typing import Any, List, Tuple


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
        # If true visualizing the estimated poses, if false visualizing the ground truth poses
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

        normalizer = PoseSequenceNormalizer()
        normalizer.compute_normalization_parameters(self.pose_sequence)
        normalizer.normalize_pose_sequence(self.pose_sequence)

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
