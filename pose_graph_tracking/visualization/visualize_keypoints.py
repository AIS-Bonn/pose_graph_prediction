from argparse import ArgumentParser

from json import load as load_json_file

from matplotlib.pyplot import figure, show as show_animation
from matplotlib.animation import FuncAnimation

from mpl_toolkits.mplot3d.axes3d import Axes3D

import numpy as np

from os.path import exists

from pose_graph_tracking.helpers.defaults import PACKAGE_ROOT_PATH
from pose_graph_tracking.helpers.human36m_definitions import COCO_COLORS, \
    CONNECTED_JOINTS_PAIRS_FOR_ESTIMATION, CONNECTED_JOINTS_PAIRS_FOR_HUMAN36M_GROUND_TRUTH



def update_lines_using_pose(lines, pose, connected_joint_pairs):
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


def get_estimated_keypoints_from_sequence_frame(sequence_frame):
    print("estimated")
    return sequence_frame["poses_3d_triang"]


def get_ground_truth_keypoints_from_sequence_frame(sequence_frame):
    print("ground truth ")
    return sequence_frame["labels"]["poses_3d"]


def update_lines_using_keypoints_sequence(frame_id,
                                          lines,
                                          pose_sequence,
                                          get_pose_from_sequence_frame_function,
                                          connected_joint_pairs):
    print("Frame: ", frame_id)
    current_frame = pose_sequence[frame_id]
    current_pose = get_pose_from_sequence_frame_function(current_frame)
    if current_pose is None:
        print('missing frame t={}!'.format(current_frame['time_idx']))
        return lines

    update_lines_using_pose(lines,
                            current_pose,
                            connected_joint_pairs)

    return lines


class PoseGraphVisualizer(object):
    def __init__(self):
        self.visualize_estimated_keypoints = False
        if self.visualize_estimated_keypoints:
            self.connected_joint_pairs = CONNECTED_JOINTS_PAIRS_FOR_ESTIMATION
        else:
            self.connected_joint_pairs = CONNECTED_JOINTS_PAIRS_FOR_HUMAN36M_GROUND_TRUTH

        self.visualizer_confidence_threshold = 0.3
        self.duration_between_frames_in_ms = 100  # 100ms = 10Hz (original playback speed)

        args = self.parse_arguments()
        self.config = None
        self.load_visualizer_config(args.config_file_path)

        self.keypoint_sequence = None
        self.action_label = None
        self.load_poses()

        self.plot3d = None
        self.lines = None

    def parse_arguments(self):
        parser = ArgumentParser()
        parser.add_argument('--config_file_path',
                            type=str,
                            default=PACKAGE_ROOT_PATH + "config/visualizer_config.json",
                            help='path to .json file containing visualizer config')
        args = parser.parse_args()
        return args

    def load_visualizer_config(self,
                               path_to_config_file):
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

        self.keypoint_sequence = data['sequences'][self.config["sequence_id"]]
        self.action_label = data['action_labels'][self.config["sequence_id"]]

        print('Playing back sequence {}, camera {}'.format(self.config["sequence_id"], self.config["camera_id"]))
        print('Action type: {}'.format(self.action_label))

    def visualize(self):
        plot = figure()
        plot.canvas.set_window_title("Pose Graph Visualization")
        self.plot3d = Axes3D(plot)
        self.set_plot_title()
        self.create_lines()
        self.set_axes_limits()

        # Create the animation
        sequence_length = len(self.keypoint_sequence)
        if self.visualize_estimated_keypoints:
            _ = FuncAnimation(plot,
                              update_lines_using_keypoints_sequence,
                              frames=sequence_length,
                              fargs=(self.lines,
                                     self.keypoint_sequence,
                                     get_estimated_keypoints_from_sequence_frame,
                                     self.connected_joint_pairs),
                              interval=self.duration_between_frames_in_ms,
                              blit=False)
        else:  # visualize ground truth keypoints - currently they have a different format
            _ = FuncAnimation(plot,
                              update_lines_using_keypoints_sequence,
                              frames=sequence_length,
                              fargs=(self.lines,
                                     self.keypoint_sequence,
                                     get_ground_truth_keypoints_from_sequence_frame,
                                     self.connected_joint_pairs),
                              interval=self.duration_between_frames_in_ms,
                              blit=False)

        show_animation()

    def set_plot_title(self):
        window_name = 'Keypoint Visualization: Sequence {}, Camera {} - Action: {}'.format(self.config["sequence_id"],
                                                                                           self.config["camera_id"],
                                                                                           self.action_label)
        self.plot3d.set_title(window_name)

    def create_lines(self):
        number_of_lines = len(self.connected_joint_pairs)
        # Create a list of empty line objects within the plot3d - .plot returns a list of lines to be plotted, we use
        #  the first for each joint pair to set a different color to each line
        self.lines = [self.plot3d.plot([], [], [])[0] for _ in range(number_of_lines)]
        self.set_line_colors()

    def set_line_colors(self):
        for line_id, line in enumerate(self.lines):
            line_color = np.array(COCO_COLORS[line_id]) / 255.
            line.set_color(line_color)

    def set_axes_limits(self):
        min_axes_limits, max_axes_limits = self.compute_axes_limits()

        self.plot3d.set_xlim3d([min_axes_limits[0] / 1000.0, max_axes_limits[0] / 1000.0])
        self.plot3d.set_xlabel('X')

        self.plot3d.set_ylim3d([min_axes_limits[1] / 1000.0, max_axes_limits[1] / 1000.0])
        self.plot3d.set_ylabel('Y')

        self.plot3d.set_zlim3d([min_axes_limits[2] / 1000.0, max_axes_limits[2] / 1000.0])
        self.plot3d.set_zlabel('Z')

    def compute_axes_limits(self):
        min_keypoint_values = [float("inf")] * 3
        max_keypoint_values = [-float("inf")] * 3
        for frame in self.keypoint_sequence:
            if frame['poses_3d_triang'] is None:
                print('missing frame t={}!'.format(frame['time_idx']))
                continue

            current_minimum = np.minimum.reduce(frame['poses_3d_triang'])
            min_keypoint_values = np.minimum(min_keypoint_values, current_minimum)

            current_maximum = np.maximum.reduce(frame['poses_3d_triang'])
            max_keypoint_values = np.maximum(max_keypoint_values, current_maximum)

        return min_keypoint_values, max_keypoint_values


if __name__ == '__main__':
    visualizer = PoseGraphVisualizer()
    visualizer.visualize()
