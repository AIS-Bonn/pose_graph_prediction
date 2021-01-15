from argparse import ArgumentParser

from json import load as load_json_file

from matplotlib.pyplot import figure, show as show_animation
from matplotlib.animation import FuncAnimation

from mpl_toolkits.mplot3d.axes3d import Axes3D

import numpy as np

from os.path import exists

from pose_graph_tracking.helpers.defaults import PACKAGE_ROOT_PATH
_img_size = (1000, 1000, 3)
_delta_t_ms = 100  # 100ms = 10Hz (original playback speed)
from pose_graph_tracking.helpers.human36m_definitions import COCO_COLORS, CONNECTED_JOINTS_PAIRS


class PoseGraphVisualizer(object):
    def __init__(self):
        self.visualizer_confidence_threshold = 0.3

        args = self.parse_arguments()
        print("args ", args)
        self.config = None
        self.load_visualizer_config(args.config_file_path)

        self.keypoint_sequence = None
        self.action_label = None
        self.load_poses()

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
        self.config = {"filename": "data/keypoints_s1_h36m.json",
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
        window_name = 'sequence {}, camera {}: {}'.format(self.config["sequence_id"],
                                                          self.config["camera_id"],
                                                          self.action_label)

        # pairs = CONNECTED_JOINTS_PAIRS
        colors = COCO_COLORS
        colors[1] = colors[17]
        colors[2] = colors[18]
        colors[3] = colors[19]
        colors[4] = colors[20]

        # print("self.keypoint_sequence \n", self.keypoint_sequence)

        fig = figure()
        fig.canvas.set_window_title(window_name)
        ax = Axes3D(fig)

        number_keypoints = len(CONNECTED_JOINTS_PAIRS)
        lines = [ax.plot([], [], [])[0] for _ in range(number_keypoints)]
        for line_id, line in enumerate(lines):
            line_color = np.array(COCO_COLORS[line_id]) / 255.
            line.set_color(line_color)

        sequence_length = len(self.keypoint_sequence)

        min_keypoint_values = [float("inf")] * 3
        max_keypoint_values = [-float("inf")] * 3
        # Get min and max values per axis for sequence
        for frame in self.keypoint_sequence:
            if frame['poses_3d_triang'] is None:
                print('missing frame t={}!'.format(frame['time_idx']))
                continue

            current_minimum = np.minimum.reduce(frame['poses_3d_triang'])
            min_keypoint_values = np.minimum(min_keypoint_values, current_minimum)

            current_maximum = np.maximum.reduce(frame['poses_3d_triang'])
            max_keypoint_values = np.maximum(max_keypoint_values, current_maximum)

        # Setting the axes properties
        ax.set_xlim3d([min_keypoint_values[0] / 1000.0, max_keypoint_values[0] / 1000.0])
        ax.set_xlabel('X')

        ax.set_ylim3d([min_keypoint_values[1] / 1000.0, max_keypoint_values[1] / 1000.0])
        ax.set_ylabel('Y')

        ax.set_zlim3d([min_keypoint_values[2] / 1000.0, max_keypoint_values[2] / 1000.0])
        ax.set_zlabel('Z')

        ax.set_title('3D Test')

        def update_lines(i, sequence, lines):
            print("Frame: ", i)
            current_frame = sequence[i]
            current_pose = current_frame["poses_3d_triang"]
            if current_pose is None:
                print('missing frame t={}!'.format(current_frame['time_idx']))
                return lines

            for link_id, linked_joint_ids in enumerate(CONNECTED_JOINTS_PAIRS):
                link_start_keypoint = current_pose[linked_joint_ids[0]]
                link_end_keypoint = current_pose[linked_joint_ids[1]]

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

            return lines

        # Creating the Animation object
        line_ani = animation.FuncAnimation(fig,
                                           update_lines,
                                           frames=sequence_length,
                                           fargs=(self.keypoint_sequence, lines),
                                           interval=_delta_t_ms,
                                           blit=False)

        show_animation()


if __name__ == '__main__':
    visualizer = PoseGraphVisualizer()
    visualizer.visualize()
