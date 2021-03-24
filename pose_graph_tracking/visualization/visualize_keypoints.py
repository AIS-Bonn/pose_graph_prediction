from argparse import ArgumentParser

from json import load as load_json_file

import numpy as np

from os.path import exists

from pose_graph_tracking.data.conversions import convert_estimated_pose_sequence_to_gt_format
from pose_graph_tracking.data.normalization import PoseSequenceNormalizer

from pose_graph_tracking.helpers.defaults import PACKAGE_ROOT_PATH
from pose_graph_tracking.helpers.human36m_definitions import COCO_COLORS, \
    CONNECTED_JOINTS_PAIRS_FOR_HUMAN36M_GROUND_TRUTH

from pose_graph_tracking.visualization.pose_graph_sequential_visualizer import PoseGraphSequentialVisualizer

from typing import Any, List, Tuple


class Human36MPoseGraphVisualizer(object):
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
        normalizer.compute_normalization_parameters(self.pose_sequence[0])
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
        sequential_visualizer = PoseGraphSequentialVisualizer()
        sequential_visualizer.set_default_link_colors(COCO_COLORS)
        sequential_visualizer.set_graph_connections(self.connected_joint_pairs)
        sequential_visualizer.set_axes_limits(*self.compute_axes_limits())
        for pose in self.pose_sequence:
            sequential_visualizer.provide_pose(pose)
            sequential_visualizer.draw_provided_poses()

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
    visualizer = Human36MPoseGraphVisualizer()
    visualizer.visualize()
