import numpy as np
import matplotlib.pyplot as plt
import cv2
import json
import argparse
import os

from pose_graph_tracking.helpers.defaults import PACKAGE_ROOT_PATH
from pose_graph_tracking.helpers.human36m_definitions import COCO_COLORS, CONNECTED_JOINTS_PAIRS

_img_size = (1000, 1000, 3)
_delta_t_ms = 100 # 100ms = 10Hz (original playback speed)

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
        parser = argparse.ArgumentParser()
        parser.add_argument('--config_file_path',
                            type=str,
                            default=PACKAGE_ROOT_PATH + "config/visualizer_config.json",
                            help='path to .json file containing visualizer config')
        args = parser.parse_args()
        return args

    def load_visualizer_config(self,
                               path_to_config_file):
        if os.path.exists(path_to_config_file):
            with open(path_to_config_file) as json_file:
                self.config = json.load(json_file)
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
            data = json.load(json_file)

        self.keypoint_sequence = data['sequences'][self.config["sequence_id"]]
        self.action_label = data['action_labels'][self.config["sequence_id"]]

        print('Playing back sequence {}, camera {}'.format(self.config["sequence_id"], self.config["camera_id"]))
        print('Action type: {}'.format(self.action_label))

    def visualize(self):
        window_name = 'sequence {}, camera {}: {}'.format(self.config["sequence_id"],
                                                          self.config["camera_id"],
                                                          self.action_label)
        window = None

        num_joints = 17
        pairs = CONNECTED_JOINTS_PAIRS
        colors = COCO_COLORS
        colors[1] = colors[17]
        colors[2] = colors[18]
        colors[3] = colors[19]
        colors[4] = colors[20]

        for idx, frame in enumerate(self.keypoint_sequence):
            if frame['poses_2d'] is None:
                print('missing frame t={}!'.format(frame['time_idx']))
                continue

            print('image: {}\r'.format(frame['time_idx']), end='')
            kps = frame['poses_2d'][self.config["camera_id"]]  # key points are [x, y, conf]

            img = np.zeros(_img_size, dtype=np.uint8)

            centers = {}
            body_parts = {}

            # draw point
            for i in range(num_joints):
                if kps[i][2] < self.visualizer_confidence_threshold:
                    continue

                body_part = kps[i]
                center = (int(body_part[0] + 0.5), int(body_part[1] + 0.5))

                centers[i] = center
                body_parts[i] = body_part
                img = cv2.circle(img, center, 6, colors[i], thickness=-1, lineType=8, shift=0)

            # draw line
            for pair_order, pair in enumerate(pairs):
                if pair[0] not in centers.keys() or pair[1] not in centers.keys():
                    continue

                img = cv2.line(img, centers[pair[0]], centers[pair[1]], colors[pair[1]], 5)

            if window is None:
                fig = plt.figure()
                fig.canvas.set_window_title(window_name)
                window = plt.imshow(img)
            else:
                window.set_data(img)

            plt.pause(_delta_t_ms / 1000. / self.config["speedup_factor"])
            plt.draw()
            # cv2.imshow(window_name, img)
            # cv2.waitKey(_delta_t_ms / self.config["speedup_factor"])

        plt.show()
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


if __name__ == '__main__':
    visualizer = PoseGraphVisualizer()
    visualizer.visualize()
