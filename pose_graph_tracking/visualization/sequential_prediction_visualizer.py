from time import time

from torch import long as torchlong, zeros

from torch_geometric.data import DataLoader, Dataset

from pose_graph_tracking.data.normalization import PoseSequenceNormalizer
from pose_graph_tracking.data.dataset_generator_utils import convert_poses_to_graph_data

from pose_graph_tracking.helpers.human36m_definitions import COCO_COLORS, \
    CONNECTED_JOINTS_PAIRS_FOR_HUMAN36M_GROUND_TRUTH

from pose_graph_tracking.model.pose_graph_prediction_net import PoseGraphPredictionNet

from pose_graph_tracking.visualization.pose_graph_sequential_visualizer import PoseGraphSequentialVisualizer


class SequentialPredictionVisualizer(object):
    def visualize_model(self,
                        model: PoseGraphPredictionNet,
                        data_set: Dataset,
                        use_output_as_next_input: bool = True):
        """
        Provides the data sequentially to the model and visualizes the results after each step.

        :param model: PoseGraphPredictionNet processing the data.
        :param data_set: Data set to visualize.
        """
        data_loader = DataLoader(data_set, batch_size=1, shuffle=False)

        sequential_visualizer = PoseGraphSequentialVisualizer(start_paused=True)
        sequential_visualizer.set_default_link_colors(COCO_COLORS)
        sequential_visualizer.set_graph_connections(CONNECTED_JOINTS_PAIRS_FOR_HUMAN36M_GROUND_TRUTH)
        sequential_visualizer.set_axes_limits([-1000, -1000, -1000], [1000, 1000, 1000])

        normalizer = PoseSequenceNormalizer()
        visualized_frames_counter = 0
        previous_pose = None
        current_pose = None
        model.eval()
        for gt_data in data_loader:
            if previous_pose is not None and current_pose is not None:
                data = convert_poses_to_graph_data(previous_pose, current_pose)
                data["ground_truth"] = gt_data["ground_truth"]
                num_nodes = data["x"].shape[0]
                batch_ids = zeros(num_nodes, dtype=torchlong)
                data["batch"] = batch_ids
            else:
                data = gt_data

            sample_start_time = time()

            # Use provided model to predict the next pose
            predicted_next_pose = model(data)

            model_processing_time = time() - sample_start_time
            print("Time needed for Model Execution: ", model_processing_time)

            # Get relevant poses
            current_pose = data["x"].numpy()[:, -3:]
            predicted_next_pose = predicted_next_pose.detach().numpy()
            next_gt_pose = data["ground_truth"].numpy()

            # Prepare normalizer
            normalizer.set_normalization_offset(data["normalization_offset"].numpy())
            normalizer.set_normalization_scale_factor(data["normalization_scale"].item())
            normalizer.set_normalization_rotation_matrix(data["normalization_rotation_matrix"].numpy())

            # Denormalize poses
            denormalized_current_pose = normalizer.denormalize_pose(current_pose)
            denormalized_predicted_next_pose = normalizer.denormalize_pose(predicted_next_pose)
            denormalized_next_gt_pose = normalizer.denormalize_pose(next_gt_pose)

            # Visualize poses
            sequential_visualizer.provide_pose_with_uniform_color(denormalized_current_pose, [124, 124, 0])
            sequential_visualizer.provide_pose_with_uniform_color(denormalized_predicted_next_pose, [200, 0, 0])
            sequential_visualizer.provide_pose_with_uniform_color(denormalized_next_gt_pose, [0, 200, 0])
            sequential_visualizer.draw_provided_poses()

            print(visualized_frames_counter)

            # Prepare next iteration
            visualized_frames_counter += 1
            if use_output_as_next_input:
                previous_pose = denormalized_current_pose
                current_pose = denormalized_predicted_next_pose
