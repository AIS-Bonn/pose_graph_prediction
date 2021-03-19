from mpl_toolkits.mplot3d.art3d import Line3D

from numpy import array

from pose_graph_tracking.helpers.human36m_definitions import COCO_COLORS, \
    CONNECTED_JOINTS_PAIRS_FOR_HUMAN36M_GROUND_TRUTH

from pose_graph_tracking.visualization.sequential_visualizer import StoppableSequentialVisualizer

from typing import List, Tuple


Pose = List[Tuple[float, float, float]]


class PoseGraphSequentialVisualizer(StoppableSequentialVisualizer):
    """
    Stoppable visualizer for 3D pose graphs.
    Input poses you want to visualize with provide_pose and provide_poses methods.
    After providing poses, call draw_provided_poses to update the visualization.

    Example
    --------
    visualizer = PoseGraphSequentialVisualizer()
    for pose in poses:
        visualizer.provide_pose(pose)
        visualizer.draw_provided_poses()
    """
    def __init__(self):
        super(PoseGraphSequentialVisualizer, self).__init__()

        self.poses_to_visualize = []
        self.connected_joint_pairs = CONNECTED_JOINTS_PAIRS_FOR_HUMAN36M_GROUND_TRUTH

        self.min_axes_limits = [-1, -1, -1]
        self.max_axes_limits = [1, 1, 1]

    def set_graph_connections(self,
                              connected_node_id_pairs: List[Tuple[int, int]]):
        self.connected_joint_pairs = connected_node_id_pairs

    def provide_poses(self,
                      poses: List[Pose]):
        for pose in poses:
            self.provide_pose(pose)

    def provide_pose(self,
                     pose: Pose):
        self.poses_to_visualize.append(pose)

    def draw_provided_poses(self):
        """ Visualized provided poses by calling update_plot of super, which internally calls our _draw_plot. """
        self.update_plot(self.poses_to_visualize)
        self.poses_to_visualize.clear()

    def _draw_plot(self,
                   poses: List[Pose]):
        for pose in poses:
            for link_id, linked_joint_ids in enumerate(self.connected_joint_pairs):
                link_start_keypoint = pose[linked_joint_ids[0]]
                link_end_keypoint = pose[linked_joint_ids[1]]

                line = self._create_line(link_start_keypoint, link_end_keypoint)
                self._set_line_appearance(line, COCO_COLORS[link_id])
                self.plot3d.add_line(line)

        self._set_plots_axes_limits()
        self._plot_xy_origin()

    def _create_line(self,
                     start_point: Tuple[float, float, float],
                     end_point: Tuple[float, float, float]) -> Line3D:
        x_values = array([start_point[0], end_point[0]])
        y_values = array([start_point[1], end_point[1]])
        z_values = array([start_point[2], end_point[2]])
        return Line3D(x_values, y_values, z_values)

    def _set_line_appearance(self,
                             line: Line3D,
                             color: Tuple[int, int, int],
                             line_width: int = 3):
        line_color = array(color) / 255.
        line.set_color(line_color)
        line.set_linewidth(line_width)

    def set_axes_limits(self,
                        min_axes_limits: Tuple[float, float, float],
                        max_axes_limits: Tuple[float, float, float]):
        self.min_axes_limits = min_axes_limits
        self.max_axes_limits = max_axes_limits

    def _set_plots_axes_limits(self):
        self.plot3d.set_xlim3d([self.min_axes_limits[0], self.max_axes_limits[0]])
        self.plot3d.set_xlabel('X')

        self.plot3d.set_ylim3d([self.min_axes_limits[1], self.max_axes_limits[1]])
        self.plot3d.set_ylabel('Y')

        self.plot3d.set_zlim3d([self.min_axes_limits[2], self.max_axes_limits[2]])
        self.plot3d.set_zlabel('Z')

    def _plot_xy_origin(self):
        self.plot3d.plot([self.min_axes_limits[0], self.max_axes_limits[0]], [0, 0], [0, 0])
        self.plot3d.plot([0, 0], [self.min_axes_limits[1], self.max_axes_limits[1]], [0, 0])
