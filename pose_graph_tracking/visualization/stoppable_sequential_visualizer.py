from matplotlib.pyplot import figure, ion
from matplotlib.backend_bases import Event

from mpl_toolkits.mplot3d import Axes3D

from time import sleep

from typing import Any


class StoppableSequentialVisualizer(object):
    """
    Abstract class providing the functionality to visualize a sequence of data frame by frame.
    The visualization can be stopped and continued by pressing the space bar.
    While stopped, the next frame can be visualized by pressing the right arrow key.

    A subclass has to implement the _draw_plot method, providing data to the plot.
    Calling the update_plot method updates the visualization.
    """
    def __init__(self,
                 window_title: str = "Sequential Visualization"):
        self.is_requested_to_pause = False
        self.is_next_frame_requested = False

        # Activate interactive mode for plots
        ion()

        # Create figure and set key press to de- and activate visualization pause
        self.fig = figure()
        self.fig.canvas.set_window_title(window_title)
        self.fig.canvas.mpl_connect('key_press_event', self._request_to_pause_or_continue_visualization)
        self.fig.canvas.mpl_connect('key_press_event', self._request_next_frame)
        # Updates the visualization, without forcing the visualization window into the foreground
        self._update_visualization = self.fig.canvas.flush_events

        self.plot3d = Axes3D(self.fig)
        self._clear_plot()

    def _clear_plot(self):
        self.plot3d.cla()
        self.plot3d.text2D(0.05,
                           0.95,
                           "Press space bar to pause/continue - right arrow key to show next frame",
                           transform=self.plot3d.transAxes)

    def _request_to_pause_or_continue_visualization(self,
                                                    event: Event):
        if event.key == " ":
            self.is_requested_to_pause = not self.is_requested_to_pause

    def _request_next_frame(self,
                            event: Event):
        if event.key == "right" and self.is_requested_to_pause:
            self.is_next_frame_requested = True

    def update_plot(self,
                    data: Any):
        self._clear_plot()
        self._draw_plot(data)
        self._update_visualization()

        self._handle_user_requests()

    def _handle_user_requests(self):
        while self.is_requested_to_pause:
            sleep(0.01)  # 100 frames per second
            self._update_visualization()
            # Break out of pause if next frame is requested, but end up in pause again after next frame is visualized
            if self.is_next_frame_requested:
                self.is_next_frame_requested = False
                break

    def _draw_plot(self,
                   data: Any):
        # Performs actual drawing of the plot.
        raise NotImplementedError('Needs to be implemented by subclasses to actually make an animation.')
