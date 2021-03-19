from matplotlib.pyplot import figure, ion, waitforbuttonpress

from mpl_toolkits.mplot3d import Axes3D

from typing import Any


class StoppableSequentialVisualizer(object):
    """
    Abstract class providing the functionality to visualize a sequence of data frame by frame.
    The visualization can be stopped and continued by pressing a keyboard key.

    A subclass has to implement the _draw_plot method, providing data to the plot.
    Calling the update_plot method updates the visualization.
    """
    def __init__(self,
                 window_title: str = "Sequential Visualization"):
        self.is_requested_to_pause = False

        # Activate interactive mode for plots
        ion()

        # Create figure and set key press to de- and activate visualization pause
        self.fig = figure()
        self.fig.canvas.set_window_title(window_title)
        self.fig.canvas.mpl_connect('key_press_event', self._request_to_pause_visualization)

        self.plot3d = Axes3D(self.fig)
        self._clear_plot()

    def _clear_plot(self):
        self.plot3d.cla()
        self.plot3d.text2D(0.05, 0.95, "Press key to pause/continue", transform=self.plot3d.transAxes)

    def _request_to_pause_visualization(self,
                                        _):
        self.is_requested_to_pause = True

    def update_plot(self,
                    data: Any):
        if self.is_requested_to_pause:
            self._wait_for_key_press()
            self.is_requested_to_pause = False

        self._clear_plot()
        self._draw_plot(data)
        # Updates the visualization, without forcing the visualization window into the foreground
        self.fig.canvas.flush_events()

    def _wait_for_key_press(self,
                            timeout: float = -1.0):
        while not waitforbuttonpress(timeout):
            pass

    def _draw_plot(self,
                   data: Any):
        # Performs actual drawing of the plot.
        raise NotImplementedError('Needs to be implemented by subclasses to actually make an animation.')
