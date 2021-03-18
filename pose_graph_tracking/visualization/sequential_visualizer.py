import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

from time import sleep


class SequentialVisualizer(object):
    def __init__(self):
        # TODO: prepare plot and data structure
        lines = []
        self.is_requested_to_pause = False

        plt.ion()

        self.fig = plt.figure()
        self.ax = self.fig.gca(projection='3d')
        self._clear_plot()

        theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
        self.z = np.linspace(-2, 2, 100)
        r = self.z ** 2 + 1
        self.x = r * np.sin(theta)
        self.y = r * np.cos(theta)

        self.fig.canvas.mpl_connect('key_press_event', self._pause_visualization)

    def update_plot(self,
                    data=None):
        # TODO: update the visualized data
        if self.is_requested_to_pause:
            self._wait_for_key_press()

        self._clear_plot()

        self.x += 0.1
        self.ax.plot(self.x,
                     self.y,
                     self.z, label='parametric curve')
        # Updates the visualization, without forcing the visualization window into the foreground
        self.fig.canvas.flush_events()

    def _clear_plot(self):
        self.ax.cla()
        self.ax.text2D(0.05, 0.95, "Press key to pause/continue", transform=self.ax.transAxes)

    def _pause_visualization(self,
                             _):
        self.is_requested_to_pause = True

    def _wait_for_key_press(self,
                            timeout: float = -1.0):
        while not plt.waitforbuttonpress(timeout):
            pass
        self.is_requested_to_pause = False


if __name__ == '__main__':
    visualizer = SequentialVisualizer()
    for i in range(600):
        visualizer.update_plot()
