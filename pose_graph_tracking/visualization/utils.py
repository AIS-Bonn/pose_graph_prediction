from matplotlib.animation import FuncAnimation


class StoppableAnimation(FuncAnimation):
    def __init__(self, fig, func, *args, **kwargs):
        self.is_running = True
        fig.canvas.mpl_connect('key_press_event', self.start_stop)

        FuncAnimation.__init__(self, fig, func, *args, **kwargs)

    def start_stop(self, _):
        if self.is_running:
            self.event_source.stop()
            self.is_running = False
        else:
            self.event_source.start()
            self.is_running = True
