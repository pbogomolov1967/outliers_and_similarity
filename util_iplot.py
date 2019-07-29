# Interactive Plot with a Slider

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


class SPlot(object):
    """Plot with a slider"""

    def __init__(self, **kwargs):
        """Constructor"""
        sld_from = kwargs['sld_from']
        sld_to = kwargs['sld_to']
        sld_init = kwargs['sld_init']
        sld_text = kwargs['sld_text']
        sld_fmt = kwargs['sld_fmt']
        fig_title = kwargs['fig_title']
        self.on_slider_update_func = kwargs['on_slider_update_func']
        self.plt = plt
        self.fig, (self.ax1, self.ax2) = \
            plt.subplots(nrows=2, ncols=1, figsize=(6, 6),
                         gridspec_kw={'height_ratios': [10, 1]})
        self.fig.canvas.set_window_title(fig_title)
        self.ax2.set_xticks([])
        self.ax2.set_yticks([])
        [l.set_visible(False) for l in self.ax2.spines.values()]
        self.slider = Slider(self.ax2, sld_text, sld_from, sld_to, valinit=sld_init, valfmt=sld_fmt)
        self.slider.on_changed(self.on_slider_update)
        self.on_slider_update_func(self.fig, self.ax1, sld_init)

    def on_slider_update(self, val):
        self.on_slider_update_func(self.fig, self.ax1, val)
        self.fig.canvas.draw_idle()

    def show(self):
        self.plt.show()


if __name__ == '__main__':
    # Generate data where we know for sure how many percent of outliers in it.

    def gen_data(param):
        pass


    def on_update(fig, ax, slider_value):
        X, y, _, _ = gen_data(slider_value / 100)
        data_color = y
        data_size = [10 for _ in range(np.shape(X)[0])]
        ax.clear()
        ax.scatter(x=X[:, 0], y=X[:, 1], marker='o', c=data_color, s=data_size, cmap='RdYlGn_r', alpha=0.5)
        ax.set_title(f'Data points {round(slider_value)}%')

    splot = SPlot(sld_from=0, sld_to=100, sld_init=1, sld_text='', sld_fmt='%1.0f%%',
                 fig_title='Generate data', on_slider_update_func=on_update)

    splot.show()

    print('done')

###