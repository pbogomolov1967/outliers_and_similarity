# We can see how 2D outlier detection works on simulated data

import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs
from util_distances_in_clusters import calc_distances, annotate_data_points
from util_iplot import SPlot

# Global context for the ichart_demo
data = None
kmeans = None
std_distances = None


# We can specify manually how many percent of outliers we can see.
def calc(data, slider_value):
    std = std_distances[0]['std']
    std_max = np.max(std)
    point_color = ['r' if std[ix] > slider_value else 'g' for ix in range(data.shape[0])]
    point_size = [10 + std[ix] / std_max * 300 for ix in range(data.shape[0])]
    return point_color, point_size


def on_update(fig, ax, slider_value):
    point_color, point_size = calc(data, slider_value)
    ax.clear()
    ax.scatter(x=data[:,0], y=data[:,1], marker='o', c=point_color, s=point_size, cmap='RdYlGn_r', alpha=0.5)
    x_center, y_center = kmeans.cluster_centers_[0]
    ax.scatter(x=x_center, y=y_center, marker='o', c='b', s=100, cmap='RdYlGn_r', alpha=0.5)
    ax.set_title(f'Outliers grater than {"{0:.1f}".format(slider_value)} STDs')
    annotate_data_points(ax, std_distances)


def run_demo_py():
    global data, kmeans, std_distances
    # Generate some data
    data, y_true = make_blobs(n_samples=20, centers=1, cluster_std=0.80, random_state=0)
    data = data[:, ::-1]  # flip axes for better plotting

    kmeans = KMeans(n_clusters=1, random_state=0)
    labels = kmeans.fit(data).predict(data)

    std_distances = calc_distances(data, kmeans)
    std = std_distances[0]['std']
    std_max = np.max(std)

    point_color, point_size = calc(data, std_max)
    splot = SPlot(sld_from=0, sld_to=std_max, sld_init=std_max, sld_text='', sld_fmt='%1.1f',
                  fig_title='Detect outliers - manually', on_slider_update_func=on_update)

    print('Move the slider on the plot')
    splot.show()
    print('demo executed')


def run_demo_nb():
    from ipywidgets import interact, interactive, fixed, interact_manual
    import ipywidgets as widgets

    global data, kmeans, std_distances
    # Generate some data
    data, y_true = make_blobs(n_samples=20, centers=1, cluster_std=0.80, random_state=0)
    data = data[:, ::-1]  # flip axes for better plotting

    kmeans = KMeans(n_clusters=1, random_state=0)
    labels = kmeans.fit(data).predict(data)

    std_distances = calc_distances(data, kmeans)
    std = std_distances[0]['std']
    std_max = np.max(std)

    def plot_func(STDs):
        data_color, data_size = calc(data, STDs)
        fig, ax1 = plt.subplots(nrows=1, ncols=1)
        ax1.clear()
        ax1.scatter(x=data[:, 0], y=data[:, 1], marker='o', c=data_color, s=data_size, cmap='RdYlGn_r', alpha=0.5)
        annotate_data_points(ax1, std_distances)
        fig.canvas.draw_idle()

    interact(plot_func, STDs=widgets.FloatSlider(min=0, max=std_max, value=std_max, step=0.1))
    # plot_func(STDs=std_max)


if __name__ == '__main__':
    run_demo_py()

