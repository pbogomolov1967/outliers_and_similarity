import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import euclidean_distances

from util_distances_in_clusters import plot_kmeans, annotate_data_points2


def run_sim_nb():

    # Generate data
    np.random.seed(0)
    data, y_true = make_blobs(n_samples=5, centers=1, cluster_std=0.80, random_state=0)
    data = data[:, ::-1]  # flip axes for better plotting

    ix = 3
    x = data[ix]
    colors = np.zeros(data.shape[0])
    colors[ix] = 1

    distances1 = euclidean_distances(data, [x])
    distances1 = [d[0] for d in distances1]

    distances1_max = np.max(distances1)
    similarity_weights = [1. - d / distances1_max for d in distances1]
    prices = [120., 80., 80., 60., 50.]

    weighted_avg_price = np.average(prices, weights=similarity_weights)
    info = f'price was {prices[ix]} and its weighted_avg now is {round(weighted_avg_price,1)}'

    prices[3] = weighted_avg_price

    kmeans = KMeans(n_clusters=1, random_state=0)
    ax = plot_kmeans(plt, kmeans, data, title='Similarity: ' + info, colors=colors)
    annotate_data_points2(plt, data, prices, ax=ax)


if __name__ == '__main__':
    run_sim_nb()
    plt.show()
    pass
