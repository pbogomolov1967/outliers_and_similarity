import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import euclidean_distances

"""
from google.colab import drive
drive.mount('/content/gdrive')
sys.path.append('/content/gdrive/My Drive/Colab Notebooks/outliers_demo')
"""


def plot_kmeans(plt, k_means, X, ax=None, title=None, x_lab=None, y_lab=None, draw_lines=False):
    labels = k_means.fit_predict(X)

    # plot the input data
    ax = ax or plt.gca()
    ax.axis('equal')
    ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)
    if title is not None: ax.set_title(title)
    if x_lab is not None: ax.set_xlabel(x_lab)
    if y_lab is not None: ax.set_ylabel(y_lab)

    # plot the representation of the KMeans model
    centers = k_means.cluster_centers_
    radii = [cdist(X[labels == i], [center]).max()
             for i, center in enumerate(centers)]
    for c_ix, (c, r) in enumerate(zip(centers, radii)):
        ax.add_patch(plt.Circle(c, r, fc='#CCCCCC', lw=3, alpha=0.5, zorder=1))
        ax.add_patch(plt.Circle(c, 0.1, fc='#34CA9C', lw=3, alpha=0.5, zorder=1))
        if draw_lines:
            for x_ix, xy in enumerate(X):
                lab_ix = labels[x_ix]
                ax.annotate("", xy=X[x_ix], xytext=centers[lab_ix], arrowprops = dict(arrowstyle="->"))

    return ax


def calc_distances(X, k_means):
    std_distances = []
    xy_and_labels = list(zip(X, k_means.labels_))

    for cluster_label in np.unique(k_means.labels_):
        X1 = filter(lambda xy_label: xy_label[1] == cluster_label, xy_and_labels)
        X1 = [tuple(xy_label[0]) for xy_label in X1]
        X1_center = tuple(k_means.cluster_centers_[cluster_label])
        distances1 = euclidean_distances(X1, [X1_center])
        distances1 = [d[0] for d in distances1]
        std1 = np.std(distances1)
        stds1 = np.array(distances1) / std1
        std_distances.append({'std': stds1, 'center': X1_center, 'xy': X1})

    return std_distances


def annotate_data_points(ax, std_distances):
    for cluster in std_distances:
        std, center, xy = cluster['std'], cluster['center'], cluster['xy']
        for ix, std in enumerate(std):
            label1 = str(round(std, 1))
            ax.annotate(label1, xy=xy[ix], xytext=np.array(xy[ix]) + 0.05, fontsize=8)
