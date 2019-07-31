import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn import mixture

from util_distances_in_clusters import annotate_data_points2


def run_gmm_nb():
    n_samples = 30
    np.random.seed(0)

    # generate zero centered stretched Gaussian data
    C = np.array([[0., -0.7], [5.5, 1.7]])
    stretched_gaussian = np.dot(np.random.randn(n_samples, 2), C)

    # concatenate the two datasets into the final training set
    X_train = stretched_gaussian

    # fit a Gaussian Mixture Model with two components
    clf = mixture.GaussianMixture(n_components=1, covariance_type='full')
    clf.fit(X_train)

    # display predicted scores by the model as a contour plot
    x = np.linspace(-15., 15.)
    y = np.linspace(-15., 15.)
    X, Y = np.meshgrid(x, y)
    XX = np.array([X.ravel(), Y.ravel()]).T
    Z = -clf.score_samples(XX)
    Z = Z.reshape(X.shape)

    CS = plt.contour(X, Y, Z, norm=LogNorm(vmin=1.0, vmax=100.0), levels=np.logspace(0, 2, 4))
    # CB = plt.colorbar(CS) # , shrink=0.8, extend='both')
    plt.scatter(X_train[:, 0], X_train[:, 1], s=10, marker='o', alpha=0.5)

    # np.exp(weighted_log_probs).sum() ~ 1.
    weighted_log_probs = clf.score_samples(X_train)
    annotate_data_points2(plt, X_train, np.exp(weighted_log_probs) * 100.)

    plt.title('GMM probs')
    plt.axis('tight')


if __name__ == '__main__':
    # If a cluster is stretched cloud of data points we can use Gaussian Mixture Model (GMM)
    # to count of covariance of the data correctly
    run_gmm_nb()
    plt.show()
