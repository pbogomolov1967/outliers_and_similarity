import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


def show_prob(mu, sigma, x1, x2):
    # a = stats.norm.rvs(loc=mu, scale=sigma, size=40)
    # plt.hist(a, normed=True)
    xs = np.linspace(x1, x2, 1000)
    plt.fill_between(xs, stats.norm.pdf(xs, loc=mu, scale=sigma), alpha=.5, label='prob')
    xs = np.linspace(-7, 7, 1000)
    plt.plot(xs, stats.norm.pdf(xs, loc=mu, scale=sigma), alpha=.5, label='pdf')
    plt.plot(xs, stats.norm.cdf(xs, loc=mu, scale=sigma), alpha=.5, label='cdf')
    prob = stats.norm.cdf(x2, loc=mu, scale=sigma) - stats.norm.cdf(x1, loc=mu, scale=sigma)
    plt.title(f'probability between x1={x1} and x2={x2} is {prob:.6f}')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':    
    show_prob(mu=0., sigma=1., x1=-1., x2=0.)
    print()
