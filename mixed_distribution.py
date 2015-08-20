import distribution
import numpy as np
import matplotlib.pyplot as plt

class MixedNormalDistribution():
    def __init__(self, num, w, mu, sigma):
        self.num = num
        self.w = w
        self.mu = mu
        self.sigma = sigma

    def distribution(self, r):
        s = 0.0
        for i in range(self.num):
            exponent = (r - self.mu[i])**2.0 / (2.0 * self.sigma[i]**2.0)
            exponential = np.exp(-exponent)
            denominator = np.sqrt(2.0 * np.pi) * self.sigma[i]
            s += self.w[i] *  exponential / denominator
        return s

def main():
    seeds = [x for x in np.arange(-7.0, 7.0, 0.1)]
    num = 2
    w = [0.3, 0.7]
    mu = [-4.0, 2.0]
    sigma = [2.0, 3.0]
    generator = MixedNormalDistribution(num, w, mu, sigma)
    harvests = [generator.distribution(x) for x in seeds]

    plt.plot(seeds, harvests)
    plt.show()

if __name__ == '__main__':
    main()