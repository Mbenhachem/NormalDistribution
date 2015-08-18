import numpy as np
import matplotlib.pyplot as plt

class NormalDistribution():
    def __init__(self, mu, sigma):
        self.mu     = mu
        self.sigma  = sigma

    def distribution(self, r):
        exponent = (r - self.mu)**2.0 / (2.0 * self.sigma**2.0)
        return np.exp(-exponent) / (np.sqrt(2.0 * np.pi) * self.sigma)

    def force(self, r):
        return -(r - self.mu) / self.sigma**2.0

    def leapfrog(self, r, p, dt):
        half_p = p + self.force(r) * dt / 2.0
        new_r = r + half_p * dt
        new_p = half_p + self.force(new_r) * dt / 2.0
        return new_r, new_p

def main():
    seeds = [x for x in np.arange(-5.0, 5.0, 0.1)]
    generator = NormalDistribution(0.0, 2.0)
    harvests = [generator.distribution(x) for x in seeds]

    plt.plot(seeds, harvests)
    plt.show()

if __name__ == '__main__':
    main()
