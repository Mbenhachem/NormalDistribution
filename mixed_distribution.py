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

    def force(self, r):
        numerator = 0.0
        denominator = 0.0
        for i in range(self.num):
            delta = r - self.mu[i]
            exponent = delta**2.0 / (2.0 * self.sigma[i]**2.0)
            exponential = np.exp(-exponent)
            
            exp_factor = self.w[i] * delta * exponential / self.sigma[i]
            
            numerator += -exp_factor / self.sigma[i]**2.0
            denominator += exp_factor
        return numerator / denominator

    def leapfrog(self, r, p, dt):
        half_p = p + self.force(r) * dt / 2.0
        new_r = r + half_p * dt
        new_p = half_p + self.force(new_r) * dt / 2.0
        return new_r, new_p

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
