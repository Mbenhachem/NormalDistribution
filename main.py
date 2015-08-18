import distribution
import numpy as np

def prob(generator, new_r, new_p, r, p):
    new_ham = new_p * new_p / 2.0 - np.log( generator.distribution(new_r) )
    ham = p * p / 2.0  - np.log( generator.distribution(r) )
    return np.exp(-(new_ham - ham))

def main(init_r, mu, sigma, dt, steps):
    generator =distribution.NormalDistribution(mu, sigma)
    r = init_r
    for t in range(steps):
        print(t, r)
        p = np.random.randn()
        new_r, new_p = generator.leapfrog(r, p, dt)
        if prob(generator, new_r, new_p, r, p) >= np.random.rand():
            r, p = new_r, new_p


if __name__ == '__main__':
    init_r = 0.0
    mu, sigma = 0.0, 2.0
    dt, steps = 0.01, 100000
    main(init_r, mu, sigma, dt, steps)
