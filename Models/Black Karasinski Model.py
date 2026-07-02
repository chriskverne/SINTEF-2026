import numpy as np
import matplotlib.pyplot as plt

class BlackKarasinskiModel:
    def __init__(self, k, theta, var, dt):
        self.k = k # currently constant
        self.theta = theta # currently constant
        self.var = var
        self.dt = dt

    def generate_paths(self, n_paths=10, n_steps=10):
        r_start = self.theta[0]
        ln_r = np.log(r_start)

        prices = np.zeros((n_steps, n_paths))
        for j in range(n_paths):
            ln_r = np.log(self.theta[0])
            for i in range(n_steps):
                prices[i][j] = np.exp(ln_r)
                ln_r = ln_r + self.k * (np.log(self.theta[i]) - ln_r)*self.dt + np.random.normal(0, scale=self.var) * self.dt

        return prices
    
    def plot_paths(self, n_paths=10, n_steps=10):
        prices = self.generate_paths(n_paths, n_steps)
        x = np.arange(n_steps)

        plt.figure(figsize=(8, 5))
        for j in range(n_paths):
            plt.plot(x, prices[:, j], alpha=0.7, label=f'Path {j+1}')

        plt.axhline(self.theta[0], color='black', linestyle='--', linewidth=1, label='θ (long-run mean)')
        plt.xlabel('Time step')
        plt.ylabel('Short rate')
        plt.title('Black-Karasinski Simulated Paths')
        plt.legend(fontsize=8, ncol=2)
        plt.show()

    def construct_trinomial_tree(self, num_steps=3):
        # we will start with assuming theta and kappa are constant
        # here delta x = sigma sqrt(3t)

        for i in range(num_steps):
            p_up = 1/6 + (self.k**2 * i**2 * self.dt**2 - self.k*self.dt)/2
            p_mid = 2/3 - self.k**2 * i**2 * self.dt**2
            p_down = 1/6 + (self.k**2 * i**2 * self.dt**2 + self.k*self.dt)/2
            print(f"Step {i}: p_up={p_up:.4f}, p_mid={p_mid:.4f}, p_down={p_down:.4f}")

        return 1

    def construct_binomial_tree(self):
        # we will start with assuming theta and kappa are constant
        # here delta x = sigma sqrt(t)
        return 1


theta = {0: 0.05, 1: 0.05, 2: 0.05, 3: 0.05, 4: 0.05, 
         5: 0.07, 6: 0.07, 7: 0.07, 8: 0.07, 9: 0.07}
theta = 0.05 # consant for now
bkm = BlackKarasinskiModel(k=0.1, theta=theta, var=0.02, dt=1)
# bkm.plot_paths()
bkm.construct_trinomial_tree(num_steps=3)
