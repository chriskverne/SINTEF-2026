import numpy as np


class GeometricBrownianMotion:
    def __init__(self, S0, mu, sigma, dt):
        """
            S0: Initial price
            mu: expected return (drift)
            sigma: standard deviation (volatility)
            dt: time step size
        """
        self.S0 = S0
        self.mu = mu
        self.sigma = sigma
        self.dt = dt

        # Compute up/down factors + probabilities + quantum rotation angle
        self.u = np.exp(self.sigma * np.sqrt(self.dt))
        self.d = 1.0 / self.u
        self.q = (self.u * np.exp(self.mu * self.dt) - 1) / (self.u**2 - 1)

        # q = sin^2(theta/2) -> theta = 2 * arcsin(sqrt(q))
        self.theta = 2 * np.arcsin(np.sqrt(self.q))

    def create_quantum_state(self, m):
        """
            m: number of qubits (time steps)
            Returns a quantum state representing the price distribution at time T = m*dt
        """

        state = np.zeros(2**m)
        ry_mat = np.array([
            [np.cos(self.theta/2), -np.sin(self.theta/2)],
            [np.sin(self.theta/2), np.cos(self.theta/2)]
        ])

        for i in range(m):
            ,,,, (Not sure where to continue)
        
        return None

    def build_binomial_tree(self, m):
        """
            m: number of time steps
            Builds a binomial tree for the underlying asset price evolution
            Will be the classical formulation of function above
        """
        return None


mean = 0
std = 1
expected_return = 0.04

dW_t = np.random.normal(loc=mean, scale=std,size = 100) 

print(dW_t)