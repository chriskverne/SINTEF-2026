import numpy as np
import pennylane as qml
import matplotlib.pyplot as plt

class GeometricBrownianMotion:
    def __init__(self, S0, mu, sigma, dt, m):
        """
            S0: Initial price
            mu: expected return (drift)
            sigma: standard deviation (volatility)
            dt: time step size
            m: number of time steps
        """
        self.S0 = S0
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.m = m

        # Compute up/down factors + probabilities + quantum rotation angle
        self.u = np.exp(self.sigma * np.sqrt(self.dt))
        self.d = 1.0 / self.u
        self.q = (self.u * np.exp(self.mu * self.dt) - 1) / (self.u**2 - 1)

        # q = sin^2(theta/2) -> theta = 2 * arcsin(sqrt(q))
        self.theta = 2 * np.arcsin(np.sqrt(self.q))

    def create_quantum_state(self):
        """
            m: number of qubits (time steps)
            Returns a quantum state representing the price distribution at time T = m*dt
        """

        # Creates quantum state with m qubits
        dev = qml.device('default.qubit', wires=self.m)
        @qml.qnode(dev)

        def preparation():
            for i in range(self.m):
                qml.RY(self.theta, wires=i)
            
            return qml.state()
        
        return preparation()
    
    def plot_state_distribution(self, state):
        probs = np.abs(state)**2 

        # We only have m+1 outcomes as ddu = dud = udd
        outcomes = np.zeros(self.m + 1)

        # Iterate over state and count no of 1's. We use this as an index in outcomes
        for i in range(len(probs)):
            outcomes[bin(i).count('1')] += probs[i]

        plt.bar(np.arange(self.m + 1), outcomes)
        plt.xticks(np.arange(self.m + 1), [f'{i} up' for i in range(self.m + 1)])
        plt.show()

    def plot_prices(self, state):
        probs = np.abs(state)**2 

        # We only have m+1 outcomes as ddu = dud = udd
        outcomes = np.zeros(self.m + 1)

        # Iterate over state and count no of 1's. We use this as an index in outcomes
        for i in range(len(probs)):
            outcomes[bin(i).count('1')] += probs[i]

        plt.bar(np.arange(self.m + 1), outcomes)
        plt.xticks(np.arange(self.m + 1), [f'{self.S0 * self.u**i * self.d**(self.m - i):.2f}$' for i in range(self.m + 1)])
        plt.show()
    
    def single_qubit_amplitude(self, qubit_index):
        """
            Add a single qubit to prepare the state (sqrt(1-p) |0> + sqrt(p) |1>) where |0>/|1> represents when a condition is met
        """


std = 0.20
expected_return = 0.08
step_size = 1.0
m_qubits = 15
GBM = GeometricBrownianMotion(S0=100, mu=expected_return, sigma=std, dt=step_size, m=m_qubits)
get_state = GBM.create_quantum_state()
GBM.plot_prices(get_state)
GBM.plot_state_distribution(get_state)