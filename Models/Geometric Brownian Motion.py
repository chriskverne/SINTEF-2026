import numpy as np
import pennylane as qml
import matplotlib.pyplot as plt

class RiskMeasures:
    @staticmethod 
    def maximum(gbm):
        # Allocate gbm.m risk factor qubits + 1 risk measure qubit
        dev = qml.device('default.qubit', wires=gbm.m + 1)
        rm_qubit = gbm.m
        
        @qml.qnode(dev)
        def circuit():
            gbm.state_preparation_gates(wires=range(gbm.m))
                
            # 'wires' takes control wires first, followed by the target wire at the end.
            qml.MultiControlledX(
                wires=list(range(gbm.m)) + [rm_qubit], 
                control_values=[1] * gbm.m
            )
            
            # 3. Measure ONLY the Risk Measure qubit
            return qml.probs(wires=rm_qubit)
            
        return circuit()

    @staticmethod 
    def minimum(gbm):
        """
        Calculates P(S_min) by checking if all paths are 'down' (|00...0>).
        """
        dev = qml.device('default.qubit', wires=gbm.m + 1)
        rm_qubit = gbm.m
        
        @qml.qnode(dev)
        def circuit():
            # 1. State Prep (Gate D) - Reusing the GBM class method!
            gbm.state_preparation_gates(wires=range(gbm.m))
                
            # Flips rm_qubit ONLY if all control_wires are exactly '0'
            qml.MultiControlledX(
                wires=list(range(gbm.m)) + [rm_qubit], 
                control_values=[0] * gbm.m
            )
            
            return qml.probs(wires=rm_qubit)
            
        return circuit()

class GeometricBrownianMotion:
    def __init__(self, S0, mu, sigma, dt, m):
        """
        S0: Initial price
        mu: expected return (drift)
        sigma: standard deviation (volatility)
        dt: time step size
        m: number of time steps (qubits)
        """
        self.S0 = S0
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.m = m

        # up/down factors + probabilities + quantum rotation angle
        self.u = np.exp(self.sigma * np.sqrt(self.dt))
        self.d = 1.0 / self.u
        self.q = (self.u * np.exp(self.mu * self.dt) - 1) / (self.u**2 - 1)

        # q = sin^2(theta/2) -> theta = 2 * arcsin(sqrt(q))
        self.theta = 2 * np.arcsin(np.sqrt(self.q))

    def state_preparation_gates(self, wires):
        """
        Creates binomial tree in |psi>
        """
        for i in wires:
            qml.RY(self.theta, wires=i)

    def create_quantum_state(self):
        """
        returns quantum state representing the price distribution at T = m*dt
        """
        dev = qml.device('default.qubit', wires=self.m)
        
        @qml.qnode(dev)
        def preparation():
            self.state_preparation_gates(wires=range(self.m))
            return qml.state()
        
        return preparation()

    def plot_prices(self, state):
        probs = np.abs(state)**2 
        outcomes = np.zeros(self.m + 1)
        for i in range(len(probs)):
            outcomes[bin(i).count('1')] += probs[i]

        plt.figure(figsize=(10, 6))
        plt.bar(np.arange(self.m + 1), outcomes, color='#1f77b4', edgecolor='black', alpha=0.8)
        plt.xticks(np.arange(self.m + 1), [f'${self.S0 * self.u**i * self.d**(self.m - i):.2f}' for i in range(self.m + 1)], rotation=45)
        plt.xlabel(f'Terminal Price after {self.m} steps')
        plt.ylabel('Probability')
        plt.title('Quantum Simulated Price Distribution')
        plt.tight_layout()
        plt.show()

std = 0.20
expected_return = 0.08
step_size = 1.0
m_qubits = 10 
GBM = GeometricBrownianMotion(S0=100, mu=expected_return, sigma=std, dt=step_size, m=m_qubits)

# Get full state distribution
get_state = GBM.create_quantum_state()
GBM.plot_prices(get_state)

# --- Evaluate Extreme Risk Measures ---
max_probs = RiskMeasures.maximum(GBM)
print(f"P(S_max) [All 'Up' moves]   : {max_probs[1]:.6%}")

min_probs = RiskMeasures.minimum(GBM)
print(f"P(S_min) [All 'Down' moves] : {min_probs[1]:.6%}")