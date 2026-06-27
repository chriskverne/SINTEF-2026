import numpy as np
import pennylane as qml
import matplotlib.pyplot as plt

class MultiJumpModel:
    def __init__(self, dt, jump_rate, target_time):
        self.dt = dt
        self.target_time = target_time
        self.jump_rate = jump_rate
        # Probability of a jump in a single dt: 1 - exp(-lambda * dt)
        self.q = 1 - np.exp(-self.jump_rate * self.dt)
        self.theta = 2 * np.arcsin(np.sqrt(self.q))
        self.m = int(np.ceil(self.target_time / self.dt))
    
    def create_quantum_state(self):
        dev = qml.device('default.qubit', wires=self.m)
        @qml.qnode(dev)
        def preparation():
            for i in range(self.m):
                qml.RY(self.theta, wires=i)
            return qml.state()
        return preparation()

    def plot_state(self):
        state = self.create_quantum_state()
        probs = np.abs(state)**2
        
        # Count total jumps (total number of 1s)
        jump_counts = [bin(i).count('1') for i in range(2**self.m)]
        results = np.zeros(self.m + 1)
        for i, p in enumerate(probs):
            results[jump_counts[i]] += p

        plt.bar(np.arange(self.m + 1), results, color='#ff7f0e', edgecolor='black', alpha=0.8)
        plt.xlabel('Number of Jumps')
        plt.ylabel('Probability')
        plt.title(f'Distribution of {self.m} Timesteps (Multiple Jumps Allowed)')
        plt.show()

if __name__ == "__main__":
    # dt=1, rate=0.1, 10 steps
    model = MultiJumpModel(dt=1, jump_rate=0.05, target_time=10)
    model.plot_state()