import numpy as np
import pennylane as qml
import matplotlib.pyplot as plt

class ReducedFormCreditRisk:
    def __init__(self, dt, T_def, target_time):
        """
        dt: timestep size
        T_def: mean time to default
        target_time: how many steps we want to take
        """
        self.dt = dt
        self.target_time = target_time
        self.T_def = T_def
        self.q = 1 - np.exp(-self.dt / self.T_def) # chance of defaulting
        self.theta = 2 * np.arcsin(np.sqrt(self.q))
        self.m = int(np.ceil(self.target_time / self.dt)) # number of steps/qubits
    
    def create_quantum_state(self):
        dev = qml.device('default.qubit', wires=self.m)
        @qml.qnode(dev)
        
        def preparation():
            for i in range(self.m):
                # t = 0, superposition of survive / default
                if i == 0:
                    qml.RY(self.theta, wires=0)
                else:
                    qml.ctrl(qml.RY(self.theta, wires=i), control=i-1, control_values=[0]) # if previous is 0 (survived) take another step
                    qml.CNOT(wires=[i-1, i]) # stay in default (1) if defaulted

            return qml.state()

        return preparation()
    
    def survival_probability(self):
        # Counts all states with atleast 1 and add's their ampltidue into rm qubit
        # We want ampltidue of |00...0>
        rm_qubit = self.m 
        dev = qml.device('default.qubit', wires=self.m + 1)
        
        @qml.qnode(dev)
        def circuit():
            for i in range(self.m):
                if i == 0:
                    qml.RY(self.theta, wires=0)
                else:
                    qml.ctrl(qml.RY(self.theta, wires=i), control=i-1, control_values=[0])
                    qml.CNOT(wires=[i-1, i])
            
            # 2. Risk measure encoding flip the rm_qubit to |1> if ALL m qubits are |0>
            control_wires = list(range(self.m))
            zero_state_values = [0] * self.m
            qml.ctrl(qml.PauliX(wires=rm_qubit), control=control_wires, control_values=zero_state_values)
            
            return qml.probs(wires=rm_qubit)
            
        return circuit()
    
    def plot_state(self):
        state = self.create_quantum_state()
        probs = np.abs(state)**2
        outcomes = np.zeros(self.m + 1)
        for i in range(len(probs)): 
            outcomes[bin(i).count('1')] += probs[i]

        plt.bar(np.arange(self.m + 1), outcomes, color='#1f77b4', edgecolor='black', alpha=0.8)
        plt.xticks(np.arange(self.m + 1), [f'{bin(i).count("1")}S, {self.m - bin(i).count("1")}D' for i in range(self.m + 1)], rotation=45)
        plt.ylabel('Probability')
        plt.tight_layout()
        print(f"Survival Chance {outcomes[0]}")
        plt.show()

if __name__ == "__main__":
    dt = 1
    T_def = 15
    target_time = 15
    RFCR = ReducedFormCreditRisk(dt=dt, T_def=T_def, target_time=target_time)
    print(RFCR.create_quantum_state())
    RFCR.plot_state()
    print(f"Survivial chance : {RFCR.survival_probability()}")