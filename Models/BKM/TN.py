import math
import numpy as np
import pennylane as qml
import matplotlib.pyplot as plt

class BlackKarasinskiModel:
    def __init__(self, k, theta, var, dt):
        self.k = k # currently constant
        self.theta = theta # currently constant
        self.var = var
        self.dt = dt

    def compute_step_angles(self, num_steps):
        angles = {}
        
        # The maximum span of j after num_steps (where i goes up to num_steps - 1)
        max_j = num_steps - 1
        
        for j in range(-max_j, max_j + 1):
            a = -self.k * j * self.dt
            
            # Calculate raw trinomial probabilities
            p_up = 1/6 + (a**2 + a) / 2
            p_mid = 2/3 - a**2
            p_down = 1/6 + (a**2 - a) / 2

            # Prevent negative probs:
            eps = 1e-8
            p_up = max(p_up, eps)
            p_mid = max(p_mid, eps)
            p_down = max(p_down, eps)
            total_p = p_up + p_mid + p_down
            p_up /= total_p
            p_mid /= total_p
            p_down /= total_p

            # |00> = down, |01> = mid, |10> = up
            # 1. Up or not up?
            theta_1 = 2 * np.arcsin(np.sqrt(p_up))
            # 2. Mid or down?
            theta_2 = 2 * np.arcsin(np.sqrt(p_mid / (p_mid + p_down)))

            # key is position 'j'
            angles[j] = (theta_1, theta_2)

        return angles

    def quantum_trinomial_state(self, T=3):
        angles = self.compute_step_angles(T)

        num_state_qubits = 2 * T
        num_pos_qubits = math.ceil(math.log2(2 * T + 1))

        state_wires = [f"s{i}" for i in range(num_state_qubits)]
        pos_wires = [f"p{i}" for i in range(num_pos_qubits)]
        all_wires = state_wires + pos_wires
        
        # --- NEW QISKIT MPS BACKEND ---
        dev = qml.device("qiskit.aer", wires=all_wires, method="matrix_product_state")


        # increases / decreases the position by 1 whether we are up / down / mid.
        dim = 2 ** num_pos_qubits
        U_inc = np.roll(np.eye(dim), 1, axis=0)
        U_dec = np.roll(np.eye(dim), -1, axis=0)
        
        @qml.qnode(dev)
        def circuit():
            # prepare position register
            binary_offset = format(T, f'0{num_pos_qubits}b')
            for idx, bit in enumerate(binary_offset):
                if bit == '1':
                    qml.PauliX(wires=pos_wires[idx])

            ############## STEP 1: INITIAL U rot ##########################
            theta_1, theta_2 = angles[0]
            qml.RY(theta_1, wires=state_wires[0])
            qml.ctrl(qml.RY(theta_2, wires=state_wires[1]), control=state_wires[0], control_values=[0])

            # Run the loop T times
            for step in range(1, T + 1):
                ############## STEP 2: Update position ######################
                prev_s0 = state_wires[2 * (step - 1)] 
                prev_s1 = state_wires[2 * (step - 1) + 1] 
                                
                qml.ctrl(qml.QubitUnitary, control=[prev_s0, prev_s1], control_values=[0, 0])(
                    U_dec, wires=pos_wires)
                qml.ctrl(qml.QubitUnitary, control=[prev_s0, prev_s1], control_values=[1, 0])(
                    U_inc, wires=pos_wires)

                ############## STEP 3: Prepare Next U(j) ####################
                if step < T:
                    curr_s0 = state_wires[2 * step] 
                    curr_s1 = state_wires[2 * step + 1] 
                    
                    for j in range(-step, step + 1):
                        theta_1, theta_2 = angles[j] 

                        pos_val = j + T 
                        pos_bin = format(pos_val, f'0{num_pos_qubits}b')
                        ctrl_vals = [int(b) for b in pos_bin]

                        qml.ctrl(qml.RY, control=pos_wires, control_values=ctrl_vals)(
                            theta_1, wires=curr_s0)

                        combined_ctrl_wires = pos_wires + [curr_s0]
                        combined_ctrl_vals = ctrl_vals + [0]
                        qml.ctrl(qml.RY, control=combined_ctrl_wires, control_values=combined_ctrl_vals)(
                            theta_2, wires=curr_s1)

            # We can go right back to returning the exact probabilities we want!
            return qml.probs(wires=state_wires), qml.probs(wires=pos_wires)

        # The QNode returns exactly the tuple of arrays you originally designed
        return circuit()
    
    def plot_position_states(self, T):
        # Unpack the second item in the tuple (position probabilities)
        _, pos_probs = self.quantum_trinomial_state(T)
        
        # It's perfectly safe to calculate num_pos_qubits here again for scaling,
        # but the logical position calculation depends on `T`.
        j_values = []
        j_probs = []
        
        # Loop through all possible states in the position register
        for i, prob in enumerate(pos_probs):
            # i is the physical integer (pos_val). 
            # We subtract T to get the logical position (j)
            j = i - T 
            
            # We only care about valid j positions between -T and T
            if -T <= j <= T:
                j_values.append(j)
                j_probs.append(prob)
                
        plt.figure(figsize=(10, 5))
        plt.bar(j_values, j_probs, color='royalblue', edgecolor='black')
        plt.xlabel('Final Position (j)')
        plt.ylabel('Probability')
        plt.title(f'Aggregated Position Distribution (T={T})')
        plt.xticks(range(-T, T + 1)) # Force x-axis to show all j integer ticks
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add labels on top of the bars for clarity
        for i, prob in enumerate(j_probs):
            if prob > 0.001:
                plt.text(j_values[i], prob + 0.01, f'{prob:.3f}', ha='center', va='bottom', fontsize=9)
                
        plt.tight_layout()
        # plt.savefig('./figures/gpurun.png') # Commented to avoid FileNotFoundError
        plt.show()

# ==========================================
# Execution Code
# ==========================================
if __name__ == "__main__":
    print("Initializing BlackKarasinskiModel...")
    theta_val = 0.5 
    bkm = BlackKarasinskiModel(k=0.0, theta=theta_val, var=0.1, dt=1)
    
    print("Running tensor network circuit for T=4. This may take a moment...")
    bkm.plot_position_states(T=4)