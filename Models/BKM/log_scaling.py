import math
import json
import numpy as np
import pennylane as qml
import matplotlib.pyplot as plt
from scipy.stats import norm, entropy, wasserstein_distance

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

        num_state_qubits = 2 
        num_pos_qubits = math.ceil(math.log2(2 * T + 1))

        state_wires = [f"s{i}" for i in range(num_state_qubits)]
        pos_wires = [f"p{i}" for i in range(num_pos_qubits)]
        all_wires = state_wires + pos_wires
        
        # MUST use default.mixed to simulate non-unitary tracing/resets
        dev = qml.device("default.mixed", wires=all_wires)

        # increases / decreases the position by 1 whether we are up / down / mid.
        dim = 2 ** num_pos_qubits
        U_inc = np.roll(np.eye(dim), 1, axis=0)
        U_dec = np.roll(np.eye(dim), -1, axis=0)

        # Define Kraus operators for resetting a qubit to |0>
        # K0 = |0><0|, K1 = |0><1|
        K0 = np.array([[1.0, 0.0], [0.0, 0.0]])
        K1 = np.array([[0.0, 1.0], [0.0, 0.0]])
        
        @qml.qnode(dev)
        def circuit():
            # prepare position register
            binary_offset = format(T, f'0{num_pos_qubits}b')
            for idx, bit in enumerate(binary_offset):
                if bit == '1':
                    qml.PauliX(wires=pos_wires[idx])

            # Start the classical-like random walk loop
            for step in range(T):
                
                ############## STEP 1: Prepare U(j) ####################
                if step == 0:
                    theta_1, theta_2 = angles[0]
                    qml.RY(theta_1, wires=state_wires[0])
                    qml.ctrl(qml.RY, control=state_wires[0], control_values=[0])(
                        theta_2, wires=state_wires[1]
                    )
                else:
                    for j in range(-step, step + 1):
                        theta_1, theta_2 = angles[j] 

                        pos_val = j + T 
                        pos_bin = format(pos_val, f'0{num_pos_qubits}b')
                        ctrl_vals = [int(b) for b in pos_bin]

                        # Apply RY_1(j) if pos = j
                        qml.ctrl(qml.RY, control=pos_wires, control_values=ctrl_vals)(
                            theta_1, wires=state_wires[0])

                        # Apply RY_2(j) if pos = j and s0 = 0
                        combined_ctrl_wires = pos_wires + [state_wires[0]]
                        combined_ctrl_vals = ctrl_vals + [0]
                        qml.ctrl(qml.RY, control=combined_ctrl_wires, control_values=combined_ctrl_vals)(
                            theta_2, wires=state_wires[1])

                ############## STEP 2: Update position ######################
                # IF |00> (Down): Decrement the position
                qml.ctrl(qml.QubitUnitary, control=state_wires, control_values=[0, 0])(
                    U_dec, wires=pos_wires)
                # IF |01> (Mid): Do nothing (Identity)
                # IF |10> (Up): Increment the position
                qml.ctrl(qml.QubitUnitary, control=state_wires, control_values=[1, 0])(
                    U_inc, wires=pos_wires)

                ############## STEP 3: Reset State Qubits ####################
                if step < T - 1: # skip reset on last step
                    # Trace out the state qubit and force it back to pure |0>
                    qml.QubitChannel([K0, K1], wires=state_wires[0])
                    qml.QubitChannel([K0, K1], wires=state_wires[1])

            return qml.probs(wires=state_wires), qml.probs(wires=pos_wires)

        return circuit()

    def true_prob_dist(self, T):
        dx = np.sqrt(self.var * 3 * self.dt)
        j_values = np.arange(-T, T + 1)
        x_values = self.theta + j_values * dx
        
        t = T * self.dt
        
        if self.k == 0:
            mean = self.theta
            variance = self.var * t
        else:
            mean = self.theta
            variance = (self.var / (2 * self.k)) * (1 - np.exp(-2 * self.k * t))
        
        probs = norm.pdf(x_values, loc=mean, scale=np.sqrt(variance))
        probs /= np.sum(probs)
        
        return x_values, probs

    def divergence(self, T):
        # --- 1. DATA PREPARATION ---
        _, pos_probs = self.quantum_trinomial_state(T)
        
        # Grid spacing and state space
        dx = np.sqrt(self.var * 3 * self.dt)
        j_values = np.arange(-T, T + 1)
        x_values = self.theta + j_values * dx
        
        # Quantum probabilities
        q_probs = np.array([pos_probs[j + T] for j in j_values])
        q_probs = np.maximum(q_probs, 1e-12)
        q_probs /= np.sum(q_probs)
        
        # Analytical discrete probabilities
        _, t_probs = self.true_prob_dist(T)
        t_probs = np.maximum(t_probs, 1e-12)
        t_probs /= np.sum(t_probs)
        
        # Analytical continuous parameters
        t = T * self.dt
        true_mean = self.theta
        true_var = self.var * t if self.k == 0 else (self.var / (2 * self.k)) * (1 - np.exp(-2 * self.k * t))
        
        x_dense = np.linspace(x_values[0], x_values[-1], 200)
        pdf_dense = norm.pdf(x_dense, loc=true_mean, scale=np.sqrt(true_var)) * dx
        
        # --- 2. MOMENT & DISTANCE CALCULATIONS ---
        # Empirical moments from quantum distribution
        q_mean = float(np.sum(q_probs * x_values))
        q_var = float(np.sum(q_probs * ((x_values - q_mean) ** 2)))
        
        # Absolute errors
        mean_err = abs(q_mean - true_mean)
        var_err = abs(q_var - true_var)
        
        # Distance metrics
        kl_div = float(entropy(q_probs, t_probs))
        wass_dist = float(wasserstein_distance(x_values, x_values, q_probs, t_probs))
        fisher_rao = float(2 * np.arccos(np.clip(np.sum(np.sqrt(q_probs * t_probs)), 0.0, 1.0)))
        
        # Structured metrics dictionary
        metrics = {
            "dt": self.dt,
            "T": T,
            "kl_divergence": kl_div,
            "wasserstein_distance": wass_dist,
            "fisher_rao_distance": fisher_rao,
            "quantum_mean": q_mean,
            "true_mean": true_mean,
            "mean_error": mean_err,
            "quantum_var": q_var,
            "true_var": true_var,
            "var_error": var_err
        }

        # --- 3. PLOTTING INDIVIDUAL STEP ---
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        
        # Left Panel: Distribution Comparison
        axes[0].bar(x_values, q_probs, width=dx * 0.8, color='royalblue', edgecolor='black', alpha=0.7, label='Quantum')
        axes[0].plot(x_dense, pdf_dense, color='red', linestyle='dashed', label='Continuous')
        axes[0].set_xlabel('State Variable (x = ln(r))')
        axes[0].set_ylabel('Probability')
        axes[0].set_title(f'Distribution Comparison (dt={self.dt}, T={T})')
        axes[0].legend()
        axes[0].grid(axis='y', linestyle='--', alpha=0.7)
        
        # Right Panel: Text Metrics Summary
        axes[1].axis('off')
        summary_text = (
            f"--- Distance Metrics ---\n"
            f"KL Divergence:   {kl_div:.6f}\n"
            f"Wasserstein Dist:{wass_dist:.6f}\n"
            f"Fisher-Rao Dist: {fisher_rao:.6f}\n\n"
            f"--- Moment Errors ---\n"
            f"Mean Error:      {mean_err:.6f}\n"
            f"  (Q: {q_mean:.4f} | True: {true_mean:.4f})\n\n"
            f"Var Error:       {var_err:.6f}\n"
            f"  (Q: {q_var:.4f} | True: {true_var:.4f})"
        )
        axes[1].text(0.1, 0.5, summary_text, ha='left', va='center', fontsize=11, family='monospace')
        axes[1].set_title('Metrics & Moments Summary', fontsize=14)
        
        plt.tight_layout()
        plt.show() 
        
        return metrics

# dt_configs = [1,0.5,0.25,0.125]
# T_configs = [1, 2, 4, 8]

T=35
dt_configs = [1/T]
T_configs = [T]

results_history = []

for dt, T in zip(dt_configs, T_configs):
    bkm = BlackKarasinskiModel(k=0.1, theta=0.5, var=0.1, dt=dt)
    metrics = bkm.divergence(T=T)
    results_history.append(metrics)
    
    # # Save step-by-step JSON
    # json_data = {
    #     "parameters": {
    #         "dt": dt,
    #         "T": T,
    #         "k": bkm.k,
    #         "theta": bkm.theta,
    #         "var": bkm.var
    #     },
    #     "metrics": metrics
    # }
    
    # file_path = f"./figures/dt={dt}_T={T}_k={bkm.k}.json"
    # with open(file_path, "w") as f:
    #     json.dump(json_data, f, indent=4)
        
    # print(f"Completed dt={dt}, T={T} -> Saved image & JSON.")
