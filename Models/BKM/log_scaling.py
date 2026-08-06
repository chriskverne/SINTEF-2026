import math
import numpy as np
import pennylane as qml
import matplotlib.pyplot as plt
from scipy.stats import norm, entropy, wasserstein_distance

# from thrid_foruth_moment import circuit

class BlackKarasinskiModel:
    def __init__(self, k, theta, var, dt):
        self.k = k
        self.theta = theta 
        self.var = var
        self.dt = dt

    def analytical_variance(self, n_steps):
        V = 0.0
        fine_steps = max(1000, n_steps * 10)
        dt_fine = (n_steps * self.dt) / fine_steps
        
        for i in range(fine_steps):
            t = i * dt_fine
            idx = int(t) 
            k_val = self.k[min(idx, len(self.k) - 1)]
            V += (-2 * k_val * V + self.var) * dt_fine
            
        return V

    def compute_step_angles(self, num_steps):
        angles = {}
        
        for step in range(num_steps):
            angles[step] = {}
            t = step * self.dt
            idx = int(t)
            current_k = self.k[min(idx, len(self.k) - 1)]
            
            for j in range(-step, step + 1):
                a = -current_k * j * self.dt
                
                p_up = 1/6 + (a**2 + a) / 2
                p_mid = 2/3 - a**2
                p_down = 1/6 + (a**2 - a) / 2

                eps = 1e-8
                p_up = max(p_up, eps)
                p_mid = max(p_mid, eps)
                p_down = max(p_down, eps)
                total_p = p_up + p_mid + p_down
                p_up /= total_p
                p_mid /= total_p
                p_down /= total_p

                theta_1 = 2 * np.arcsin(np.sqrt(p_up))
                theta_2 = 2 * np.arcsin(np.sqrt(p_mid / (p_mid + p_down)))

                angles[step][j] = (theta_1, theta_2)

        return angles
   
    def quantum_trinomial_state(self, T=3):
        angles = self.compute_step_angles(T)

        num_state_qubits = 2 
        num_pos_qubits = math.ceil(math.log2(2 * T + 1))

        state_wires = [f"s{i}" for i in range(num_state_qubits)]
        pos_wires = [f"p{i}" for i in range(num_pos_qubits)]
        all_wires = state_wires + pos_wires
        
        dev = qml.device("default.mixed", wires=all_wires)

        dim = 2 ** num_pos_qubits
        U_inc = np.roll(np.eye(dim), 1, axis=0)
        U_dec = np.roll(np.eye(dim), -1, axis=0)

        K0 = np.array([[1.0, 0.0], [0.0, 0.0]])
        K1 = np.array([[0.0, 1.0], [0.0, 0.0]])
        
        @qml.qnode(dev)
        def circuit():
            # position register initialization to |0> = -n_steps
            binary_offset = format(T, f'0{num_pos_qubits}b')
            for idx, bit in enumerate(binary_offset):
                if bit == '1':
                    qml.PauliX(wires=pos_wires[idx])

            for step in range(T):
                
                if step == 0:
                    theta_1, theta_2 = angles[0][0]
                    qml.RY(theta_1, wires=state_wires[0])
                    qml.ctrl(qml.RY, control=state_wires[0], control_values=[0])(
                        theta_2, wires=state_wires[1]
                    )
                else:
                    for j in range(-step, step + 1):
                        theta_1, theta_2 = angles[step][j] 

                        pos_val = j + T 
                        pos_bin = format(pos_val, f'0{num_pos_qubits}b')
                        ctrl_vals = [int(b) for b in pos_bin]

                        qml.ctrl(qml.RY, control=pos_wires, control_values=ctrl_vals)(
                            theta_1, wires=state_wires[0])

                        combined_ctrl_wires = pos_wires + [state_wires[0]]
                        combined_ctrl_vals = ctrl_vals + [0]
                        qml.ctrl(qml.RY, control=combined_ctrl_wires, control_values=combined_ctrl_vals)(
                            theta_2, wires=state_wires[1])

                qml.ctrl(qml.QubitUnitary, control=state_wires, control_values=[0, 0])(
                    U_dec, wires=pos_wires)
                
                qml.ctrl(qml.QubitUnitary, control=state_wires, control_values=[1, 0])(
                    U_inc, wires=pos_wires)

                if step < T - 1:
                    qml.QubitChannel([K0, K1], wires=state_wires[0])
                    qml.QubitChannel([K0, K1], wires=state_wires[1])

            return qml.probs(wires=state_wires), qml.probs(wires=pos_wires)


        #### DRAW CIRCUIT ###
        # print(f"\n--- Circuit Diagram for T={T} ---")
        # print(qml.draw(circuit)())

        print(f"\n--- Circuit Specs for T={T} ---")
        specs = qml.specs(circuit)()
        res = specs['resources']  
        print(f"Total operations: {res.num_gates}")
        print("Gate breakdown:")
        for gate, count in res.gate_types.items():
            print(f"  - {gate}: {count}")
        print(f"Circuit Depth: {res.depth}")
        print("==========================================\n")
        return circuit()

    def true_prob_dist(self, T):
        current_time = T * self.dt
        idx = int(current_time)
        current_theta = self.theta[min(idx, len(self.theta) - 1)]

        dx = np.sqrt(self.var * 3 * self.dt)
        j_values = np.arange(-T, T + 1)
        x_values = current_theta + j_values * dx
        
        mean = current_theta
        variance = self.analytical_variance(T)
        
        if variance > 0:
            probs = norm.pdf(x_values, loc=mean, scale=np.sqrt(variance))
            probs /= np.sum(probs)
        else:
            probs = np.zeros_like(x_values)
            probs[T] = 1.0 
            
        return x_values, probs

    def divergence(self, target_time):
        T = int(round(target_time / self.dt))
        
        if T == 0:
            raise ValueError("Target time is too small for the given dt. T must be >= 1.")

        _, pos_probs = self.quantum_trinomial_state(T)
        
        current_time = T * self.dt
        idx = int(current_time)
        current_theta = self.theta[min(idx, len(self.theta) - 1)]
        current_k = self.k[min(idx, len(self.k) - 1)]

        dx = np.sqrt(self.var * 3 * self.dt)
        j_values = np.arange(-T, T + 1)
        
        x_values = current_theta + j_values * dx
        
        q_probs = np.array([pos_probs[j + T] for j in j_values])
        q_probs = np.maximum(q_probs, 1e-12)
        q_probs /= np.sum(q_probs)
        
        _, t_probs = self.true_prob_dist(T)
        t_probs = np.maximum(t_probs, 1e-12)
        t_probs /= np.sum(t_probs)
        
        true_mean = current_theta
        true_var = self.analytical_variance(T)
        
        x_dense = np.linspace(x_values[0], x_values[-1], 200)
        
        if true_var > 0:
            pdf_dense = norm.pdf(x_dense, loc=true_mean, scale=np.sqrt(true_var)) * dx
        else:
            pdf_dense = np.zeros_like(x_dense)
        
        q_mean = float(np.sum(q_probs * x_values))
        q_var = float(np.sum(q_probs * ((x_values - q_mean) ** 2)))
        
        mean_err = abs(q_mean - true_mean)
        var_err = abs(q_var - true_var)
        
        kl_div = float(entropy(q_probs, t_probs))
        wass_dist = float(wasserstein_distance(x_values, x_values, q_probs, t_probs))
        fisher_rao = float(2 * np.arccos(np.clip(np.sum(np.sqrt(q_probs * t_probs)), 0.0, 1.0)))
        
        metrics = {
            "dt": self.dt,
            "target_time": target_time,
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

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        
        axes[0].bar(x_values, q_probs, width=dx * 0.8, color='royalblue', edgecolor='black', alpha=0.7, label='Quantum')
        axes[0].plot(x_dense, pdf_dense, color='red', linestyle='dashed', label='Continuous')
        axes[0].set_xlabel('State Variable (x = ln(r))')
        axes[0].set_ylabel('Probability')
        axes[0].set_title(f'Distribution Comparison (t={target_time}, dt={self.dt})')
        axes[0].legend()
        axes[0].grid(axis='y', linestyle='--', alpha=0.7)
        
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
        # plt.show() 
        plt.savefig(f'./figures/BKM_dt:{self.dt}_Time:{target_time}_nSteps:{T}_mean:{true_mean}_reversionRate:{current_k}.png')
        return metrics

dt = 0.25
k_array = [0.1, 0.1, 0.1, 0.1] 
theta_array = [2.0, 7.0, 4.0, 5.0, 5.0] 

bkm = BlackKarasinskiModel(k=k_array, theta=theta_array, var=0.1, dt=dt)
metrics = bkm.divergence(target_time=0.5)
metrics = bkm.divergence(target_time=1)
metrics = bkm.divergence(target_time=2.0)
metrics = bkm.divergence(target_time=3.0)
metrics = bkm.divergence(target_time=4.0)
metrics = bkm.divergence(target_time=5.0)