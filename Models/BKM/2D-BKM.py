import math
import numpy as np
import pennylane as qml
import matplotlib.pyplot as plt
from scipy.stats import norm, entropy, wasserstein_distance

class BlackKarasinski2DModel:
    def __init__(self, k1, k2, theta1, theta2, var1, var2, rho, dt):
        """
        k1, k2, theta1, theta2 are lists or arrays for time-dependency.
        """
        self.k1 = k1
        self.k2 = k2
        self.theta1 = theta1
        self.theta2 = theta2
        self.var1 = var1
        self.var2 = var2
        self.rho = rho
        self.dt = dt

    def compute_step_probabilities(self, num_steps):
        joint_probs = {}
        
        for step in range(num_steps):
            joint_probs[step] = {}
            t = step * self.dt
            idx = int(t)
            
            # time-dependent reversion rates
            curr_k1 = self.k1#[min(idx, len(self.k1) - 1)] """CHANGE BACK LATER"""
            curr_k2 = self.k2#[min(idx, len(self.k2) - 1)]
            
            # loop over indices of both interest rates
            for j1 in range(-step, step + 1):
                for j2 in range(-step, step + 1):
                    # 1. 1D marginal probabilities for r1
                    a1 = -curr_k1 * j1 * self.dt
                    p1_u = 1/6 + (a1**2 + a1) / 2
                    p1_m = 2/3 - a1**2
                    p1_d = 1/6 + (a1**2 - a1) / 2
                    
                    # 2. 1D marginal probabilities for r2
                    a2 = -curr_k2 * j2 * self.dt
                    p2_u = 1/6 + (a2**2 + a2) / 2
                    p2_m = 2/3 - a2**2
                    p2_d = 1/6 + (a2**2 - a2) / 2
                    
                    # 3. compute the 9 joint probabilities
                    adj = self.rho / 4.0
                    
                    # corners (adjusted by correlation)
                    p_uu = (p1_u * p2_u) + adj
                    p_ud = (p1_u * p2_d) - adj
                    p_du = (p1_d * p2_u) - adj
                    p_dd = (p1_d * p2_d) + adj
                    
                    # middle edges (uncorrelated base product)
                    p_um = (p1_u * p2_m)
                    p_mu = (p1_m * p2_u)
                    p_md = (p1_m * p2_d)
                    p_dm = (p1_d * p2_m)
                    
                    # center (uncorrelated base product)
                    p_mm = (p1_m * p2_m)
                    
                    # normalize probabilities
                    eps = 1e-8
                    raw_probs = np.array([p_uu, p_um, p_ud, p_mu, p_mm, p_md, p_du, p_dm, p_dd])
                    clipped_probs = np.maximum(raw_probs, eps)
                    normalized_probs = clipped_probs / np.sum(clipped_probs)
                    
                    joint_probs[step][(j1, j2)] = {
                        "uu": normalized_probs[0],
                        "um": normalized_probs[1],
                        "ud": normalized_probs[2],
                        "mu": normalized_probs[3],
                        "mm": normalized_probs[4],
                        "md": normalized_probs[5],
                        "du": normalized_probs[6],
                        "dm": normalized_probs[7],
                        "dd": normalized_probs[8]
                    }
                    
        return joint_probs

    def quantum_trinomial_state_2d(self, n_steps=3):
        joint_probs = self.compute_step_probabilities(n_steps)

        # 1. Wire Setup (4 State Qubits, 2 Positional Registers)
        num_state_qubits = 4 
        num_pos_qubits = math.ceil(math.log2(2 * n_steps + 1))

        # s0, s1 for Rate 1 | s2, s3 for Rate 2
        state_wires = [f"s{i}" for i in range(num_state_qubits)]
        # Positional registers for Rate 1 and Rate 2
        pos1_wires = [f"p1_{i}" for i in range(num_pos_qubits)]
        pos2_wires = [f"p2_{i}" for i in range(num_pos_qubits)]
        all_wires = state_wires + pos1_wires + pos2_wires
        
        dev = qml.device("default.mixed", wires=all_wires)

        @qml.qnode(dev)
        def circuit():
            # (Skipping positional initialization for now to focus purely on state)

            for step in range(n_steps):
                if step == 0:
                    probs = joint_probs[0][(0, 0)]
                    
                    # --- RATE 1 (Marginal Probabilities) ---
                    p1_u = probs["uu"] + probs["um"] + probs["ud"]
                    p1_m = probs["mu"] + probs["mm"] + probs["md"]
                    p1_d = probs["du"] + probs["dm"] + probs["dd"]

                    # Apply Rate 1 Rotations (to s0 and s1)
                    theta_1 = 2 * np.arcsin(np.sqrt(p1_u))
                    theta_2 = 2 * np.arcsin(np.sqrt(p1_m / (p1_m + p1_d)))
                    qml.RY(theta_1, wires=state_wires[0])
                    qml.ctrl(qml.RY, control=state_wires[0], control_values=[0])(
                        theta_2, wires=state_wires[1]
                    )
                    
                    # --- RATE 2 (Conditional Probabilities) ---
                    # Condition A: IF Rate 1 went UP (|10>)
                    cond_u_u = probs["uu"] / p1_u
                    cond_u_m = probs["um"] / p1_u
                    cond_u_d = probs["ud"] / p1_u
                    # Apply controlled on s0=1, s1=0
                    t3_up = 2 * np.arcsin(np.sqrt(cond_u_u))
                    t4_up = 2 * np.arcsin(np.sqrt(cond_u_m / (cond_u_m + cond_u_d)))
                    qml.ctrl(qml.RY, control=[state_wires[0], state_wires[1]], control_values=[1, 0])(t3_up, wires=state_wires[2]) # check that r1 went up and apply up rotation for r2
                    qml.ctrl(qml.RY, control=[state_wires[0], state_wires[1], state_wires[2]], control_values=[1, 0, 0])(t4_up, wires=state_wires[3]) # check that r1 went up and r2 down to apply mid/down rotation for r2
                    
                    # Condition B: IF Rate 1 stayed MID (|01>)
                    cond_m_u = probs["mu"] / p1_m
                    cond_m_m = probs["mm"] / p1_m
                    cond_m_d = probs["md"] / p1_m
                    t3_mid = 2 * np.arcsin(np.sqrt(cond_m_u))
                    t4_mid = 2 * np.arcsin(np.sqrt(cond_m_m / (cond_m_m + cond_m_d)))
                    # Apply controlled on s0=0, s1=1
                    qml.ctrl(qml.RY, control=[state_wires[0], state_wires[1]], control_values=[0, 1])(t3_mid, wires=state_wires[2])
                    qml.ctrl(qml.RY, control=[state_wires[0], state_wires[1], state_wires[2]], control_values=[0, 1, 0])(t4_mid, wires=state_wires[3])
                    
                    # Condition C: IF Rate 1 went DOWN (|00>)
                    cond_d_u = probs["du"] / p1_d
                    cond_d_m = probs["dm"] / p1_d
                    cond_d_d = probs["dd"] / p1_d
                    t3_down = 2 * np.arcsin(np.sqrt(cond_d_u))
                    t4_down = 2 * np.arcsin(np.sqrt(cond_d_m / (cond_d_m + cond_d_d)))
                    # Apply controlled on s0=0, s1=0
                    qml.ctrl(qml.RY, control=[state_wires[0], state_wires[1]], control_values=[0, 0])(t3_down, wires=state_wires[2])
                    qml.ctrl(qml.RY, control=[state_wires[0], state_wires[1], state_wires[2]], control_values=[0, 0, 0])(t4_down, wires=state_wires[3])

                else:
                    pass # We will tackle the loop and multiplexer next!

            return qml.probs(wires=state_wires), qml.probs(wires=pos1_wires), qml.probs(wires=pos2_wires)

        return circuit()


TWO_BKM = BlackKarasinski2DModel(k1=0.1, k2=0.1, theta1=1, theta2=1, var1=0.1, var2=0.1, rho=0.5, dt=1)
_,r1, r2=TWO_BKM.compute_step_probabilities(num_steps=3)
print("Rate 1 Probabilities at Step 0:", r1)
print("Rate 2 Probabilities at Step 0:", r2)