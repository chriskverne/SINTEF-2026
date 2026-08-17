import math
from collections import defaultdict
from scipy.stats import multivariate_normal, entropy, wasserstein_distance
import matplotlib.pyplot as plt
import os
import json
import numpy as np
import pennylane as qml


class BlackKarasinski2DModel:
    """
    Two-factor Black-Karasinski trinomial tree encoded as a quantum circuit.

    Wire layout
    -----------
      s0, s1   : branch register for rate 1   (|10> = up, |01> = mid, |00> = down)
      s2, s3   : branch register for rate 2   (same encoding)
      p1_*     : positional register for rate 1, holding j1 + T  (unsigned)
      p2_*     : positional register for rate 2, holding j2 + T

    Each step:
      1. multiplexer: rotations on (s0,s1,s2,s3) controlled on the *joint*
         positional state (p1, p2), loading the 9 joint branch probabilities
         for that node,
      2. shift: p1 +/- 1 controlled on (s0,s1), p2 +/- 1 controlled on (s2,s3),
      3. reset: the four branch qubits are traced out and re-prepared in |0>
         so they can be reused by the next step.
    """

    def __init__(self, k1, k2, theta1, theta2, var1, var2, rho, dt,
                 corr_mode="legacy"):
        self.k1 = k1
        self.k2 = k2
        self.theta1 = theta1
        self.theta2 = theta2
        self.var1 = var1
        self.var2 = var2
        self.rho = rho
        self.dt = dt
        self.corr_mode = corr_mode  # "legacy" (rho/4 corner shift) or "mixture"

    # ------------------------------------------------------------------
    # Classical branch probabilities
    # ------------------------------------------------------------------
    @staticmethod
    def _marginal(a):
        p_u = 1 / 6 + (a ** 2 + a) / 2
        p_m = 2 / 3 - a ** 2
        p_d = 1 / 6 + (a ** 2 - a) / 2
        return p_u, p_m, p_d

    @staticmethod
    def _comonotone(m1, m2, flip=False):
        """
        Maximal- (or minimal-, if flip) correlation coupling of two 3-point
        marginals via the north-west corner rule. Returns a 3x3 array indexed
        [rate1_move, rate2_move] with move order (u, m, d).
        """
        order = [2, 1, 0]                       # ascending: d, m, u
        o2 = order if not flip else [0, 1, 2]
        C = np.zeros((3, 3))
        a = [m1[i] for i in order]
        b = [m2[i] for i in o2]
        i = j = 0
        ra, rb = a[0], b[0]
        while i < 3 and j < 3:
            t = min(ra, rb)
            C[order[i], o2[j]] += t
            ra -= t
            rb -= t
            if ra <= 1e-15:
                i += 1
                ra = a[i] if i < 3 else 0.0
            if rb <= 1e-15:
                j += 1
                rb = b[j] if j < 3 else 0.0
        return C

    def compute_step_probabilities(self, num_steps):
        joint_probs = {}
        keys = ["uu", "um", "ud", "mu", "mm", "md", "du", "dm", "dd"]

        for step in range(num_steps):
            joint_probs[step] = {}
            t = step * self.dt
            idx = int(t)

            # time-dependent reversion rates (kept constant for now)
            curr_k1 = self.k1  # [min(idx, len(self.k1) - 1)]
            curr_k2 = self.k2  # [min(idx, len(self.k2) - 1)]

            for j1 in range(-step, step + 1):
                for j2 in range(-step, step + 1):
                    p1_u, p1_m, p1_d = self._marginal(-curr_k1 * j1 * self.dt)
                    p2_u, p2_m, p2_d = self._marginal(-curr_k2 * j2 * self.dt)

                    if self.corr_mode == "mixture":
                        m1 = np.array([p1_u, p1_m, p1_d])
                        m2 = np.array([p2_u, p2_m, p2_d])
                        m1 = np.maximum(m1, 0); m1 /= m1.sum()
                        m2 = np.maximum(m2, 0); m2 /= m2.sum()
                        indep = np.outer(m1, m2)
                        extreme = self._comonotone(m1, m2, flip=(self.rho < 0))
                        M = (1 - abs(self.rho)) * indep + abs(self.rho) * extreme
                        raw_probs = M.reshape(-1)
                    else:
                        adj = self.rho / 4.0
                        raw_probs = np.array([
                            p1_u * p2_u + adj, p1_u * p2_m, p1_u * p2_d - adj,
                            p1_m * p2_u, p1_m * p2_m, p1_m * p2_d,
                            p1_d * p2_u - adj, p1_d * p2_m, p1_d * p2_d + adj,
                        ])

                    eps = 1e-8
                    normalized = np.maximum(raw_probs, eps)
                    normalized = normalized / np.sum(normalized)

                    joint_probs[step][(j1, j2)] = dict(zip(keys, normalized))

        return joint_probs

    # ------------------------------------------------------------------
    # Angle helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _angles(p_u, p_m, p_d):
        """RY angles for the 2-qubit encoding |10>=u, |01>=m, |00>=d."""
        eps = 1e-15
        tot = p_u + p_m + p_d
        p_u, p_m, p_d = p_u / tot, p_m / tot, p_d / tot
        theta_1 = 2 * np.arcsin(np.sqrt(np.clip(p_u, 0.0, 1.0)))
        den = p_m + p_d
        ratio = p_m / den if den > eps else 0.0
        theta_2 = 2 * np.arcsin(np.sqrt(np.clip(ratio, 0.0, 1.0)))
        return theta_1, theta_2

    def _load_node(self, probs, sw, ctrl_wires, ctrl_vals):
        """
        Prepare sum_ij sqrt(p_ij) |branch_i>|branch_j> on the four branch
        qubits, optionally conditioned on (ctrl_wires == ctrl_vals).

        Rate 1 gets the marginal, rate 2 gets the conditional -> the product
        reproduces the joint exactly.
        """
        s0, s1, s2, s3 = sw

        p1_u = probs["uu"] + probs["um"] + probs["ud"]
        p1_m = probs["mu"] + probs["mm"] + probs["md"]
        p1_d = probs["du"] + probs["dm"] + probs["dd"]

        t1, t2 = self._angles(p1_u, p1_m, p1_d)

        if ctrl_wires:
            qml.ctrl(qml.RY, control=ctrl_wires, control_values=ctrl_vals)(t1, wires=s0)
        else:
            qml.RY(t1, wires=s0)
        qml.ctrl(qml.RY, control=ctrl_wires + [s0],
                 control_values=ctrl_vals + [0])(t2, wires=s1)

        # rate 2, conditional on the branch rate 1 just took
        for code, s01 in (("u", [1, 0]), ("m", [0, 1]), ("d", [0, 0])):
            c_u, c_m, c_d = probs[code + "u"], probs[code + "m"], probs[code + "d"]
            t3, t4 = self._angles(c_u, c_m, c_d)
            qml.ctrl(qml.RY, control=ctrl_wires + [s0, s1],
                     control_values=ctrl_vals + s01)(t3, wires=s2)
            qml.ctrl(qml.RY, control=ctrl_wires + [s0, s1, s2],
                     control_values=ctrl_vals + s01 + [0])(t4, wires=s3)

    # ------------------------------------------------------------------
    # Circuit
    # ------------------------------------------------------------------
    def quantum_trinomial_state_2d(self, n_steps=3, draw=False, specs=False):
        T = n_steps
        joint_probs = self.compute_step_probabilities(T)

        num_state_qubits = 4
        num_pos_qubits = math.ceil(math.log2(2 * T + 1))

        state_wires = [f"s{i}" for i in range(num_state_qubits)]
        pos1_wires = [f"p1_{i}" for i in range(num_pos_qubits)]
        pos2_wires = [f"p2_{i}" for i in range(num_pos_qubits)]
        all_wires = state_wires + pos1_wires + pos2_wires

        dev = qml.device("default.mixed", wires=all_wires)

        dim = 2 ** num_pos_qubits
        U_inc = np.roll(np.eye(dim), 1, axis=0)
        U_dec = np.roll(np.eye(dim), -1, axis=0)

        K0 = np.array([[1.0, 0.0], [0.0, 0.0]])
        K1 = np.array([[0.0, 1.0], [0.0, 0.0]])

        def bits(j):
            return [int(b) for b in format(j + T, f"0{num_pos_qubits}b")]

        @qml.qnode(dev)
        def circuit():
            # both positional registers start at the offset origin j = 0
            for w, b in zip(pos1_wires, format(T, f"0{num_pos_qubits}b")):
                if b == "1":
                    qml.PauliX(wires=w)
            for w, b in zip(pos2_wires, format(T, f"0{num_pos_qubits}b")):
                if b == "1":
                    qml.PauliX(wires=w)

            for step in range(T):
                if step == 0:
                    # only one node, no multiplexing needed
                    self._load_node(joint_probs[0][(0, 0)], state_wires, [], [])
                else:
                    # ---- MULTIPLEXER -------------------------------------
                    # one branch of the mux per lattice node (j1, j2); the
                    # control pattern is the concatenation of both positional
                    # registers, so the rotations that fire are the ones whose
                    # (j1, j2) label matches the register contents.
                    pos_ctrl = pos1_wires + pos2_wires
                    for j1 in range(-step, step + 1):
                        b1 = bits(j1)
                        for j2 in range(-step, step + 1):
                            self._load_node(
                                joint_probs[step][(j1, j2)],
                                state_wires,
                                pos_ctrl,
                                b1 + bits(j2),
                            )

                # ---- SHIFT ----------------------------------------------
                qml.ctrl(qml.QubitUnitary, control=state_wires[:2],
                         control_values=[0, 0])(U_dec, wires=pos1_wires)
                qml.ctrl(qml.QubitUnitary, control=state_wires[:2],
                         control_values=[1, 0])(U_inc, wires=pos1_wires)

                qml.ctrl(qml.QubitUnitary, control=state_wires[2:],
                         control_values=[0, 0])(U_dec, wires=pos2_wires)
                qml.ctrl(qml.QubitUnitary, control=state_wires[2:],
                         control_values=[1, 0])(U_inc, wires=pos2_wires)

                # ---- RESET ----------------------------------------------
                if step < T - 1:
                    for w in state_wires:
                        qml.QubitChannel([K0, K1], wires=w)

            return (qml.probs(wires=pos1_wires + pos2_wires),
                    qml.probs(wires=pos1_wires),
                    qml.probs(wires=pos2_wires),
                    qml.probs(wires=state_wires))

        if draw:
            print(qml.draw(circuit, max_length=200)())
        if specs:
            res = qml.specs(circuit)()["resources"]
            print(f"  gates={res.num_gates}  depth={res.depth}  wires={len(all_wires)}")

        joint, m1, m2, sp = circuit()

        # reshape the flat joint into a (2T+1) x (2T+1) lattice in j-coordinates
        J = np.zeros((2 * T + 1, 2 * T + 1))
        for a in range(2 * T + 1):
            for b in range(2 * T + 1):
                J[a, b] = joint[a * dim + b]
        return J, np.array(m1[: 2 * T + 1]), np.array(m2[: 2 * T + 1]), sp

    # ------------------------------------------------------------------
    # Classical reference lattice (exact forward propagation)
    # ------------------------------------------------------------------
    def classical_reference(self, n_steps):
        T = n_steps
        jp = self.compute_step_probabilities(T)
        mv = {"u": +1, "m": 0, "d": -1}
        dist = {(0, 0): 1.0}
        for step in range(T):
            nxt = defaultdict(float)
            for (j1, j2), p in dist.items():
                pr = jp[step][(j1, j2)]
                for m1 in "umd":
                    for m2 in "umd":
                        nxt[(j1 + mv[m1], j2 + mv[m2])] += p * pr[m1 + m2]
            dist = dict(nxt)

        J = np.zeros((2 * T + 1, 2 * T + 1))
        for (j1, j2), p in dist.items():
            J[j1 + T, j2 + T] = p
        return J


    def analytical_covariance(self, n_steps):
        """Computes the theoretical 2x2 covariance matrix for the 2D OU process."""
        V1, V2, Cov = 0.0, 0.0, 0.0
        fine_steps = max(1000, n_steps * 10)
        dt_fine = (n_steps * self.dt) / fine_steps
        
        # In this simplified version, k1 and k2 are constants. 
        # If they become arrays later, you can index them as in your 1D code.
        for i in range(fine_steps):
            # dV1 = -2*k1*V1 + sigma1^2
            V1 += (-2 * self.k1 * V1 + self.var1) * dt_fine
            # dV2 = -2*k2*V2 + sigma2^2
            V2 += (-2 * self.k2 * V2 + self.var2) * dt_fine
            # dCov = -(k1+k2)*Cov + rho*sigma1*sigma2
            Cov += (-(self.k1 + self.k2) * Cov + self.rho * np.sqrt(self.var1 * self.var2)) * dt_fine
            
        return np.array([[V1, Cov], [Cov, V2]])

    def true_prob_dist_2d(self, T):
        """Generates the continuous bivariate Gaussian distribution."""
        dx1 = np.sqrt(self.var1 * 3 * self.dt)
        dx2 = np.sqrt(self.var2 * 3 * self.dt)
        
        j_values = np.arange(-T, T + 1)
        x1_values = self.theta1 + j_values * dx1
        x2_values = self.theta2 + j_values * dx2
        
        X1, X2 = np.meshgrid(x1_values, x2_values, indexing='ij')
        pos = np.dstack((X1, X2))
        
        mean = np.array([self.theta1, self.theta2])
        cov_matrix = self.analytical_covariance(T)
        
        # Handle t=0 or zero variance
        if cov_matrix[0,0] > 0 and cov_matrix[1,1] > 0:
            rv = multivariate_normal(mean, cov_matrix)
            probs = rv.pdf(pos)
            probs /= np.sum(probs) # Normalize to grid
        else:
            probs = np.zeros((2*T+1, 2*T+1))
            probs[T, T] = 1.0 
            
        return x1_values, x2_values, probs, mean, cov_matrix

    def divergence_2d(self, target_time):
        """Evaluates the quantum circuit against the analytical 2D distribution."""
        T = int(round(target_time / self.dt))
        if T == 0:
            raise ValueError("Target time is too small for the given dt. T must be >= 1.")

        # 1. Get Quantum Output
        print(f"Running circuit for T={T} steps...")
        Q_joint, Q_m1, Q_m2, _ = self.quantum_trinomial_state_2d(T)
        
        # 2. Get True Analytical Output
        x1_vals, x2_vals, T_joint, true_mean, true_cov = self.true_prob_dist_2d(T)
        
        # Flatten for KL divergence
        q_flat = np.maximum(Q_joint.flatten(), 1e-12)
        q_flat /= np.sum(q_flat)
        t_flat = np.maximum(T_joint.flatten(), 1e-12)
        t_flat /= np.sum(t_flat)
        
        kl_div = float(entropy(q_flat, t_flat))
        
        # Marginal Wassersteins (Computing true 2D Wasserstein is computationally heavy)
        T_m1 = np.sum(T_joint, axis=1)
        T_m2 = np.sum(T_joint, axis=0)
        w_dist_1 = float(wasserstein_distance(x1_vals, x1_vals, Q_m1, T_m1))
        w_dist_2 = float(wasserstein_distance(x2_vals, x2_vals, Q_m2, T_m2))
        
        # Empirical Quantum Moments
        X1, X2 = np.meshgrid(x1_vals, x2_vals, indexing='ij')
        q_mean_1 = np.sum(Q_joint * X1)
        q_mean_2 = np.sum(Q_joint * X2)
        
        q_var_1 = np.sum(Q_joint * (X1 - q_mean_1)**2)
        q_var_2 = np.sum(Q_joint * (X2 - q_mean_2)**2)
        q_cov = np.sum(Q_joint * (X1 - q_mean_1) * (X2 - q_mean_2))
        
        # --- Plotting ---
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Quantum Heatmap
        im1 = axes[0].pcolormesh(X1, X2, Q_joint, shading='auto', cmap='Blues')
        axes[0].set_title('Quantum 2D Distribution')
        axes[0].set_xlabel('Rate 1 (x1)')
        axes[0].set_ylabel('Rate 2 (x2)')
        fig.colorbar(im1, ax=axes[0])
        
        # True Heatmap
        im2 = axes[1].pcolormesh(X1, X2, T_joint, shading='auto', cmap='Reds')
        axes[1].set_title('Analytical 2D Distribution')
        axes[1].set_xlabel('Rate 1 (x1)')
        axes[1].set_ylabel('Rate 2 (x2)')
        fig.colorbar(im2, ax=axes[1])
        
        # Metrics Text
        axes[2].axis('off')
        summary_text = (
            f"--- Distance Metrics ---\n"
            f"KL Divergence:     {kl_div:.6f}\n"
            f"W-Dist (Rate 1):   {w_dist_1:.6f}\n"
            f"W-Dist (Rate 2):   {w_dist_2:.6f}\n\n"
            f"--- Moment Errors ---\n"
            f"Mean 1 Error:      {abs(q_mean_1 - true_mean[0]):.6f}\n"
            f"Mean 2 Error:      {abs(q_mean_2 - true_mean[1]):.6f}\n"
            f"Var 1 Error:       {abs(q_var_1 - true_cov[0,0]):.6f}\n"
            f"Var 2 Error:       {abs(q_var_2 - true_cov[1,1]):.6f}\n"
            f"Covariance Error:  {abs(q_cov - true_cov[0,1]):.6f}\n"
        )
        axes[2].text(0.1, 0.5, summary_text, ha='left', va='center', fontsize=11, family='monospace')
        axes[2].set_title(f'Summary (t={target_time}, dt={self.dt})', fontsize=14)
        
        plt.tight_layout()
        
        # Make figures directory if it doesn't exist
        # os.makedirs('./figures', exist_ok=True)
        # plt.savefig(f'./figures/BKM2D_dt{self.dt}_Time{target_time}.png')
        plt.show()
        
        return kl_div

if __name__ == "__main__":
    dt_values = [1/4]#[1, 1/2]  # Keep it small to start, as 2D multiplexers are slow!
    
    for dt in dt_values:
        print(f"\n--- Running evaluation for dt = {dt} ---")
        bkm2d = BlackKarasinski2DModel(
            k1=0.1, k2=0.1, 
            theta1=2.0, theta2=2.0, 
            var1=0.1, var2=0.1, 
            rho=0.5, 
            dt=dt
        )
        bkm2d.divergence_2d(target_time=1.0)