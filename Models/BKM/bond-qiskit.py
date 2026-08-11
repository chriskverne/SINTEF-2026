"""
Black-Karasinski quantum trinomial tree, simulated with Matrix Product States
(Qiskit Aer `matrix_product_state`) at a chosen bond dimension chi.

Qubit layout (dilated / "sequentially generated" MPS ordering):
[coin_0 coin_0'] [coin_1 coin_1'] ... [coin_{T-1} coin_{T-1}'] [pos_0 ... pos_{n-1}]
<------------------ emitted in time order -------------------> <-- "bond" register

NOTE: reusing one coin pair via mid-circuit `reset` does NOT work here.  Aer's
MPS is a pure-state simulator, so `reset` is measure+conditional-X: it samples a
single trajectory rather than averaging the ensemble, and the saved
probabilities collapse onto one branch.  The dilated layout below is exact.

Requirements: pip install qiskit qiskit-aer numpy scipy matplotlib
"""

import os
import math
import time
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm, entropy, wasserstein_distance
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import RYGate
from qiskit_aer import AerSimulator

FIGDIR = "./bond_figures"


def _aer_basis(method, _cache={}):
    """Aer's gate set, filtered to the gates `transpile(basis_gates=...)` accepts."""
    if method not in _cache:
        from qiskit.circuit.library.standard_gates import get_standard_gate_name_mapping
        std = set(get_standard_gate_name_mapping())
        target = AerSimulator(method=method).target
        _cache[method] = [n for n in target.operation_names if n in std]
    return _cache[method]


class BlackKarasinskiModel:
    def __init__(self, k, theta, var, dt):
        self.k = k
        self.theta = theta
        self.var = var
        self.dt = dt

    # ------------------------------------------------------------------
    # model
    # ------------------------------------------------------------------
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
                p_up = 1 / 6 + (a ** 2 + a) / 2
                p_mid = 2 / 3 - a ** 2
                p_down = 1 / 6 + (a ** 2 - a) / 2

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

    def grid(self, T):
        """Lattice x-values and the spacing dx at step T."""
        idx = int(T * self.dt)
        theta_t = self.theta[min(idx, len(self.theta) - 1)]
        dx = np.sqrt(self.var * 3 * self.dt)
        return theta_t + np.arange(-T, T + 1) * dx, dx, theta_t

    def true_prob_dist(self, T):
        x_values, _, mean = self.grid(T)
        variance = self.analytical_variance(T)
        if variance > 0:
            probs = norm.pdf(x_values, loc=mean, scale=np.sqrt(variance))
            probs /= np.sum(probs)
        else:
            probs = np.zeros_like(x_values)
            probs[T] = 1.0
        return x_values, probs

    # ------------------------------------------------------------------
    # circuit
    # ------------------------------------------------------------------
    @staticmethod
    def _controlled_increment(qc, ctrls, pos, sign):
        """Modular +-1 on the register `pos`, controlled on `ctrls`."""
        n = len(pos)
        if sign > 0:
            for i in range(n - 1, 0, -1):
                qc.mcx(ctrls + pos[:i], pos[i])
            qc.mcx(ctrls, pos[0])
        else:
            qc.mcx(ctrls, pos[0])
            for i in range(1, n):
                qc.mcx(ctrls + pos[:i], pos[i])

    def build_circuit(self, T):
        n_pos = math.ceil(math.log2(2 * T + 1))
        n_coin = 2 * T                      # a fresh coin pair per step
        n_total = n_coin + n_pos
        pos = [n_coin + i for i in range(n_pos)]   # pos[0] = LSB

        qc = QuantumCircuit(n_total, name=f"BKM_T{T}")

        # position register initialised to the value T (i.e. j = 0)
        for i in range(n_pos):
            if (T >> i) & 1:
                qc.x(pos[i])

        angles = self.compute_step_angles(T)

        for step in range(T):
            s0, s1 = 2 * step, 2 * step + 1

            if step == 0:
                th1, th2 = angles[0][0]
                qc.ry(th1, s0)
                qc.x(s0)
                qc.append(RYGate(th2).control(1), [s0, s1])
                qc.x(s0)
            else:
                for j in range(-step, step + 1):
                    th1, th2 = angles[step][j]
                    v = j + T
                    zeros = [pos[i] for i in range(n_pos) if not ((v >> i) & 1)]
                    for q in zeros:
                        qc.x(q)
                    # RY(th1) on coin0, controlled on pos == v
                    qc.append(RYGate(th1).control(n_pos), pos + [s0])
                    # RY(th2) on coin1, controlled on pos == v AND coin0 == 0
                    qc.x(s0)
                    qc.append(RYGate(th2).control(n_pos + 1), pos + [s0, s1])
                    qc.x(s0)
                    for q in zeros:
                        qc.x(q)

            # coin == |00>  -> down move (decrement)
            qc.x(s0); qc.x(s1)
            self._controlled_increment(qc, [s0, s1], pos, -1)
            qc.x(s0); qc.x(s1)

            # coin == |10>  -> up move (increment);  |01> -> middle (no move)
            qc.x(s1)
            self._controlled_increment(qc, [s0, s1], pos, +1)
            qc.x(s1)


        return qc, pos

    # ------------------------------------------------------------------
    # simulation
    # ------------------------------------------------------------------
    def position_distribution(self, T, chi=None, method="matrix_product_state",
                              report_bond_dim=False, _cache={}):
        """
        chi = None  -> no bond-dimension cap (exact, up to Aer's threshold)
        chi = int   -> truncate every SVD to at most `chi` singular values
        """
        key = (T, method, report_bond_dim)
        if key not in _cache:
            qc, pos = self.build_circuit(T)

            # Transpile against a *basis-gate list*, not the backend object.
            # Passing the backend makes Qiskit check the circuit width against
            # Aer's target, which is hard-capped at 63 qubits for every method.
            t0 = time.time()
            tqc = transpile(qc, basis_gates=_aer_basis(method),
                            optimization_level=1)
            # Save instructions are appended *after* transpiling: they are not
            # standard gates and cannot be passed through `basis_gates`.
            # No coupling map -> trivial layout -> qubit indices are unchanged.
            tqc.save_probabilities(pos, label="probabilities")
            if report_bond_dim:
                tqc.save_matrix_product_state(label="mps")
            print(f"  transpiled T={T}: {qc.num_qubits} qubits, "
                  f"{tqc.size()} gates, depth {tqc.depth()} "
                  f"({time.time() - t0:.1f}s)")
            _cache[key] = (tqc, len(pos))

        tqc, n_pos = _cache[key]

        opts = {}
        if chi is not None:
            opts["matrix_product_state_max_bond_dimension"] = int(chi)
            opts["matrix_product_state_truncation_threshold"] = 1e-16
        sim = AerSimulator(method=method, **opts)
        # sim = AerSimulator(method=method, device="GPU", **opts)

        t0 = time.time()
        result = sim.run(tqc, shots=1).result()
        runtime = time.time() - t0

        data = result.data(0)
        probs = np.asarray(data["probabilities"], dtype=float)
        probs = np.maximum(probs, 0.0)
        probs /= probs.sum()                             # renormalise

        bond = None
        if report_bond_dim and "mps" in data:
            lambdas = data["mps"][1]
            bond = max((len(l) for l in lambdas), default=1)

        return probs[: 2 * T + 1], runtime, bond

    # ------------------------------------------------------------------
    # plot (metrics computed inline)
    # ------------------------------------------------------------------
    def plot(self, target_time, chi=None,
             method="matrix_product_state", show=False):
        T = int(round(target_time / self.dt))
        if T == 0:
            raise ValueError("Target time is too small for the given dt; T must be >= 1.")

        q_probs, runtime, _ = self.position_distribution(
            T, chi=chi, method=method)

        x_values, dx, true_mean = self.grid(T)
        q_probs = np.maximum(np.asarray(q_probs, float), 1e-12)
        q_probs /= q_probs.sum()
        _, t_probs = self.true_prob_dist(T)
        t_probs = np.maximum(t_probs, 1e-12)
        t_probs /= t_probs.sum()

        true_var = self.analytical_variance(T)
        q_mean = float(np.sum(q_probs * x_values))
        q_var = float(np.sum(q_probs * (x_values - q_mean) ** 2))

        m = {
            "chi": chi, # Saved as null in JSON if chi is None
            "n_steps": T,
            "runtime": runtime,
            "kl": float(entropy(q_probs, t_probs)),
            "wass": float(wasserstein_distance(x_values, x_values, q_probs, t_probs)),
            "fr": float(2 * np.arccos(np.clip(np.sum(np.sqrt(q_probs * t_probs)), 0, 1))),
            "q_mean": q_mean, "true_mean": float(true_mean), "mean_err": abs(q_mean - true_mean),
            "q_var": q_var, "true_var": float(true_var), "var_err": abs(q_var - true_var),
            "dx": float(dx),
            "x_values": x_values.tolist(),
            "q_probs": q_probs.tolist(),
            "t_probs": t_probs.tolist()
        }

        os.makedirs(FIGDIR, exist_ok=True)
        
        # Save dictionary to JSON for future replotting
        json_out = f"{FIGDIR}/BKM_dt{self.dt}_t{target_time}_chi{chi}.json"
        with open(json_out, "w") as f:
            json.dump(m, f, indent=4)
        print(f"Saved data -> {json_out}")

        # --- figure -----------------------------------------------------
        x_dense = np.linspace(x_values[0], x_values[-1], 200)
        pdf_dense = (norm.pdf(x_dense, loc=true_mean, scale=np.sqrt(true_var)) * dx
                     if true_var > 0 else np.zeros_like(x_dense))

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        axes[0].bar(x_values, q_probs, width=dx * 0.8, color="royalblue",
                    edgecolor="black", alpha=0.7, label="Quantum (MPS)")
        axes[0].plot(x_dense, pdf_dense, "r--", label="Continuous")
        axes[0].set_xlabel("State Variable (x = ln r)")
        axes[0].set_ylabel("Probability")
        axes[0].set_title(f"t={target_time}, dt={self.dt}, chi={chi}")
        axes[0].legend(); axes[0].grid(axis="y", ls="--", alpha=0.7)

        # Print "Exact" in place of None for formatting if chi wasn't set
        print_chi = chi if chi is not None else "Exact"

        axes[1].axis("off")
        axes[1].text(0.05, 0.5,
                     f"--- Distance Metrics ---\n"
                     f"KL Divergence:    {m['kl']:.6f}\n"
                     f"Wasserstein Dist: {m['wass']:.6f}\n"
                     f"Fisher-Rao Dist:  {m['fr']:.6f}\n\n"
                     f"--- Moment Errors ---\n"
                     f"Mean Error:       {m['mean_err']:.6f}\n"
                     f"  (Q: {m['q_mean']:.4f} | True: {m['true_mean']:.4f})\n\n"
                     f"Var Error:        {m['var_err']:.6f}\n"
                     f"  (Q: {m['q_var']:.4f} | True: {m['true_var']:.4f})\n\n"
                     f"--- Run ---\n"
                     f"chi:              {print_chi}\n"
                     f"steps T:          {T}\n"
                     f"sim runtime:      {runtime:.2f} s",
                     ha="left", va="center", fontsize=11, family="monospace")
        axes[1].set_title("Metrics & Moments Summary", fontsize=14)
        plt.tight_layout()
        out = f"{FIGDIR}/BKM_dt{self.dt}_t{target_time}_chi{chi}.png"
        plt.savefig(out, dpi=130)
        plt.show() if show else plt.close(fig)
        print(f"Saved plot -> {out}")
        return m


if __name__ == "__main__":
    dt = 0.25
    k_array = np.full(20, 0.1)
    theta_array = np.full(20, 2.0)

    bkm = BlackKarasinskiModel(k=k_array, theta=theta_array, var=0.1, dt=dt)

    target_time = 10
    chi_values = [5, 10, 20, 30]  # None runs exact simulation

    for chi in chi_values:
        print(f"\n--- Running for chi = {chi} ---")
        m = bkm.plot(target_time=target_time, chi=chi)
        print(f"Finished chi = {chi}. KL Divergence: {m['kl']:.6f}")