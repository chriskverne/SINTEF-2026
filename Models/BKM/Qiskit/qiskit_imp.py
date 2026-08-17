"""
Black-Karasinski quantum trinomial tree, simulated with Matrix Product States (and without)
(Qiskit Aer `matrix_product_state`) at a sweep of bond dimensions chi.

Key idea
--------
Qubit layout:
[coin_0 coin_0'] [coin_1 coin_1'] ... [coin_{T-1} coin_{T-1}'] [pos_0 ... pos_{n-1}]
<------------------- emitted in time order --------------->    <-- "bond" register
This is the *sequentially generated* MPS ordering: the position
register acts as the bond and each step emits two physical sites.  The exact
Schmidt rank across any coin-coin cut is therefore bounded by 2^n_pos, i.e. by
the number of tree nodes -- so the state is genuinely MPS-friendly and chi
controls how much of the tree's mixedness you keep.

Requirements:  pip install qiskit qiskit-aer numpy scipy matplotlib
"""

import os
import math
import time
# from turtle import pos
import numpy as np
# import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm, entropy, wasserstein_distance

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import RYGate
from qiskit_aer import AerSimulator

FIGDIR = "./figures"
MAX_STATEVECTOR_QUBITS = 24 

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
        chi = None  -> no bond-dimension cap (exact, up to Aer's 1e-16 threshold)
        chi = int   -> truncate every SVD to at most `chi` singular values
        """
        key = (T, method, report_bond_dim)
        if key not in _cache:
            qc, pos = self.build_circuit(T)
            qc.save_probabilities(pos, label="probabilities")
            if report_bond_dim:
                qc.save_matrix_product_state(label="mps")
            probe = AerSimulator(method=method)
            # transpile once, reuse for every chi (transpilation is chi-independent)
            _cache[key] = (transpile(qc, probe, optimization_level=1), len(pos))


        tqc, n_pos = _cache[key]

        opts = {}
        if chi is not None:
            opts["matrix_product_state_max_bond_dimension"] = int(chi)
            opts["matrix_product_state_truncation_threshold"] = 1e-16
        sim = AerSimulator(method=method, **opts)

        t0 = time.time()
        result = sim.run(tqc, shots=1).result()
        runtime = time.time() - t0

        data = result.data(0)
        probs = np.asarray(data["probabilities"], dtype=float)
        probs = np.maximum(probs, 0.0)
        probs /= probs.sum() # renormalise

        bond = None
        if report_bond_dim and "mps" in data:
            lambdas = data["mps"][1]
            bond = max((len(l) for l in lambdas), default=1)

        return probs[: 2 * T + 1], runtime, bond


    def divergence(self, target_time, chi=None, plot=True):
        T = int(round(target_time / self.dt))
        if T == 0:
            raise ValueError("Target time is too small for the given dt. T must be >= 1.")

        pos_probs, _, _ = self.position_distribution(T, chi=chi)
        return self._metrics(T, target_time, pos_probs, chi=chi, plot=plot)

    def _metrics(self, T, target_time, q_probs, chi=None, plot=False, ref_probs=None):
        current_time = T * self.dt
        idx = int(current_time)
        current_theta = self.theta[min(idx, len(self.theta) - 1)]

        dx = np.sqrt(self.var * 3 * self.dt)
        j_values = np.arange(-T, T + 1)
        x_values = current_theta + j_values * dx

        q_probs = np.maximum(np.asarray(q_probs, float), 1e-12)
        q_probs = q_probs / q_probs.sum()

        _, t_probs = self.true_prob_dist(T)
        t_probs = np.maximum(t_probs, 1e-12)
        t_probs /= t_probs.sum()

        true_mean = current_theta
        true_var = self.analytical_variance(T)

        q_mean = float(np.sum(q_probs * x_values))
        q_var = float(np.sum(q_probs * (x_values - q_mean) ** 2))

        m = {
            "chi": chi if chi is not None else np.inf,
            "dt": self.dt,
            "target_time": target_time,
            "n_steps": T,
            "kl_divergence": float(entropy(q_probs, t_probs)),
            "wasserstein_distance": float(
                wasserstein_distance(x_values, x_values, q_probs, t_probs)),
            "fisher_rao_distance": float(
                2 * np.arccos(np.clip(np.sum(np.sqrt(q_probs * t_probs)), 0.0, 1.0))),
            "quantum_mean": q_mean,
            "true_mean": true_mean,
            "mean_error": abs(q_mean - true_mean),
            "quantum_var": q_var,
            "true_var": true_var,
            "var_error": abs(q_var - true_var),
        }

        if ref_probs is not None:
            r = np.maximum(np.asarray(ref_probs, float), 1e-12)
            r /= r.sum()
            m["tv_vs_exact"] = float(0.5 * np.abs(q_probs - r).sum())
            m["kl_vs_exact"] = float(entropy(q_probs, r))
            m["fisher_rao_vs_exact"] = float(
                2 * np.arccos(np.clip(np.sum(np.sqrt(q_probs * r)), 0.0, 1.0)))

        if plot:
            self._plot_single(x_values, q_probs, dx, true_mean, true_var,
                              m, target_time, T, chi)
        return m

    def _plot_single(self, x_values, q_probs, dx, true_mean, true_var, m,
                     target_time, T, chi):
        os.makedirs(FIGDIR, exist_ok=True)
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

        axes[1].axis("off")
        axes[1].text(0.05, 0.5,
                     f"--- Distance Metrics ---\n"
                     f"KL Divergence:    {m['kl_divergence']:.6f}\n"
                     f"Wasserstein Dist: {m['wasserstein_distance']:.6f}\n"
                     f"Fisher-Rao Dist:  {m['fisher_rao_distance']:.6f}\n\n"
                     f"--- Moment Errors ---\n"
                     f"Mean Error:       {m['mean_error']:.6f}\n"
                     f"  (Q: {m['quantum_mean']:.4f} | True: {m['true_mean']:.4f})\n\n"
                     f"Var Error:        {m['var_error']:.6f}\n"
                     f"  (Q: {m['quantum_var']:.4f} | True: {m['true_var']:.4f})",
                     ha="left", va="center", fontsize=11, family="monospace")
        axes[1].set_title("Metrics & Moments Summary", fontsize=14)
        plt.tight_layout()
        plt.savefig(f"{FIGDIR}/BKM_dt{self.dt}_t{target_time}_chi{chi}.png", dpi=130)
        plt.close(fig)

    # # ------------------------------------------------------------------
    # #  bond-dimension study
    # # ------------------------------------------------------------------
    # def bond_dimension_study(self, target_time, chis=(2, 4, 8, 10, 16, 32, 64),
    #                          show_chis=None):
    #     T = int(round(target_time / self.dt))
    #     n_pos = math.ceil(math.log2(2 * T + 1))
    #     n_qubits = 2 * T + n_pos
    #     print(f"\n{'='*74}")
    #     print(f"Bond-dimension study | t={target_time}  dt={self.dt}  T={T} steps")
    #     print(f"  qubits: {2*T} coin (dilated resets) + {n_pos} position = {n_qubits}")
    #     print(f"  theoretical exact bond dimension <= 2^{n_pos} = {2**n_pos}")
    #     print(f"{'='*74}")

    #     if n_qubits <= MAX_STATEVECTOR_QUBITS:
    #         ref, t_ref, _ = self.position_distribution(T, method="statevector")
    #         ref_label = "statevector"
    #     else:
    #         ref, t_ref, _ = self.position_distribution(T, chi=None)
    #         ref_label = "MPS (uncapped)"
    #     _, _, bond = self.position_distribution(T, chi=None, report_bond_dim=True)
    #     print(f"reference: {ref_label} in {t_ref:.2f}s | "
    #           f"max bond dim actually reached by exact MPS: {bond}")

    #     rows = []
    #     for chi in chis:
    #         p, rt, _ = self.position_distribution(T, chi=chi)
    #         m = self._metrics(T, target_time, p, chi=chi, ref_probs=ref)
    #         m["runtime"] = rt
    #         m["probs"] = p
    #         rows.append(m)

    #     m_ex = self._metrics(T, target_time, ref, chi=None, ref_probs=ref)
    #     m_ex["runtime"] = t_ref
    #     m_ex["probs"] = ref
    #     rows.append(m_ex)

    #     hdr = f"{'chi':>6} {'TV|exact':>11} {'KL|exact':>11} {'KL|normal':>11} " \
    #           f"{'Wass|normal':>12} {'mean err':>10} {'var err':>10} {'t[s]':>7}"
    #     print("\n" + hdr)
    #     print("-" * len(hdr))
    #     for m in rows:
    #         c = "exact" if np.isinf(m["chi"]) else f"{int(m['chi'])}"
    #         print(f"{c:>6} {m['tv_vs_exact']:11.3e} {m['kl_vs_exact']:11.3e} "
    #               f"{m['kl_divergence']:11.3e} {m['wasserstein_distance']:12.3e} "
    #               f"{m['mean_error']:10.3e} {m['var_error']:10.3e} {m['runtime']:7.2f}")

    #     self._plot_study(rows, target_time, T, n_pos, show_chis)
    #     return rows

    # ------------------------------------------------------------------
    def bond_dimension_study(self, target_time, chis=(2, 4, 8, 10, 16, 32, 64),
                             show_chis=None):
        T = int(round(target_time / self.dt))
        n_pos = math.ceil(math.log2(2 * T + 1))
        n_qubits = 2 * T + n_pos
        print(f"\n{'='*74}")
        print(f"Bond-dimension study | t={target_time}  dt={self.dt}  T={T} steps")
        print(f"  qubits: {2*T} coin (dilated resets) + {n_pos} position = {n_qubits}")
        print(f"  theoretical exact bond dimension <= 2^{n_pos} = {2**n_pos}")
        print(f"{'='*74}")

        # --- exact reference -------------------------------------------------
        if n_qubits <= MAX_STATEVECTOR_QUBITS:
            ref, t_ref, _ = self.position_distribution(T, method="statevector")
            ref_label = "statevector"
        else:
            ref, t_ref, _ = self.position_distribution(T, chi=None)
            ref_label = "MPS (uncapped)"
            
        _, _, bond = self.position_distribution(T, chi=None, report_bond_dim=True)
        print(f"reference: {ref_label} in {t_ref:.2f}s | "
              f"max bond dim actually reached by exact MPS: {bond}")

        rows = []
        for chi in chis:
            p, rt, _ = self.position_distribution(T, chi=chi)
            m = self._metrics(T, target_time, p, chi=chi, ref_probs=ref)
            m["runtime"] = rt
            m["probs"] = p
            rows.append(m)

        m_ex = self._metrics(T, target_time, ref, chi=None, ref_probs=ref)
        m_ex["runtime"] = t_ref
        m_ex["probs"] = ref
        m_ex["max_bond_reached"] = bond  # Explicitly store the actual max bond
        rows.append(m_ex)

        hdr = f"{'chi':>6} {'TV|exact':>11} {'KL|exact':>11} {'KL|normal':>11} " \
              f"{'Wass|normal':>12} {'mean err':>10} {'var err':>10} {'t[s]':>7}"
        print("\n" + hdr)
        print("-" * len(hdr))
        for m in rows:
            c = "exact" if np.isinf(m["chi"]) else f"{int(m['chi'])}"
            print(f"{c:>6} {m['tv_vs_exact']:11.3e} {m['kl_vs_exact']:11.3e} "
                  f"{m['kl_divergence']:11.3e} {m['wasserstein_distance']:12.3e} "
                  f"{m['mean_error']:10.3e} {m['var_error']:10.3e} {m['runtime']:7.2f}")

        self._plot_study(rows, target_time, T, n_pos, show_chis)
        return rows

    def _plot_study(self, rows, target_time, T, n_pos, show_chis):
        os.makedirs(FIGDIR, exist_ok=True)
        idx = int(T * self.dt)
        theta_t = self.theta[min(idx, len(self.theta) - 1)]
        dx = np.sqrt(self.var * 3 * self.dt)
        x_values = theta_t + np.arange(-T, T + 1) * dx
        x_dense = np.linspace(x_values[0], x_values[-1], 300)
        true_var = self.analytical_variance(T)
        pdf_dense = norm.pdf(x_dense, loc=theta_t, scale=np.sqrt(true_var)) * dx

        finite = [m for m in rows if np.isfinite(m["chi"])]
        exact = rows[-1]
        chis = [int(m["chi"]) for m in finite]
        actual_max_bond = exact.get("max_bond_reached", 2**n_pos)

        # Fix duplicate chi values in display list
        if show_chis is None:
            raw_chis = [finite[0]["chi"], finite[len(finite) // 2]["chi"], finite[-1]["chi"]]
            show_chis = sorted(list(set([int(c) for c in raw_chis])))

        # Create a 2-panel figure with linear scale
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

        # ------------------------------------------------------------------
        # Panel 1: Position Distributions
        # ------------------------------------------------------------------
        ax1 = axes[0]
        ax1.plot(x_dense, pdf_dense, "k--", lw=2, label="Continuous (Analytic)")
        ax1.plot(x_values, exact["probs"], "o-", color="black", lw=2.2, ms=5, 
                 label=rf"Exact Quantum ($\chi_{{max}}={actual_max_bond}$)")
        
        cmap = plt.cm.viridis(np.linspace(0.1, 0.85, len(show_chis)))
        for c, col in zip(show_chis, cmap):
            m = next(r for r in finite if int(r["chi"]) == c)
            ax1.plot(x_values, m["probs"], "s--", color=col, ms=4, label=rf"MPS $\chi={c}$")

        ax1.set_xlabel("x = ln r", fontsize=11)
        ax1.set_ylabel("Probability", fontsize=11)
        ax1.set_title(f"Position Distribution (t={target_time}, T={T} steps)", fontsize=12)
        ax1.legend(fontsize=9, loc="upper right")
        ax1.grid(True, ls="--", alpha=0.5)

        # ------------------------------------------------------------------
        # Panel 2: Error Metrics vs Bond Dimension (Linear Scale)
        # ------------------------------------------------------------------
        ax2 = axes[1]
        
        kl_vals = [m["kl_divergence"] for m in finite]
        wass_vals = [m["wasserstein_distance"] for m in finite]
        fr_vals = [m["fisher_rao_distance"] for m in finite]

        ax2.plot(chis, kl_vals, "o-", color="C0", lw=1.8, ms=6, label="KL Divergence")
        ax2.plot(chis, wass_vals, "s-", color="C1", lw=1.8, ms=6, label="Wasserstein Distance")
        ax2.plot(chis, fr_vals, "^-", color="C2", lw=1.8, ms=6, label="Fisher-Rao Distance")

        # Exact quantum baselines
        ax2.axhline(exact["kl_divergence"], color="C0", ls=":", lw=1.5, alpha=0.6, label="KL (Exact floor)")
        ax2.axhline(exact["wasserstein_distance"], color="C1", ls=":", lw=1.5, alpha=0.6, label="Wasserstein (Exact floor)")
        ax2.axhline(exact["fisher_rao_distance"], color="C2", ls=":", lw=1.5, alpha=0.6, label="Fisher-Rao (Exact floor)")

        # Exact quantum points plotted at the actual maximum bond dimension
        ax2.plot([actual_max_bond], [exact["kl_divergence"]], "o", color="C0", ms=8)
        ax2.plot([actual_max_bond], [exact["wasserstein_distance"]], "s", color="C1", ms=8)
        ax2.plot([actual_max_bond], [exact["fisher_rao_distance"]], "^", color="C2", ms=8)

        # Vertical line for observed max bond dimension
        ax2.axvline(actual_max_bond, color="red", ls="--", lw=1.5, label=f"Max Bond Reached ({actual_max_bond})")

        ax2.set_xlabel(r"Bond Dimension $\chi$", fontsize=11)
        ax2.set_ylabel("Distance to Continuous Model", fontsize=11)
        ax2.set_title("Model Metrics Convergence vs Bond Dimension", fontsize=12)
        ax2.legend(fontsize=8.5, loc="upper right")
        ax2.grid(True, ls="--", alpha=0.5)

        plt.tight_layout()
        out = f"{FIGDIR}/BKM_bonddim_t{target_time}_dt{self.dt}.png"
        plt.savefig(out, dpi=130)
        plt.close(fig)
        print(f"\nSaved updated study plot -> {out}")

    def bond_dimension_scaling_study(self, max_T=20):
        """
        Evaluates and plots how the exact maximum bond dimension scales 
        with the number of steps T and target time t.
        """
        T_values = list(range(1, max_T + 1))
        actual_bonds = []

        print(f"\n{'='*60}")
        print(f"Tracking Exact Bond Dimension Scaling up to T={max_T}")
        print(f"{'='*60}")
        print(f"{'T':>4} | {'t':>5} | {'n_pos':>5} | {'Total Qubits':>12} | {'Actual Max Bond':>15}")
        print("-" * 60)

        for T in T_values:
            t = T * self.dt
            n_pos = math.ceil(math.log2(2 * T + 1))
            n_qubits = 2 * T + n_pos
            
            # Force MPS method to guarantee we get a bond dimension report
            _, _, bond = self.position_distribution(
                T, chi=None, method="matrix_product_state", report_bond_dim=True
            )
            actual_bonds.append(bond)
            
            print(f"{T:4d} | {t:5.2f} | {n_pos:5d} | {n_qubits:12d} | {bond:15d}")

        os.makedirs(FIGDIR, exist_ok=True)
        fig, ax1 = plt.subplots(figsize=(9, 5.5))
        
        ax1.plot(T_values, actual_bonds, 'o-', color='royalblue', lw=2, ms=6, label="Actual Max Bond Dimension (Qiskit MPS)")
        
        ax1.set_xlabel("Number of Steps (T)", fontsize=11)
        ax1.set_ylabel(r"Maximum Bond Dimension $\chi$", fontsize=11)
        ax1.set_title("Exact MPS Bond Dimension Scaling vs Circuit Depth", fontsize=13)
        ax1.grid(True, ls="--", alpha=0.6)
        ax1.legend(fontsize=10, loc="upper left")
        
        # Add secondary X-axis on top to show the target time 't'
        ax2 = ax1.twiny()
        ax2.set_xlim(ax1.get_xlim())
        # Map the T ticks to time t ticks
        ax2.set_xticks(ax1.get_xticks())
        ax2.set_xticklabels([f"{tick * self.dt:.1f}" for tick in ax1.get_xticks()])
        ax2.set_xlabel("Target Time (t)", fontsize=11)
        
        plt.tight_layout()
        out = f"{FIGDIR}/BKM_bond_scaling_maxT{max_T}.png"
        plt.savefig(out, dpi=130)
        plt.close(fig)
        print(f"\nSaved scaling plot -> {out}")
        
        return T_values, actual_bonds

    def gate_count_study(self, max_T=10):
        print(f"\n{'='*85}")
        print(f"Gate Count Scaling Study up to T={max_T}")
        print(f"{'='*85}")
        print(f"{'T':>4} | {'Total Qubits':>12} | {'Raw Gates':>10} | {'Depth':>6} | "
              f"{'X':>5} | {'RY':>4} | {'Ctrl-X':>8} | {'Ctrl-RY':>7}")
        print("-" * 85)
        for T in range(1, max_T + 1):
            qc, _ = self.build_circuit(T)
            n_qubits = qc.num_qubits
            gate_count = qc.size()
            depth = qc.depth()
            ops = qc.count_ops()
            x_count = ops.get('x', 0)
            ry_count = ops.get('ry', 0)
            ctrl_x = sum(v for k, v in ops.items() if k in ['cx', 'ccx', 'mcx'] or (k.startswith('c') and k.endswith('x')))
            ctrl_ry = sum(v for k, v in ops.items() if 'ry' in k and k != 'ry')
            print(f"{T:4d} | {n_qubits:12d} | {gate_count:10d} | {depth:6d} | "
                  f"{x_count:5d} | {ry_count:4d} | {ctrl_x:8d} | {ctrl_ry:7d}")
                        
if __name__ == "__main__":
    dt = 0.25
    k_array = np.full(20, 0.1) #[0.1, 0.1, 0.1, 0.1]
    theta_array = np.full(20, 2.0) #[2.0, 7.0, 4.0, 5.0, 5.0]

    bkm = BlackKarasinskiModel(k=k_array, theta=theta_array, var=0.1, dt=dt)

    # your original single call still works, now with an optional chi
    bkm.divergence(target_time=5, chi=None, plot=True)

    # the bond-dimension sweep
    # bkm.bond_dimension_study(target_time=2.0, chis=[2, 4, 8, 10, 16, 32, 64, 128])

    # bkm.bond_dimension_scaling_study(max_T=15)
    # scales as Bdim(T) = 4T - 2
    # bkm.gate_count_study(max_T=20)