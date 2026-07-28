"""
Black-Karasinski trinomial tree -> quantum state preparation, simulated with
matrix product states (MPS) instead of a dense statevector.

Only the *position* register (node index j) is measured. The per-step branch
qubits are still in the circuit (they carry the path history) but we never
ask for their joint distribution, which is what keeps this tractable.

Why MPS works here:
    The past only talks to the future through the current position j, which
    takes 2T+1 values. So the Schmidt rank across any time-cut is <= 2T+1 and
    an MPS with max_bond_dim >= 2T+1 is EXACT (no truncation error).
"""

import math

import numpy as np
import matplotlib.pyplot as plt
import pennylane as qml


class BKTrinomialQuantum:
    def __init__(self, k, var, dt):
        self.k = k
        self.var = var          # only needed to map j -> rate, not for the walk
        self.dt = dt

    # ------------------------------------------------------------------
    # 1. Branch probabilities and the rotation angles that encode them
    # ------------------------------------------------------------------
    def _branch_probs(self, j):
        """Hull-White style trinomial branching probabilities at node j."""
        a = -self.k * j * self.dt
        p_up = 1 / 6 + (a**2 + a) / 2
        p_mid = 2 / 3 - a**2
        p_down = 1 / 6 + (a**2 - a) / 2

        eps = 1e-12
        p_up, p_mid, p_down = max(p_up, eps), max(p_mid, eps), max(p_down, eps)
        s = p_up + p_mid + p_down
        return p_up / s, p_mid / s, p_down / s

    def compute_step_angles(self, T):
        """
        angles[j] = (theta_1, theta_2) for the 2-qubit branch encoding
            |10> = up, |01> = mid, |00> = down
        theta_1: up vs not-up.   theta_2: mid vs down, given not-up.
        """
        angles = {}
        max_j = T - 1                       # |j| can be at most T-1 when branching
        for j in range(-max_j, max_j + 1):
            p_up, p_mid, p_down = self._branch_probs(j)
            theta_1 = 2 * np.arcsin(np.sqrt(p_up))
            theta_2 = 2 * np.arcsin(np.sqrt(p_mid / (p_mid + p_down)))
            angles[j] = (theta_1, theta_2)
        return angles

    # ------------------------------------------------------------------
    # 2. Wire layout
    # ------------------------------------------------------------------
    def _layout(self, T):
        """
        An MPS is a LINE, so wire order = cost. Every step-k gate couples
        branch pair k to the position register, so we park the position
        register in the middle of the branch pairs to halve the average
        interaction distance.
        """
        n_pos = math.ceil(math.log2(2 * T + 1))
        state = [f"s{i}" for i in range(2 * T)]
        pos = [f"p{i}" for i in range(n_pos)]
        cut = 2 * (T // 2)                  # split between two branch pairs
        ordered = state[:cut] + pos + state[cut:]
        return state, pos, ordered

    # ------------------------------------------------------------------
    # 3. Controlled +/-1 on the position register
    # ------------------------------------------------------------------
    @staticmethod
    def _shift(pos, sign, ctrl_wires, ctrl_vals):
        """
        Ripple increment/decrement, big-endian (pos[0] = MSB).
        Written as MultiControlledX ladders rather than one dense
        QubitUnitary: tensor-network devices choke on wide dense gates,
        and MCX decomposes into 2-qubit gates the MPS can absorb.
        """
        n = len(pos)
        order = range(n) if sign > 0 else reversed(range(n))
        for i in order:
            lower = list(pos[i + 1:])
            ctrls = lower + list(ctrl_wires)
            vals = [1] * len(lower) + list(ctrl_vals)
            if ctrls:
                qml.MultiControlledX(wires=ctrls + [pos[i]], control_values=vals)
            else:
                qml.PauliX(wires=pos[i])

    # ------------------------------------------------------------------
    # 4. Device
    # ------------------------------------------------------------------
    @staticmethod
    def _device(wires, backend, max_bond_dim, **kw):
        if backend == "mps-gpu":            # cuTensorNet
            return qml.device("lightning.tensor", wires=wires, method="mps",
                              max_bond_dim=max_bond_dim, **kw)
        if backend == "mps-cpu":            # quimb
            return qml.device("default.tensor", wires=wires, method="mps",
                              max_bond_dim=max_bond_dim, **kw)
        if backend == "statevector":        # reference, small T only
            return qml.device("default.qubit", wires=wires)
        raise ValueError(f"unknown backend {backend!r}")

    # ------------------------------------------------------------------
    # 5. The circuit
    # ------------------------------------------------------------------
    def position_probs(self, T, backend="mps-cpu", max_bond_dim=None, **dev_kw):
        angles = self.compute_step_angles(T)
        state, pos, ordered = self._layout(T)
        n_pos = len(pos)

        if max_bond_dim is None:
            max_bond_dim = 2 * T + 2        # >= 2T+1 => exact

        dev = self._device(ordered, backend, max_bond_dim, **dev_kw)

        @qml.qnode(dev)
        def circuit():
            # offset the register so logical j = 0 sits at integer T
            for idx, bit in enumerate(format(T, f"0{n_pos}b")):
                if bit == "1":
                    qml.PauliX(wires=pos[idx])

            # first branch always happens at j = 0
            t1, t2 = angles[0]
            qml.RY(t1, wires=state[0])
            qml.ctrl(qml.RY, control=[state[0]], control_values=[0])(t2, wires=state[1])

            for step in range(1, T + 1):
                # --- apply the jump that was just prepared ---
                s0, s1 = state[2 * (step - 1)], state[2 * (step - 1) + 1]
                self._shift(pos, -1, [s0, s1], [0, 0])   # |00> = down
                self._shift(pos, +1, [s0, s1], [1, 0])   # |10> = up
                #                                          |01> = mid = identity

                # --- prepare the next jump, conditioned on where we are ---
                if step < T:
                    c0, c1 = state[2 * step], state[2 * step + 1]
                    for j in range(-step, step + 1):
                        t1, t2 = angles[j]
                        bits = [int(b) for b in format(j + T, f"0{n_pos}b")]
                        qml.ctrl(qml.RY, control=pos, control_values=bits)(t1, wires=c0)
                        qml.ctrl(qml.RY, control=pos + [c0],
                                 control_values=bits + [0])(t2, wires=c1)

            # NOTE: tensor-network devices don't implement qml.probs (a full
            # probability vector isn't a natural MPS quantity). We ask for one
            # basis-state projector per j instead -- 2T+1 cheap contractions of
            # the same MPS, and it works on every backend.
            out = []
            for j in range(-T, T + 1):
                bits = [int(b) for b in format(j + T, f"0{n_pos}b")]
                proj = qml.prod(*[qml.Projector(np.array([b]), wires=w)
                                  for b, w in zip(bits, pos)])
                out.append(qml.expval(proj))
            return tuple(out)

        probs = np.array(circuit(), dtype=float).ravel()
        return np.arange(-T, T + 1), probs

    # ------------------------------------------------------------------
    # 6. Exact classical reference (sanity check the MPS truncation)
    # ------------------------------------------------------------------
    def classical_position_probs(self, T):
        dist = {0: 1.0}
        for _ in range(T):
            nxt = {}
            for j, p in dist.items():
                pu, pm, pd = self._branch_probs(j)
                nxt[j + 1] = nxt.get(j + 1, 0.0) + p * pu
                nxt[j] = nxt.get(j, 0.0) + p * pm
                nxt[j - 1] = nxt.get(j - 1, 0.0) + p * pd
            dist = nxt
        j_vals = np.arange(-T, T + 1)
        return j_vals, np.array([dist.get(int(j), 0.0) for j in j_vals])

    # ------------------------------------------------------------------
    # 7. Plot
    # ------------------------------------------------------------------
    def plot_position_states(self, T, backend="mps-cpu", max_bond_dim=None,
                             show_reference=True, savepath=None, **dev_kw):
        j_vals, probs = self.position_probs(T, backend=backend,
                                            max_bond_dim=max_bond_dim, **dev_kw)

        fig, ax = plt.subplots(figsize=(11, 5))
        ax.bar(j_vals, probs, color="royalblue", edgecolor="black",
               label=f"MPS ({backend})")

        if show_reference:
            _, ref = self.classical_position_probs(T)
            ax.plot(j_vals, ref, "o--", color="darkorange", linewidth=1.5,
                    markersize=5, label="exact classical")
            err = np.max(np.abs(probs - ref))
            ax.set_title(f"Position distribution (T={T})   max |MPS - exact| = {err:.2e}")
        else:
            ax.set_title(f"Position distribution (T={T})")

        ax.set_xlabel("Final position j")
        ax.set_ylabel("Probability")
        ax.set_xticks(j_vals[:: max(1, len(j_vals) // 21)])
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        ax.legend()

        for j, p in zip(j_vals, probs):
            if p > 0.005:
                ax.text(j, p + 0.005, f"{p:.3f}", ha="center", fontsize=8)

        plt.tight_layout()
        if savepath:
            plt.savefig(savepath, dpi=150)
        return fig, ax


if __name__ == "__main__":
    bkm = BKTrinomialQuantum(k=0.0, var=0.1, dt=1.0)
    bkm.plot_position_states(T=8, backend="mps-cpu", savepath="position_T8.png")
    plt.show()