import numpy as np
import pennylane as qml
import matplotlib.pyplot as plt

class VasicekInterestModel:
    """generation for the bounded trinomial (Vasicek) interest-rate tree from arXiv:2303.09682:

      mid_angles  -> turn transition row into two angles
      read_gate   -> apply the rotations based on current level
      write_gate  -> decode rf pair into register
      state_preparation_gates -> m steps, erasing each level before the next m(m...(m(|psi>)))

    Encoding (mid-pivot, uniform across all levels):
      rf0 = 0          -> mid
      rf0 = 1, rf1 = 1 -> high
      rf0 = 1, rf1 = 0 -> low
    state_wires are one-hot [s_high, s_mid, s_low].
    """

    def __init__(self, m):
        self.m = m
        self.states = {
            'high': {'high': 19/24, 'mid':  4/24, 'low':  1/24},
            'mid':  {'high':  4/24, 'mid': 16/24, 'low':  4/24},
            'low':  {'high':  1/24, 'mid':  4/24, 'low': 19/24},
        }

    def mid_angles(self, frm):
        row = self.states[frm]
        q_not_mid = 1 - row['mid']
        theta_mid  = 2 * np.arcsin(np.sqrt(q_not_mid))                # mid vs not-mid
        theta_high = 2 * np.arcsin(np.sqrt(row['high'] / q_not_mid))  # high vs low
        return theta_mid, theta_high

    def read_gate(self, state_wires, rf_wires):
        rf0, rf1 = rf_wires
        for frm, s in zip(('high', 'mid', 'low'), state_wires):
            theta_mid, theta_high = self.mid_angles(frm)
            qml.ctrl(qml.RY(theta_mid,  wires=rf0), control=s)
            qml.ctrl(qml.RY(theta_high, wires=rf1), control=[s, rf0])

    def write_gate(self, state_wires, rf_wires):
        s_high, s_mid, s_low = state_wires
        rf0, rf1 = rf_wires
        qml.ctrl(qml.PauliX(s_mid),  control=[rf0],      control_values=[0])
        qml.ctrl(qml.PauliX(s_high), control=[rf0, rf1], control_values=[1, 1])
        qml.ctrl(qml.PauliX(s_low),  control=[rf0, rf1], control_values=[1, 0])

    def state_preparation_gates(self, state_wires, rf_pairs):
        qml.PauliX(state_wires[1])
        prev_rf = None
        for rf in rf_pairs:
            self.read_gate(state_wires, rf)
            if prev_rf is None:
                qml.PauliX(state_wires[1])             # erase initial mid
            else:
                self.write_gate(state_wires, prev_rf)  # erase previous level
            self.write_gate(state_wires, rf)           # write new level
            prev_rf = rf


    def plot_state(self, state_wires, rf_pairs):
        n_wires = max(max(state_wires), max(w for pair in rf_pairs for w in pair)) + 1
        dev = qml.device('default.qubit', wires=n_wires)

        @qml.qnode(dev)
        def circ():
            self.state_preparation_gates(state_wires, rf_pairs)
            return qml.probs(wires=state_wires)

        p = circ()
        labels = ['high', 'mid', 'low']
        probs = [p[4], p[2], p[1]]

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(labels, probs, color=['#c0392b', '#7f8c8d', '#2980b9'])
        ax.set_ylabel('probability')
        ax.set_ylim(0, 1)
        ax.set_title(f'Vasicek state, m={self.m} steps')
        for i, v in enumerate(probs):
            ax.text(i, v + 0.015, f'{v:.4f}', ha='center')
        fig.tight_layout()
        plt.show()
        return fig, ax


def classical_verification(model):
    P = np.array([[model.states[a][b] for b in ('high', 'mid', 'low')]
                  for a in ('high', 'mid', 'low')])
    d = np.array([0.0, 1.0, 0.0])
    for _ in range(model.m):
        d = d @ P
    return d


if __name__ == "__main__":
    m=2
    model = VasicekInterestModel(m)
    state_wires = [0, 1, 2]
    rf_pairs = [[3 + 2 * t, 4 + 2 * t] for t in range(m)]
    model.plot_state(state_wires, rf_pairs)

    for m in (1, 2, 3, 4, 8):
        model = VasicekInterestModel(m)
        state_wires = [0, 1, 2]
        rf_pairs = [[3 + 2 * t, 4 + 2 * t] for t in range(m)]
        dev = qml.device('default.qubit', wires=3 + 2 * m)

        @qml.qnode(dev)
        def circ():
            model.state_preparation_gates(state_wires, rf_pairs)
            return qml.probs(wires=state_wires)

        p = circ()
        q = (p[4], p[2], p[1])         # one-hot |100>, |010>, |001> -> high, mid, low
        c = classical_verification(model)
        print(f"m={m}  quantum  high={q[0]:.4f} mid={q[1]:.4f} low={q[2]:.4f}")
        print(f"      classical high={c[0]:.4f} mid={c[1]:.4f} low={c[2]:.4f}")