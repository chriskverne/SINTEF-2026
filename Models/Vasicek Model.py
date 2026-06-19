# """
# For modelling interest rate risk factors
# they forced $\delta r = \sqrt{3 Var}$ I think we can make this more general
# Sticking to the 3 state seems better than making it grow altough maybe we can make it grow as well
# """

# import numpy as np
# import pennylane as qml
# import matplotlib.pyplot as plt

# class VasicekInterestModel:
#     def __init__(self, m):
#         self.m = m
        
#         # define 9 transition probabilites (table 4)
#         self.states = {
#             'high': {'high': 19/24, 'mid':  4/24, 'low':  1/24},
#             'mid':  {'high':  4/24, 'mid': 16/24, 'low':  4/24},
#             'low':  {'high':  1/24, 'mid':  4/24, 'low': 19/24},
#         }
    
#     def mid_angles(self, current_state):
#         row = self.states[current_state]
#         q_not_mid = 1 - row['mid']
#         theta_mid  = 2 * np.arcsin(np.sqrt(q_not_mid))                # mid vs not-mid
#         theta_high = 2 * np.arcsin(np.sqrt(row['high'] / q_not_mid))  # high vs low
#         return theta_mid, theta_high
    
#     def high_angles(self, current_state):
#         row = self.states[current_state]
#         q_not_high = 1 - row['high']
#         theta_high = 2 * np.arcsin(np.sqrt(q_not_high))                # high vs not-high
#         theta_mid  = 2 * np.arcsin(np.sqrt(row['mid'] / q_not_high))   # mid vs low
#         return theta_high, theta_mid

#     def low_angles(self, current_state):
#         row = self.states[current_state]
#         q_not_low = 1 - row['low']
#         theta_low  = 2 * np.arcsin(np.sqrt(q_not_low))                 # low vs not-low
#         theta_high = 2 * np.arcsin(np.sqrt(row['high'] / q_not_low))   # high vs mid
#         return theta_low, theta_high

#     def read_gate(self, state_wires, rf_wires):
#         # state_wires = [s_high, s_mid, s_low], one-hot: exactly one is |1>
#         rf0, rf1 = rf_wires
#         for frm, s in zip(('high', 'mid', 'low'), state_wires):
#             theta_mid, theta_high = self.mid_angles(frm)
#             qml.ctrl(qml.RY(theta_mid,  wires=rf0), control=s)         # only when at level `frm`
#             qml.ctrl(qml.RY(theta_high, wires=rf1), control=[s, rf0])  # ...and only if not-mid    

#     def state_preparation_gates(self, wires):
#         theta_mid, theta_high = self.mid_angles('mid')
#         qml.RY(theta_mid, wires=wires[0])                              # mid vs not-mid
#         qml.ctrl(qml.RY(theta_high, wires=wires[1]), control=wires[0]) # high vs low, only if not-mid

#     def create_quantum_state(self):
#         dev = qml.device('default.qubit', wires=self.m)
        
#         @qml.qnode(dev)
#         def preparation():
#             self.state_preparation_gates(wires=range(self.m))
#             return qml.state()
        
#         return preparation()




import numpy as np
import pennylane as qml


class VasicekInterestModel:
    """Quantum scenario generation for the bounded trinomial (Vasicek) interest-rate
    tree from arXiv:2303.09682, built bottom-up:

      mid_angles  -> turn one transition row into two coin angles (chain rule)
      read_gate   -> apply the right rotations conditioned on the current level
      write_gate  -> decode the rf pair into the one-hot level register
      state_preparation_gates -> chain m steps, erasing each level before the next

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
            qml.ctrl(qml.RY(theta_mid,  wires=rf0), control=s)         # only when at level frm
            qml.ctrl(qml.RY(theta_high, wires=rf1), control=[s, rf0])  # ...and only if not-mid

    def write_gate(self, state_wires, rf_wires):
        s_high, s_mid, s_low = state_wires
        rf0, rf1 = rf_wires
        qml.ctrl(qml.PauliX(s_mid),  control=[rf0],      control_values=[0])
        qml.ctrl(qml.PauliX(s_high), control=[rf0, rf1], control_values=[1, 1])
        qml.ctrl(qml.PauliX(s_low),  control=[rf0, rf1], control_values=[1, 0])

    def state_preparation_gates(self, state_wires, rf_pairs):
        qml.PauliX(state_wires[1])                     # start at mid (r0 = b)
        prev_rf = None
        for rf in rf_pairs:
            self.read_gate(state_wires, rf)
            if prev_rf is None:
                qml.PauliX(state_wires[1])             # erase initial mid
            else:
                self.write_gate(state_wires, prev_rf)  # erase previous level (self-inverse)
            self.write_gate(state_wires, rf)           # write new level
            prev_rf = rf


def _classical_final(model):
    P = np.array([[model.states[a][b] for b in ('high', 'mid', 'low')]
                  for a in ('high', 'mid', 'low')])
    d = np.array([0.0, 1.0, 0.0])
    for _ in range(model.m):
        d = d @ P
    return d


if __name__ == "__main__":
    for m in (1, 2, 3, 4):
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
        c = _classical_final(model)
        print(f"m={m}  quantum  high={q[0]:.4f} mid={q[1]:.4f} low={q[2]:.4f}")
        print(f"      classical high={c[0]:.4f} mid={c[1]:.4f} low={c[2]:.4f}")