import pennylane as qml
import numpy as np

def create_Q(p, m):
    """
    Creates the Q operator matrix for an m-qubit state register.
    Returns a (2^m x 2^m) numpy array.
    """
    N = 2**m
    
    # 1. Q0: Flips the sign of the |00...0> state
    Q0 = np.eye(N)
    Q0[0, 0] = -1
    
    # 2. Q_psi: I - 2|psi><psi|
    # Build the psi vector: sqrt(1-p) at index 0, sqrt(p) at the last index
    psi = np.zeros(N)
    psi[0] = np.sqrt(1-p)
    psi[-1] = np.sqrt(p)
    
    # np.outer does the |psi><psi| matrix multiplication
    Q_psi = np.eye(N) - 2 * np.outer(psi, psi)
    
    # 3. Combine them
    return Q_psi @ Q0

# --- Testing the Scalable Q ---
m_state = 6  # Try changing this to 4 or 5!
p_true = 0.25

dev = qml.device("default.qubit", wires=m_state)

@qml.qnode(dev)
def step2_scalable_test():
    # 1. State Preparation
    # PennyLane lets us load our mathematical vector directly into the qubits
    N = 2**m_state
    initial_state = np.zeros(N)
    initial_state[0] = np.sqrt(1-p_true)
    initial_state[-1] = np.sqrt(p_true)
    
    qml.StatePrep(initial_state, wires=range(m_state))
    
    # 2. Apply the dynamically generated Q matrix
    Q_matrix = create_Q(p_true, m_state)
    qml.QubitUnitary(Q_matrix, wires=range(m_state))
    
    return qml.state()

result_state = step2_scalable_test()

# Print only the non-zero amplitudes to make it easy to read
print(f"Testing with m = {m_state} qubits ({2**m_state} possible states)")
print("Non-zero amplitudes after 1 application of Q:")
for i, amp in enumerate(result_state):
    if np.abs(amp) > 1e-5:
        # Format the binary string to match the number of qubits
        binary_state = format(i, f'0{m_state}b')
        print(f"|{binary_state}> : {np.round(np.real(amp), 4)}")