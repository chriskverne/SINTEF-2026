import pennylane as qml
import numpy as np

# --- Configuration ---
n_precision = 5  # phase estimation precision
m_state = 4      # financial states

precision_wires = range(n_precision)
state_wires = range(n_precision, n_precision + m_state)
dev = qml.device("default.qubit", wires=n_precision + m_state)

# --- 1. Create a Realistic Financial Distribution ---
# stock prices from $0 to $15 as a normal distribution woth mean 7.5$.
N = 2**m_state
prices = np.arange(N)
bell_curve = np.exp(-(prices - 7.5)**2 / (2 * 2.5**2))
probabilities = bell_curve / np.sum(bell_curve) # Normalize to 1

# The quantum state vector requires amplitudes (sqrt of probabilities)
psi_vector = np.sqrt(probabilities)

# --- 2. Define the "Good" States ---
# Find probability that the stock price is LESS than $5.
strike_price = 5
good_indices = [i for i, price in enumerate(prices) if price < strike_price]
true_p = sum(probabilities[good_indices])


# --- 3. Build the General Q Operator ---
def create_general_Q(psi, good_idx):
    """Builds Q = S_psi * S_chi for ANY arbitrary distribution and condition."""
    num_states = len(psi)
    
    # S_chi: Flip the sign of all "good" states (The Oracle)
    S_chi = np.eye(num_states)
    for idx in good_idx:
        S_chi[idx, idx] = -1
        
    # S_psi: Reflect around the initial distribution (I - 2|psi><psi|)
    S_psi = np.eye(num_states) - 2 * np.outer(psi, psi)
    
    # Combine them
    return -(S_psi @ S_chi)


# --- 4. The QAE Circuit ---
@qml.qnode(dev)
def qae_general_circuit():
    # 1. State Preparation (A): Load the full bell curve into the m qubits
    qml.StatePrep(psi_vector, wires=state_wires)
    
    # 2. Generate the exact Q matrix for this specific distribution & condition
    Q_matrix = create_general_Q(psi_vector, good_indices)
    
    # 3. Apply QPE
    qml.QuantumPhaseEstimation(
        qml.QubitUnitary(Q_matrix, wires=state_wires),
        estimation_wires=precision_wires
    )
    
    return qml.probs(wires=precision_wires)


# --- 5. Execution ---
qae_probs = qae_general_circuit()

# Find the highest probability measurement
measured_y = np.argmax(qae_probs)

# Decode y back into probability p
estimated_theta = np.pi * measured_y / (2**n_precision)
estimated_p = np.sin(estimated_theta)**2

print(f"Target Condition:           Stock Price < ${strike_price}")
print(f"Good Indices identified:    {good_indices}")
print(f"True Classical Probability: {true_p:.4f}")
print("--------------------------------------------------")
print(f"Most likely integer (y):    {measured_y}")
print(f"Quantum Estimated 'p':      {estimated_p:.4f}")