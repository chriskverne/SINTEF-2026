import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. Setup & Classical Data Generation
# ==========================================
n_qubits = 5
layers = 5
n_states = 2 ** n_qubits  # 16 possible states (0 to 15)
epochs = 500

dev = qml.device("default.qubit", wires=n_qubits)

# Let's generate 1000 raw samples of historical data (X)
# We'll use a normal distribution centered at 7.5 with a spread of 2.0
np.random.seed(42)
X_samples = np.random.normal(loc=7.5, scale=2.0, size=1000)
X_samples = np.clip(X_samples, 0, n_states - 1) # Keep within 0-15 bounds

# We completely skip building P_target. We just find the moments!
mu_target = np.mean(X_samples)
var_target = np.var(X_samples)

print(f"Target Mean: {mu_target:.4f}")
print(f"Target Variance: {var_target:.4f}")

# ==========================================
# 2. The Quantum Circuit (Ansatz)
# ==========================================
def ansatz(weights):
    layers = weights.shape[0]
    
    # Start in a uniform superposition
    for i in range(n_qubits):
        qml.Hadamard(wires=i)

    # Apply parameterized rotations and entanglement
    for layer in range(layers):
        for i in range(n_qubits):
            qml.RY(weights[layer, i, 0], wires=i)
            qml.RX(weights[layer, i, 1], wires=i)
        
        # Ring of CNOTs for correlation/entanglement
        for i in range(n_qubits):
            qml.CNOT(wires=[i, (i + 1) % n_qubits])

@qml.qnode(dev)
def circuit(weights):
    ansatz(weights)
    # We return the probability distribution of the states
    return qml.probs(wires=range(n_qubits))

# ==========================================
# 3. The Moment Matching Loss Function
# ==========================================
# The integer values each quantum state represents (0 to 15)
state_values = np.arange(n_states)

def cost(weights):
    probs = circuit(weights)
    
    # 1st Moment: Quantum Mean E[X]
    q_mean = np.sum(probs * state_values)
    
    # 2nd Moment: Quantum Variance E[X^2] - E[X]^2
    q_var = np.sum(probs * (state_values ** 2)) - q_mean ** 2
    
    # Objective: Match the target moments We need to scale these somehow so one doesnt compete or overwhelm the other
    loss_mean = (q_mean - mu_target) ** 2
    loss_var = (q_var - var_target) ** 2
    
    # Weighting them equally for now
    return 0.70*loss_mean + 0.3 *loss_var

# ==========================================
# 4. Training Loop (Adam Optimizer)
# ==========================================
# Initialize random weights: (layers, n_qubits, 2 parameters per qubit)
weights = np.random.normal(0, np.pi, (layers, n_qubits, 2), requires_grad=True)

opt = qml.AdamOptimizer(stepsize=0.1)

print("\nStarting Training...")
for i in range(epochs):
    weights, current_cost = opt.step_and_cost(cost, weights)
    
    if (i + 1) % 20 == 0:
        probs = circuit(weights)
        curr_mean = np.sum(probs * state_values)
        curr_var = np.sum(probs * (state_values ** 2)) - curr_mean ** 2
        print(f"Epoch {i+1:3d} | Cost: {current_cost:.4f} | Q-Mean: {curr_mean:.2f} | Q-Var: {curr_var:.2f}")

# ==========================================
# 5. Plotting the Results
# ==========================================
final_probs = circuit(weights)

plt.figure(figsize=(10, 6))
# Plot classical histogram just for visual reference (the AI never trained on this shape!)
plt.hist(X_samples, bins=np.arange(n_states+1)-0.5, density=True, alpha=0.3, color='gray', label="Classical Data $X$ (Hidden from AI)")
# Plot the Quantum Distribution
plt.bar(state_values, final_probs, alpha=0.8, color='blue', edgecolor='black', label="Trained Quantum State")

plt.axvline(mu_target, color='red', linestyle='dashed', linewidth=2, label=f"Target Mean ({mu_target:.2f})")
plt.title("Quantum Moment Matching (Mean + Variance Only)")
plt.xlabel("State $|x\\rangle$ (Portfolio Value)")
plt.ylabel("Probability")
plt.xticks(state_values)
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.show()