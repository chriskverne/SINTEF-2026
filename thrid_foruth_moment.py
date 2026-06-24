import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. Setup & Target Moments
# ==========================================
n_qubits = 6
n_states = 2 ** n_qubits
dev = qml.device("default.qubit", wires=n_qubits)

# Generate the same classical data
np.random.seed(42)
X_samples = np.random.normal(loc=7.5, scale=2.0, size=1000)
X_samples = np.clip(X_samples, 0, n_states - 1)

# Calculate Targets: Mean, Variance, Skewness (unstandardized), Kurtosis (unstandardized)
mu_target = np.mean(X_samples)
var_target = np.var(X_samples)
skew_target = np.mean((X_samples - mu_target)**3)
kurt_target = np.mean((X_samples - mu_target)**4)

print(f"Target Mean:  {mu_target:.4f}")
print(f"Target Var:   {var_target:.4f}")
print(f"Target Skew:  {skew_target:.4f}")
print(f"Target Kurt:  {kurt_target:.4f}")

# ==========================================
# 2. Quantum Circuit (Same as before)
# ==========================================
def ansatz(weights):
    layers = weights.shape[0]
    for i in range(n_qubits):
        qml.Hadamard(wires=i)
    for layer in range(layers):
        for i in range(n_qubits):
            qml.RY(weights[layer, i, 0], wires=i)
            qml.RX(weights[layer, i, 1], wires=i)
        for i in range(n_qubits):
            qml.CNOT(wires=[i, (i + 1) % n_qubits])

@qml.qnode(dev)
def circuit(weights):
    ansatz(weights)
    return qml.probs(wires=range(n_qubits))

# ==========================================
# 3. The 4-Moment Loss Function
# ==========================================
state_values = np.arange(n_states)

# We MUST weight the higher moments down, otherwise X^4 dominates the loss
w_mean = 1.0
w_var  = 0.3
w_skew = 0.05   # Scaled down 
w_kurt = 0.001  # Heavily scaled down

def cost(weights):
    probs = circuit(weights)
    
    # Calculate Quantum Moments
    q_mean = np.sum(probs * state_values)
    q_var  = np.sum(probs * ((state_values - q_mean) ** 2))
    q_skew = np.sum(probs * ((state_values - q_mean) ** 3))
    q_kurt = np.sum(probs * ((state_values - q_mean) ** 4))
    
    # Calculate Loss Components
    loss_mean = w_mean * (q_mean - mu_target) ** 2
    loss_var  = w_var  * (q_var - var_target) ** 2
    loss_skew = w_skew * (q_skew - skew_target) ** 2
    loss_kurt = w_kurt * (q_kurt - kurt_target) ** 2
    
    return loss_mean + loss_var + loss_skew + loss_kurt

# ==========================================
# 4. Training Loop
# ==========================================
layers = 6
weights = np.random.normal(0, np.pi, (layers, n_qubits, 2), requires_grad=True)
opt = qml.AdamOptimizer(stepsize=0.25)

# Increased epochs because optimization is harder now!
epochs = 1500 

print("\nStarting Training (4 Moments)...")
for i in range(epochs):
    weights, current_cost = opt.step_and_cost(cost, weights)
    
    if (i + 1) % 50 == 0:
        print(f"Epoch {i+1:3d} | Cost: {current_cost:.4f}")

# ==========================================
# 5. Plotting
# ==========================================
final_probs = circuit(weights)

plt.figure(figsize=(10, 6))
plt.hist(X_samples, bins=np.arange(n_states+1)-0.5, density=True, alpha=0.3, color='gray', label="Classical Data $X$")
plt.bar(state_values, final_probs, alpha=0.8, color='blue', edgecolor='black', label="Trained Quantum State (4 Moments)")

plt.axvline(mu_target, color='red', linestyle='dashed', linewidth=2, label="Target Mean")
plt.title("Quantum Moment Matching (Mean + Var + Skew + Kurt)")
plt.xlabel("State $|x\\rangle$ (Portfolio Value)")
plt.ylabel("Probability")
plt.xticks(state_values)
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.show()