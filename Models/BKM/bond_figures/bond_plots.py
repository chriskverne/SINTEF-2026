import os
import json
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator

t_vals = [20, 30, 40, 50, 60]
chi_vals = [5, 10, 20, 30, 40, 50, 60, 70]
dt = 0.25

fig, axes = plt.subplots(2, 5, figsize=(18, 6))

# Helper function to format lists into TikZ coordinate strings
def to_tikz_coords(x_vals, y_vals):
    return " ".join([f"({x}, {y:.6e})" for x, y in zip(x_vals, y_vals)])

for i, t in enumerate(t_vals):
    valid_chis = []
    errors_kl, errors_wass, errors_fr, var_errs, runtimes = [], [], [], [], []
    
    T = int(round(t / dt))
    n_qubits = 2 * T + math.ceil(math.log2(2 * T + 1))
    
    for chi in chi_vals:
        # if (t == 60 and chi == 70):
        #     continue

        filepath = f'./bond_figures/BKM_dt0.25_t{t}_chi{chi}.json'
        
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
                valid_chis.append(chi)
                
                errors_kl.append(data['kl'])
                errors_wass.append(data['wass'])
                errors_fr.append(data['fr'])
                var_errs.append(data['var_err'])
                runtimes.append(data['runtime'])

    if valid_chis:
        # --- PRINT TIKZ DATA ---
        print(f"\n%%% Data for Plot Column {i+1}: t={t} (Qubits: {n_qubits}) %%%")
        print("% --- Top Row (Errors) ---")
        print(f"\\addplot coordinates {{ {to_tikz_coords(valid_chis, errors_kl)} }}; % KL")
        print(f"\\addplot coordinates {{ {to_tikz_coords(valid_chis, errors_wass)} }}; % Wasserstein")
        print(f"\\addplot coordinates {{ {to_tikz_coords(valid_chis, errors_fr)} }}; % Fisher-Rao")
        print(f"\\addplot coordinates {{ {to_tikz_coords(valid_chis, var_errs)} }}; % Var Error")
        
        print("% --- Bottom Row (Runtimes) ---")
        print(f"\\addplot coordinates {{ {to_tikz_coords(valid_chis, runtimes)} }}; % Runtime")
        print("-" * 50)
        # -----------------------

        # Top Row: All Error Metrics combined
        ax_err = axes[0, i]
        ax_err.plot(valid_chis, errors_kl, marker='o', color='red', label='KL', linewidth=4, markersize=12)
        ax_err.plot(valid_chis, errors_wass, marker='s', color='green', label='Wasserstein', linewidth=4, markersize=12)
        ax_err.plot(valid_chis, errors_fr, marker='^', color='orange', label='Fisher-Rao', linewidth=4, markersize=12)
        ax_err.plot(valid_chis, var_errs, marker='D', color='purple', label='Var Error', linewidth=4, markersize=12)
        
        ax_err.set_title(f"t={t} (Qubits: {n_qubits})")
        ax_err.set_yscale('log')
        
        # Force X-axis ticks to match exactly the simulated chi values
        ax_err.set_xticks(valid_chis)
        
        if i == 0:
            ax_err.set_ylabel("Error (Log Scale)")
            ax_err.legend(framealpha=1)

        # Bottom Row: Runtimes
        ax_run = axes[1, i]
        ax_run.plot(valid_chis, runtimes, marker='o', color='teal', linewidth=4, markersize=14)
        ax_run.set_xlabel("Bond Dimension (chi)")
        
        # Force X-axis ticks to match exactly the simulated chi values
        ax_run.set_xticks(valid_chis)
        
        # Force exactly 5 Y-axis ticks dynamically spanning the data range
        ax_run.yaxis.set_major_locator(LinearLocator(5))
        
        if i == 0:
            ax_run.set_ylabel("Runtime (seconds)")

plt.tight_layout()
plt.show()