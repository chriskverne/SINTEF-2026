import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

# Set up the figure with 2 subplots (Tree on the left, PDF on the right)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={'width_ratios': [2, 1]})

# ==========================================
# 1. Plotting the Trinomial Tree (Left Plot)
# ==========================================
steps = 2

# Draw edges to the next step (up, mid, down)
for t in range(steps):
    for y in range(-t, t + 1):
        ax1.plot([t, t + 1], [y, y + 1], color='steelblue', alpha=0.6, zorder=1)
        ax1.plot([t, t + 1], [y, y], color='steelblue', alpha=0.6, zorder=1)
        ax1.plot([t, t + 1], [y, y - 1], color='steelblue', alpha=0.6, zorder=1)

# Scatter plot for the actual nodes
for t in range(steps + 1):
    for y in range(-t, t + 1):
        ax1.scatter(t, y, color='darkorange', s=150, zorder=2, edgecolor='black')

# Add a curly bracket '{' between node j=1 and j=0 at t=1 using a large text character
ax1.text(0.96, 0.5, '}', fontsize=44, color='purple', va='center', ha='left', family='serif')
# Label the difference as \Delta \mathbf{X}
ax1.text(0.9, 0.5, r'$\Delta \mathbf{X}$', color='purple', fontsize=14, va='center', ha='right', fontweight='bold')

# Add a curly bracket '{' between node j=0 and j=-1 at t=1
ax1.text(0.96, -0.5, '}', fontsize=44, color='purple', va='center', ha='left', family='serif')
# Label the difference as -\Delta \mathbf{X}
ax1.text(0.9, -0.5, r'$-\Delta \mathbf{X}$', color='purple', fontsize=14, va='center', ha='right', fontweight='bold')

# ==========================================
# Adding Probability Labels
# ==========================================

# 1. From node j = 0 at t=0 (the first 3 edges)
x_pos_0 = 0.5
ax1.text(x_pos_0, 0.5 + 0.03, r'$p_{up}^{(0,0)}$', color='darkred', fontsize=11, ha='center', va='bottom')
ax1.text(x_pos_0, 0.0 + 0.03, r'$p_{mid}^{(0,0)}$', color='darkred', fontsize=11, ha='center', va='bottom')
ax1.text(x_pos_0, -0.5 + 0.03, r'$p_{down}^{(0,0)}$', color='darkred', fontsize=11, ha='center', va='bottom')

# 2. From all nodes at t=1 (represented as \Delta t)
x_pos_1 = 1.25

# From node j = 1 (top node at t=1)
ax1.text(x_pos_1, 1.25 + 0.03, r'$p_{up}^{(1,\Delta t)}$', color='darkred', fontsize=11, ha='center', va='bottom')
ax1.text(x_pos_1, 1.0 + 0.03, r'$p_{mid}^{(1,\Delta t)}$', color='darkred', fontsize=11, ha='center', va='bottom')
ax1.text(x_pos_1, 0.75 + 0.03, r'$p_{down}^{(1,\Delta t)}$', color='darkred', fontsize=11, ha='center', va='bottom')

# From node j = 0 (middle node at t=1)
ax1.text(x_pos_1, 0.25 + 0.03, r'$p_{up}^{(0,\Delta t)}$', color='darkred', fontsize=11, ha='center', va='bottom')
ax1.text(x_pos_1, 0.0 + 0.03, r'$p_{mid}^{(0,\Delta t)}$', color='darkred', fontsize=11, ha='center', va='bottom')
ax1.text(x_pos_1, -0.25 + 0.03, r'$p_{down}^{(0,\Delta t)}$', color='darkred', fontsize=11, ha='center', va='bottom')

# From node j = -1 (bottom node at t=1)
ax1.text(x_pos_1, -0.75 + 0.03, r'$p_{up}^{(-1,\Delta t)}$', color='darkred', fontsize=11, ha='center', va='bottom')
ax1.text(x_pos_1, -1.0 + 0.03, r'$p_{mid}^{(-1,\Delta t)}$', color='darkred', fontsize=11, ha='center', va='bottom')
ax1.text(x_pos_1, -1.25 + 0.03, r'$p_{down}^{(-1,\Delta t)}$', color='darkred', fontsize=11, ha='center', va='bottom')

# Formatting the tree plot
ax1.set_xlabel(r'Time Step ($\Delta t$)', fontsize=12)
ax1.set_ylabel(r'State Space ($\mathbf{X}$)', fontsize=12)
ax1.set_xticks(range(steps + 1))
ax1.set_yticks(range(-steps, steps + 1))
ax1.grid(True, linestyle='--', alpha=0.3)

# ==========================================
# 2. Plotting the Resulting PDF (Right Plot)
# ==========================================
# Simulating the resulting distribution across the final states (y = -2, -1, 0, 1, 2)
y_vals = np.linspace(-2.5, 2.5, 100)
pdf_vals = stats.norm.pdf(y_vals, loc=0, scale=1) # Mean 0, StdDev 1 for visual purposes

# Plotting the PDF sideways to match the tree's y-axis
ax2.plot(pdf_vals, y_vals, color='darkred', linewidth=2)
ax2.fill_betweenx(y_vals, pdf_vals, 0, color='darkred', alpha=0.3)

# Add horizontal dashed lines aligning the final tree nodes with the PDF
for y in range(-steps, steps + 1):
    ax2.axhline(y=y, color='black', linestyle=':', alpha=0.2)

# Add the P(X) text to the top right corner
ax2.text(0.92, 0.96, r'$P(\mathbf{X}_t)$', transform=ax2.transAxes, fontsize=16, fontweight='bold', va='top', ha='right', color='black')

# Formatting the PDF plot
ax2.set_xlabel('Probability Density', fontsize=12)
ax2.set_ylim(-steps - 0.5, steps + 0.5)
ax2.set_yticks(range(-steps, steps + 1))
ax2.grid(True, linestyle='--', alpha=0.3)

plt.tight_layout()
# plt.show()
plt.savefig('./trinomial_tree.png')