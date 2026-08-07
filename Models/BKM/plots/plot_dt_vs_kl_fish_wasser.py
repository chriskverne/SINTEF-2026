import os
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# The dt values you generated data for
dt_values = [1, 0.5, 0.25, 0.125, 0.0625, 0.03125]
target_time = 1

# Lists to hold the extracted metrics
valid_dts = []
kl_divs = []
wass_dists = []
fisher_raos = []
var_errors = []

# Extract data from JSON files
for dt in dt_values:
    filename = f'../figures/metrics_dt{dt}_time{target_time}.json'
    
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            metrics = json.load(f)
            
        valid_dts.append(dt)
        kl_divs.append(metrics['kl_divergence'])
        wass_dists.append(metrics['wasserstein_distance'])
        fisher_raos.append(metrics['fisher_rao_distance'])
        var_errors.append(metrics['var_error'])
    else:
        print(f"Warning: Data for dt={dt} not found at {filename}")

# Create a 1x4 grid. 
# Using figsize=(16, 3.5) makes it a compact, wide banner.
fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(16, 2.75))

# Force the figure background to be white
fig.patch.set_facecolor('white')
axes = axes.flatten()

# 1. KL Divergence Plot
axes[0].plot(valid_dts, kl_divs, marker='o', color='crimson', linewidth=4, markersize=12)
axes[0].set_ylabel('KL Divergence', fontsize=14)

# 2. Wasserstein Distance Plot
axes[1].plot(valid_dts, wass_dists, marker='s', color='forestgreen', linewidth=4, markersize=12)
axes[1].set_ylabel('Wasserstein Distance', fontsize=14)

# 3. Fisher-Rao Distance Plot
axes[2].plot(valid_dts, fisher_raos, marker='^', color='darkorange', linewidth=4, markersize=12)
axes[2].set_ylabel('Fisher-Rao Distance', fontsize=14)

# 4. Variance Error Plot
axes[3].plot(valid_dts, var_errors, marker='D', color='purple', linewidth=4, markersize=12)
axes[3].set_ylabel('Variance Error', fontsize=14)

# Apply common formatting to all subplots
for ax in axes:
    ax.set_xlabel(r'Time Step Size ($\Delta t$)', fontsize=14)
    
    # Use a log base 2 scale for x-axis since dt halves each time
    ax.set_xscale('log', base=2)
    
    # Invert x-axis so it reads left-to-right from largest dt (1) to smallest dt (0.03125)
    ax.invert_xaxis()
    
    # Remove the grid completely
    ax.grid(False)
    
    # Force maximum of 4 ticks on the y-axis
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    
    # Make x and y tick labels much larger
    ax.tick_params(axis='both', which='major', labelsize=13)
    
    # Customize x-ticks to display the actual dt values as fractions
    ax.set_xticks(valid_dts)
    ax.set_xticklabels(['1' if dt == 1 else f'1/{int(1/dt)}' for dt in valid_dts])

plt.tight_layout()

# Save the figure to the current directory (./)
plt.savefig('./divergence_metrics_convergence.png', facecolor='white', transparent=False, dpi=300)

plt.show()