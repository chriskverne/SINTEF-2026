import os
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter

# The dt values you generated data for
dt_values = [1, 0.5, 0.25, 0.125, 0.0625, 0.03125]
target_time = 1

# Create a 2x3 grid of subplots - Reduced height from 7 to 6
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 6))

# Force the figure background to be white (prevents transparent/black background on save)
fig.patch.set_facecolor('white')

axes = axes.flatten()  # Flatten the 2D array of axes for easy iteration

for i, dt in enumerate(dt_values):
    # Construct the filename matching our previous format
    filename = f'../figures/metrics_dt{dt}_time{target_time}.json'
    
    # Calculate tree size (n steps)
    n_steps = int(round(target_time / dt))
    
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            metrics = json.load(f)
            
        x_values = metrics['x_values']
        quantum_probs = metrics['quantum_probs']
        x_dense = metrics['x_dense']
        pdf_dense = metrics['pdf_dense']
        
        # Calculate dx for the bar width based on the distance between the first two x_values
        dx = x_values[1] - x_values[0] if len(x_values) > 1 else 0.1
        
        # Plot the Quantum Distribution (Bar Chart)
        axes[i].bar(x_values, quantum_probs, width=dx * 0.8, color='royalblue', 
                    edgecolor='black', alpha=0.7, label='Quantum')
        
        # Plot the True Continuous Distribution (Line Chart)
        axes[i].plot(x_dense, pdf_dense, color='red', linestyle='dashed', 
                     linewidth=2, label='True Continuous')
        
        # Formatting: Updated title to include tree size (n)
        axes[i].set_title(rf'$\Delta t = {dt}$ ($n = {n_steps}$)', fontsize=13)
        axes[i].set_xlabel('State Variable (x = ln(r))', fontsize=12)
        
        # Only add the Y-axis label to the leftmost plots
        if i == 0:
            axes[i].legend(loc='upper right', framealpha=1)

        if i % 3 == 0:
            axes[i].set_ylabel('Probability', fontsize=12)
            
            # axes[i].grid(axis='y', linestyle='--', alpha=0.7)
        
        # --- NEW: Tick Adjustments ---
        # Make x and y tick labels larger
        axes[i].tick_params(axis='both', which='major', labelsize=11)
        
        # Force exactly (or up to) 5 ticks on the y-axis
        axes[i].yaxis.set_major_locator(MaxNLocator(nbins=5, prune='lower'))
        
        # Force x-ticks to consistently show 2 decimal places (e.g., 6.00, 6.50)
        axes[i].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        
        # Force y-ticks to consistently show 2 decimal places (e.g., 0.20, 0.40)
        axes[i].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        
    else:
        # Graceful fallback if a file is missing
        axes[i].text(0.5, 0.5, f"File not found:\n{filename}", 
                     ha='center', va='center', fontsize=10, color='red')
        axes[i].set_title(rf'$\Delta t = {dt}$ ($n = {n_steps}$)')
        axes[i].axis('off')

# Adjust layout so titles and labels don't overlap
plt.tight_layout()

# Use facecolor='white' and transparent=False if saving programmatically
plt.savefig('dt_vs_accuracy.png', facecolor='white', transparent=False, dpi=300)

plt.show()