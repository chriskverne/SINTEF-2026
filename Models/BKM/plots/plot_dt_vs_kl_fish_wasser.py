import os
import json

# The dt values you generated data for
dt_values = [1, 0.5, 0.25, 0.125, 0.0625, 0.03125]
target_time = 1

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
        print(f"% Warning: Data for dt={dt} not found at {filename}")

# Helper function to format coordinates for PGFPlots
def get_coords(x_data, y_data):
    return " ".join([f"({x},{y})" for x, y in zip(x_data, y_data)])

print("% ==========================================")
print("% COPY AND PASTE THIS INTO YOUR LATEX FILE")
print("% ==========================================\n")

# Define the custom gold color
print(r"\definecolor{gold}{RGB}{218, 165, 32}")
print()

# Print the \addplot commands
print("% 1. KL Divergence")
print(rf"\addplot[color=red, mark=*, thick] coordinates {{ {get_coords(valid_dts, kl_divs)} }};")
print()

print("% 2. Wasserstein Distance")
print(rf"\addplot[color=green, mark=square*, thick] coordinates {{ {get_coords(valid_dts, wass_dists)} }};")
print()

print("% 3. Fisher-Rao Distance")
print(rf"\addplot[color=gold, mark=triangle*, thick] coordinates {{ {get_coords(valid_dts, fisher_raos)} }};")
print()

print("% 4. Variance Error")
print(rf"\addplot[color=purple, mark=diamond*, thick] coordinates {{ {get_coords(valid_dts, var_errors)} }};")