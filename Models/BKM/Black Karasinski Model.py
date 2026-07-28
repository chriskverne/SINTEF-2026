import numpy as np
import matplotlib.pyplot as plt
import pennylane as qml
import math

class BlackKarasinskiModel:
    def __init__(self, k, theta, var, dt):
        self.k = k # currently constant
        self.theta = theta # currently constant
        self.var = var
        self.dt = dt
        
    def generate_paths(self, n_paths=10, n_steps=10):
        r_start = self.theta[0]
        ln_r = np.log(r_start)

        prices = np.zeros((n_steps, n_paths))
        for j in range(n_paths):
            ln_r = np.log(self.theta[0])
            for i in range(n_steps):
                prices[i][j] = np.exp(ln_r)
                ln_r = ln_r + self.k * (np.log(self.theta[i]) - ln_r)*self.dt + np.random.normal(0, 1) * self.var * np.sqrt(self.dt)

        return prices
    
    def plot_paths(self, n_paths=10, n_steps=10):
        prices = self.generate_paths(n_paths, n_steps)
        x = np.arange(n_steps)

        plt.figure(figsize=(8, 5))
        for j in range(n_paths):
            plt.plot(x, prices[:, j], alpha=0.7, label=f'Path {j+1}')

        # plt.axhline(self.theta[0], color='black', linestyle='--', linewidth=1, label='θ (long-run mean)')
        plt.xlabel('Time step')
        plt.ylabel('Short rate')
        plt.title('Black-Karasinski Simulated Paths')
        plt.legend(fontsize=8, ncol=2)
        plt.show()

    def construct_trinomial_tree(self, num_steps=3):
        # we will start with assuming theta and kappa are constant
        # here delta x = sigma sqrt(3t)

        # the probability depends on the distnace from the center which I havent coded properly.

        tree = np.zeros((num_steps, 2*num_steps + 1))

        for i in range(num_steps):
            # Num(vertices) = 2t + 1
            for j in range(-i, i+1): #starts at {-1, 0 1} then {-2, -1, 0, 1, 2}, and so forth
                p_up = 1/6 + (self.k**2 * j**2 * self.dt**2 - self.k * j * self.dt) / 2
                p_mid = 2/3 - (self.k**2 * j**2 * self.dt**2)
                p_down = 1/6 + (self.k**2 * j**2 * self.dt**2 + self.k * j * self.dt) / 2
                if p_up < 0 or p_up > 1 or p_mid < 0 or p_mid > 1 or p_down < 0 or p_down > 1:
                    print(f"WARNING: Invalid probabilities at step {i}, node {j}: p_up={p_up}, p_mid={p_mid}, p_down={p_down}")
                val = j * self.var * np.sqrt(3 * self.dt)
                tree[i][j] = val
                
                print(f"  Node j={j:2d}: p_up={p_up:.4f}, p_mid={p_mid:.4f}, p_down={p_down:.4f}")

        return tree
    
    def construct_trinomial_tree_changing_mean(self, num_steps=3):
        tree = np.zeros((num_steps, 2*num_steps + 1))

        # use theta = {t0: r0, t1: r1, t2: r2, ...} to change the mean at each step
        for i in range(num_steps):
            for j in range(-i, i+1):
                p_up = 1/6 + (self.k**2 * j**2 * self.dt**2 - self.k * j * self.dt) / 2
                p_mid = 2/3 - (self.k**2 * j**2 * self.dt**2)
                p_down = 1/6 + (self.k**2 * j**2 * self.dt**2 + self.k * j * self.dt) / 2
                
                log_rate = np.log(self.theta[i]) + j * self.var * np.sqrt(3 * self.dt)
                tree[i][j] = np.exp(log_rate)

        return tree
    
    def plot_trinomial_tree_changing_mean(self, num_steps=10):
        # Fetch the rate grid
        tree = self.construct_trinomial_tree_changing_mean(num_steps=num_steps)

        fig, ax = plt.subplots(figsize=(12, 7))
        branch_colors = {1: 'seagreen', 0: 'gray', -1: 'indianred'}

        for i in range(num_steps):
            for j in range(-i, i + 1):
                current_rate = tree[i][j]

                # Draw edges to the next step's up/mid/down children
                if i < num_steps - 1:
                    for dj in (1, 0, -1):
                        next_rate = tree[i + 1][j + dj]
                        
                        # CRITICAL FIX: Plot (time, rate) instead of (time, node index)
                        ax.plot([i, i + 1], [current_rate, next_rate],
                                color=branch_colors[dj], alpha=0.6, linewidth=1.2, zorder=1)

                # Draw the vertex
                ax.scatter(i, current_rate, color='steelblue', s=80, zorder=3, edgecolor='white')
                
                # Annotate the rates (decluttered for larger trees)
                if num_steps <= 5 or j % 2 == 0: 
                    ax.annotate(f"{current_rate:.3f}", (i, current_rate), textcoords="offset points",
                                xytext=(0, 8), ha='center', fontsize=8, zorder=4)

        # Plot the target mean (theta) as a dashed black line to show the tree following it
        target_means = [self.theta[i] for i in range(num_steps)]
        ax.plot(range(num_steps), target_means, color='black', linestyle='--', linewidth=2, label='Target Mean (θ)', zorder=0)

        ax.set_xlabel('Time step')
        ax.set_ylabel('Interest Rate (Real Value)') # The Y-axis is now the actual rate!
        ax.set_title('Black-Karasinski Trinomial Tree (Shifting Mean)')
        ax.set_xticks(range(num_steps))
        ax.legend()
        
        plt.tight_layout()
        plt.show()

    
    def plot_trinomial_tree(self, num_steps=3):
        tree = self.construct_trinomial_tree(num_steps=num_steps)

        fig, ax = plt.subplots(figsize=(10, 7))
        branch_colors = {1: 'seagreen', 0: 'gray', -1: 'indianred'}

        for i in range(num_steps):
            for j in range(-i, i + 1):
                val = tree[i][j]
                rate = self.theta * np.exp(val)

                # edges to next step's up/mid/down children
                if i < num_steps - 1:
                    for dj in (1, 0, -1):
                        ax.plot([i, i + 1], [j, j + dj],
                                color=branch_colors[dj], alpha=0.6, linewidth=1.2, zorder=1)

                # vertex + price label
                ax.scatter(i, j, color='steelblue', s=220, zorder=3, edgecolor='white')
                ax.annotate(f"{rate:.4f}", (i, j), textcoords="offset points",
                            xytext=(0, 12), ha='center', fontsize=8, zorder=4)

        ax.set_xlabel('Time step')
        ax.set_ylabel('Node index (j)')
        ax.set_title('Black-Karasinski Trinomial Tree')
        ax.set_xticks(range(num_steps))
        plt.tight_layout()
        plt.show()

    def construct_binomial_tree(self, num_steps=3):
        # we will start with assuming theta and kappa are constant
        # here delta x = sigma sqrt(t)

        tree_rates = np.full((num_steps, 2*num_steps + 1), np.nan)
        tree_probs = np.full((num_steps, 2*num_steps + 1), np.nan)

        dx = self.var * np.sqrt(self.dt)
        ln_r0 = np.log(self.theta) # starting value

        for i in range(num_steps):
            for j in range(-i, i + 1,2):
                x_n = ln_r0 + j * dx # price = starting price + position * factor: postion = +2 (upup), 0 (mid), -2 (downdown)
                price = np.exp(x_n)
                # compute probability of certain position to increase / decrease it's value
                p_up = 1/2  + (self.k*(np.log(self.theta) - x_n) * np.sqrt(self.dt)) / (2 * self.var)
                if(p_up < 0 or p_up > 1):
                    print(f"WARNING p = {p_up}")

                # add to matrix
                col_index = j + num_steps
                tree_rates[i, col_index] = price
                tree_probs[i, col_index] = p_up

        return tree_rates, tree_probs
    
    def construct_binomial_tree_changing_mean(self, num_steps=3):
        tree_rates = np.zeros((num_steps, 2*num_steps + 1))
        tree_probs = np.full((num_steps, 2*num_steps + 1), np.nan)


        for i in range(num_steps):
            for j in range(-i, i, 2):
                x_n = np.log(self.theta[i]) * j * (self.var * np.sqrt(self.dt))
                tree_rates[i][j] = np.exp(x_n)
                tree_probs[i][j] = 1/2 - (j * self.k * self.dt)/2 # probability only depends on position j and kappa which is rather clean

            return tree_rates, tree_probs
        
    def plot_binomial_tree(self, num_steps=3):
            # Fetch the rates and probabilities from your new method
            tree_rates, tree_probs = self.construct_binomial_tree(num_steps=num_steps)

            fig, ax = plt.subplots(figsize=(10, 7))
            
            for i in range(num_steps):
                # Binomial nodes expand by 1 step up or down, meaning valid nodes step by 2
                for j in range(-i, i + 1, 2):
                    col_index = j + num_steps
                    rate = tree_rates[i, col_index]
                    
                    # Skip unpopulated nodes
                    if np.isnan(rate):
                        continue

                    # Draw edges to the next step's up/down children
                    if i < num_steps - 1:
                        # Up branch (j + 1)
                        ax.plot([i, i + 1], [j, j + 1], 
                                color='seagreen', alpha=0.6, linewidth=1.5, zorder=1)
                        # Down branch (j - 1)
                        ax.plot([i, i + 1], [j, j - 1], 
                                color='indianred', alpha=0.6, linewidth=1.5, zorder=1)

                    # Draw the vertex
                    ax.scatter(i, j, color='steelblue', s=220, zorder=3, edgecolor='white')
                    
                    # Annotate the node with the physical rate
                    ax.annotate(f"{rate:.4f}", (i, j), textcoords="offset points",
                                xytext=(0, 12), ha='center', fontsize=8, zorder=4)

            ax.set_xlabel('Time step')
            ax.set_ylabel('Node index (j)')
            ax.set_title('Black-Karasinski Binomial Tree')
            ax.set_xticks(range(num_steps))
            
            # Adjust y-axis to comfortably fit the highest and lowest possible nodes
            ax.set_ylim(-num_steps, num_steps)
            
            plt.tight_layout()
            plt.show()

    def compute_step_angles(self, num_steps):
            angles = {}
            
            # The maximum span of j after num_steps (where i goes up to num_steps - 1)
            max_j = num_steps - 1
            
            for j in range(-max_j, max_j + 1):
                a = -self.k * j * self.dt
                
                # Calculate raw trinomial probabilities
                p_up = 1/6 + (a**2 + a) / 2
                p_mid = 2/3 - a**2
                p_down = 1/6 + (a**2 - a) / 2

                # Prevent negative probs:
                eps = 1e-8
                p_up = max(p_up, eps)
                p_mid = max(p_mid, eps)
                p_down = max(p_down, eps)
                total_p = p_up + p_mid + p_down
                p_up /= total_p
                p_mid /= total_p
                p_down /= total_p

                # |00> = down, |01> = mid, |10> = up
                # 1. Up or not up?
                theta_1 = 2 * np.arcsin(np.sqrt(p_up))
                # 2. Mid or down?
                theta_2 = 2 * np.arcsin(np.sqrt(p_mid / (p_mid + p_down)))

                # key is position 'j'
                angles[j] = (theta_1, theta_2)

            return angles
   
    def quantum_trinomial_state(self, T=3):
        angles = self.compute_step_angles(T)

        num_state_qubits = 2 * T
        num_pos_qubits = math.ceil(math.log2(2 * T + 1))

        state_wires = [f"s{i}" for i in range(num_state_qubits)]
        pos_wires = [f"p{i}" for i in range(num_pos_qubits)]
        all_wires = state_wires + pos_wires
        dev = qml.device("default.qubit", wires=all_wires)


        # increases / decreases the position by 1 whether we are up / down / mid.
        dim = 2 ** num_pos_qubits
        U_inc = np.roll(np.eye(dim), 1, axis=0)
        U_dec = np.roll(np.eye(dim), -1, axis=0)
        
        @qml.qnode(dev)
        def circuit():
            # prepare position register
            binary_offset = format(T, f'0{num_pos_qubits}b')
            for idx, bit in enumerate(binary_offset):
                if bit == '1':
                    qml.PauliX(wires=pos_wires[idx])

            ############## STEP 1: INITIAL U rot ##########################
            # get angle for j = 0
            theta_1, theta_2 = angles[0]
            # apply U(0)
            qml.RY(theta_1, wires=state_wires[0])
            qml.ctrl(qml.RY(theta_2, wires=state_wires[1]), control=state_wires[0], control_values=[0])


            # Run the loop T times to ensure the final position is always updated
            for step in range(1, T + 1):
                
                ############## STEP 2: Update position ######################
                # Grab the qubits from the jump that JUST happened
                prev_s0 = state_wires[2 * (step - 1)] 
                prev_s1 = state_wires[2 * (step - 1) + 1] 
                                
                # IF |00> (Down): Decrement the position
                qml.ctrl(qml.QubitUnitary, control=[prev_s0, prev_s1], control_values=[0, 0])(
                    U_dec, wires=pos_wires)
                # IF |01> (Mid): Do nothing (Identity)
                # IF |10> (Up): Increment the position
                qml.ctrl(qml.QubitUnitary, control=[prev_s0, prev_s1], control_values=[1, 0])(
                    U_inc, wires=pos_wires)

                ############## STEP 3: Prepare Next U(j) ####################
                # Only prepare the next jump if we are NOT on the final step
                if step < T:
                    curr_s0 = state_wires[2 * step] 
                    curr_s1 = state_wires[2 * step + 1] 
                    
                    for j in range(-step, step + 1):
                        theta_1, theta_2 = angles[j] 

                        pos_val = j + T 
                        pos_bin = format(pos_val, f'0{num_pos_qubits}b')
                        ctrl_vals = [int(b) for b in pos_bin]

                        # Apply RY_1(j) if pos = j
                        qml.ctrl(qml.RY, control=pos_wires, control_values=ctrl_vals)(
                            theta_1, wires=curr_s0)

                        # Apply RY_2(j) if pos = j and curr_s0 = 0
                        combined_ctrl_wires = pos_wires + [curr_s0]
                        combined_ctrl_vals = ctrl_vals + [0]
                        qml.ctrl(qml.RY, control=combined_ctrl_wires, control_values=combined_ctrl_vals)(
                            theta_2, wires=curr_s1)

            return qml.probs(wires=state_wires), qml.probs(wires=pos_wires)

        return circuit()

    def plot_path_states(self, T):
        # Unpack the first item in the tuple (state probabilities)
        state_probs, _ = self.quantum_trinomial_state(T)
        
        num_state_qubits = 2 * T
        valid_labels = []
        valid_probs = []
        
        for i, prob in enumerate(state_probs):
            if prob > 1e-5:
                binary_state = format(i, f'0{num_state_qubits}b')
                readable_state = " ".join([binary_state[k:k+2] for k in range(0, len(binary_state), 2)])
                valid_labels.append(readable_state)
                valid_probs.append(prob)
                
        plt.figure(figsize=(12, 5))
        plt.bar(valid_labels, valid_probs, color='seagreen', edgecolor='black')
        plt.xlabel('Path History')
        plt.ylabel('Probability')
        plt.title(f'Probability of Each Unique Path (T={T})')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        # plt.savefig('./figures/gpurun.png')
        plt.show()


    def plot_position_states(self, T):
        # Unpack the second item in the tuple (position probabilities)
        _, pos_probs = self.quantum_trinomial_state(T)
        
        num_pos_qubits = math.ceil(math.log2(2 * T + 1))
        
        j_values = []
        j_probs = []
        
        # Loop through all possible states in the position register
        for i, prob in enumerate(pos_probs):
            # i is the physical integer (pos_val). 
            # We subtract T to get the logical position (j)
            j = i - T 
            
            # We only care about valid j positions between -T and T
            if -T <= j <= T:
                j_values.append(j)
                j_probs.append(prob)
                
        plt.figure(figsize=(10, 5))
        plt.bar(j_values, j_probs, color='royalblue', edgecolor='black')
        plt.xlabel('Final Position (j)')
        plt.ylabel('Probability')
        plt.title(f'Aggregated Position Distribution (T={T})')
        plt.xticks(range(-T, T + 1)) # Force x-axis to show all j integer ticks
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add labels on top of the bars for clarity
        for i, prob in enumerate(j_probs):
            if prob > 0.001:
                plt.text(j_values[i], prob + 0.01, f'{prob:.3f}', ha='center', va='bottom', fontsize=9)
                
        plt.tight_layout()
        plt.savefig('./figures/gpurun.png')

        plt.show()


    


# theta = {0: 0.05, 1: 0.05, 2: 0.05, 3: 0.05, 4: 0.05, 
#          5: 0.15, 6: 0.15, 7: 0.15, 8: 0.15, 9: 0.15}
# theta = {0: 0.05, 1: 0.05, 2: 0.05, 3: 0.1, 4: 0.1}
# bkm = BlackKarasinskiModel(k=0.1, theta=theta, var=0.2, dt=1)
# bkm.plot_trinomial_tree_changing_mean(num_steps=5)
# bkm.plot_paths(n_paths=len(theta), n_steps=len(theta))


theta = 0.5 # consant for now
bkm = BlackKarasinskiModel(k=0.0, theta=theta, var=0.1, dt=1)
bkm.plot_position_states(T=8)
# bkm.plot_path_states(T=8)
# print(bkm.quantum_trinomial_state(T=6))
# bkm.plot_trinomial_tree(num_steps=10)
# bkm.plot_binomial_tree(num_steps=10)
