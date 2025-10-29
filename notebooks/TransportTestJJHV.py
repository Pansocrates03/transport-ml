import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt

# --- PARAMETERS ---
N_VARS = 3  # Number of logistic/binary variables (X1, X2, X3)
N_ITERATIONS = 1000  # Total training steps
EPSILON_START = 1.0  # Epsilon for epsilon-greedy policy
EPSILON_END = 0.05
EPSILON_DECAY = (EPSILON_START - EPSILON_END) / N_ITERATIONS
LAPLACE_SMOOTHING = 1  # For CPD estimation (alpha=1 for add-one smoothing)

# The state space for binary variables: all variables are {0, 1}
# Actions are interventions (do-operator): do(X_i = 1) or no-op (N+1 action)
ACTIONS = [f'do(X{i+1}=1)' for i in range(N_VARS)] + ['no_op'] 
N_ACTIONS = len(ACTIONS)

# --- AGENT STATE ---
# P_beliefs: N x N matrix of edge probabilities P(Xi -> Xj)
# Initialized uniformly (maximum uncertainty)
P_beliefs = np.full((N_VARS, N_VARS), 0.5)
np.fill_diagonal(P_beliefs, 0.0) # No self-loops

# O_data: Observation Buffer (stores (action, outcome) tuples)
# outcome is a tuple (x1, x2, ..., xN)
O_data = [] 

# --- RL State ---
current_epsilon = EPSILON_START
total_reward = 0

# --- GROUND TRUTH ENVIRONMENT (Unknown to Agent) ---

# Ground Truth DAG Adjacency Matrix
# Example: X1 -> X2, X2 -> X3
TRUE_G = np.array([
    [0, 1, 0],
    [0, 0, 1],
    [0, 0, 0]
])

# Ground Truth Conditional Probability Distributions (CPDs)
# For binary variables, CPD[i][parent_state] is P(Xi=1 | Pa(Xi) = parent_state)
# The parent_state is a binary encoding of the parents' values.
# Example: for X3 with parent X2, parent_state=0 means X2=0, state=1 means X2=1.

# P(X1=1) (No Parents - index 0 is P(X1=1 | NoParents))
CPD_X1 = {0: 0.3} 

# P(X2=1 | X1) (Parent X1 - indices 0, 1 are P(X2=1 | X1=0), P(X2=1 | X1=1))
CPD_X2 = {0: 0.1, 1: 0.9} 

# P(X3=1 | X2) (Parent X2 - indices 0, 1 are P(X3=1 | X2=0), P(X3=1 | X2=1))
CPD_X3 = {0: 0.2, 1: 0.8}

TRUE_CPDS = [CPD_X1, CPD_X2, CPD_X3]

# Utility Function (Reward for the Agent)
# The goal is to maximize P(X3=1 AND X1=0).
def calculate_utility(state):
    """Calculates the reward for an observed state."""
    # State is a tuple (x1, x2, x3)
    x1, x2, x3 = state
    if x3 == 1 and x1 == 0:
        return 10.0
    return 0.0

# Function to simulate the causal process
def simulate_environment(action_index):
    """Generates an outcome state given an intervention."""
    state = np.zeros(N_VARS, dtype=int)
    intervention = None

    if action_index < N_VARS:
        # Intervention do(X_i = 1)
        intervened_var = action_index
        state[intervened_var] = 1 # Fix the intervened variable
        intervention = intervened_var
    
    # Causal sampling respecting the topological order of TRUE_G
    for i in range(N_VARS):
        if intervention == i:
            continue # Skip the intervened variable
        
        parents = np.where(TRUE_G[:, i] == 1)[0]
        
        if len(parents) == 0:
            # P(Xi=1)
            prob_one = TRUE_CPDS[i][0]
        else:
            # Determine parent state (binary encoding of parent values)
            parent_values = state[parents]
            # Convert binary parent values to integer index (e.g., (1, 0) -> 2)
            parent_state_index = int(np.sum(parent_values * (2**np.arange(len(parents))[::-1])))
            prob_one = TRUE_CPDS[i][parent_state_index]
            
        # Sample the variable
        if random.random() < prob_one:
            state[i] = 1
        else:
            state[i] = 0
            
    return tuple(state)


# --- Causal Inference Tools ---

def sample_dag(P_beliefs):
    """Samples a concrete DAG G based on the belief matrix P."""
    G = (np.random.rand(N_VARS, N_VARS) < P_beliefs).astype(int)
    np.fill_diagonal(G, 0.0)
    return G

def estimate_cpd_param(O_data, child_idx, parents_indices, child_value):
    """
    Estimates P(Child=child_value | Parents=parent_state) from the observation buffer O_data.
    Uses Maximum Likelihood Estimation with Laplace smoothing.
    """
    
    # Total possible parent states is 2^|Parents|
    n_parent_states = 2**len(parents_indices)
    
    # Store counts: counts[parent_state][child_value]
    counts = np.full((n_parent_states, 2), LAPLACE_SMOOTHING) # Start with smoothing counts
    
    for observation in O_data:
        # observation is {'action': action_idx, 'outcome': (x1, x2, x3)}
        outcome = observation['outcome']
        
        # 1. Determine the parent state
        if not parents_indices.size:
            parent_state_index = 0
        else:
            parent_values = np.array([outcome[p] for p in parents_indices])
            parent_state_index = int(np.sum(parent_values * (2**np.arange(len(parents_indices))[::-1])))
        
        # 2. Increment the count
        current_child_value = outcome[child_idx]
        counts[parent_state_index, current_child_value] += 1
    
    # Return the probability P(Child=1 | Parents) for all parent states
    # P(Child=1 | Pa) = count(Child=1, Pa) / count(Pa)
    prob_one_given_pa = counts[:, 1] / counts.sum(axis=1)
    
    # This function is usually simplified to return the whole distribution
    return prob_one_given_pa

def calculate_likelihood(G, action_idx, state, O_data):
    """
    Calculates the likelihood P(state | do(action), G) for the observed state.
    
    This requires estimating the CPDs for G from the observation data O_data.
    """
    likelihood = 1.0
    
    # Determine intervention variable
    intervention_idx = action_idx if action_idx < N_VARS else None
    
    for i in range(N_VARS):
        
        if i == intervention_idx:
            # If variable is intervened, P(Xi | do(a)) is 1 if Xi matches the intervention value, 0 otherwise
            # Note: Our actions only intervene to set X_i=1
            if state[i] == 1:
                likelihood *= 1.0
            else:
                return 0.0 # Intervention sets it to 1, but we observed 0. Likelihood is zero.
            continue
        
        # Find parents in the sampled graph G
        parents_indices = np.where(G[:, i] == 1)[0]
        
        # Estimate the CPDs for this node based on all data observed so far
        # Simplified: We estimate P(Xi=1 | Pa(Xi)) for all Pa states
        prob_one_given_pa = estimate_cpd_param(O_data, i, parents_indices, 1)

        # Get the actual parent state for the observed state
        if not parents_indices.size:
            parent_state_index = 0
        else:
            parent_values = np.array([state[p] for p in parents_indices])
            parent_state_index = int(np.sum(parent_values * (2**np.arange(len(parents_indices))[::-1])))
            
        # Get P(Xi | Pa) from the estimated CPD
        p_xi_given_pa = prob_one_given_pa[parent_state_index]
        
        # Multiply likelihood
        if state[i] == 1:
            likelihood *= p_xi_given_pa
        else:
            likelihood *= (1.0 - p_xi_given_pa)
            
    return likelihood

def calculate_expected_reward(G, action_idx, O_data):
    """
    Calculates E[Utility | do(action), G] by summing over all 2^N possible outcome states.
    This is computationally expensive and is simplified here by sampling outcomes.
    """
    # For simplicity and speed in this example, we use the estimated CPDs 
    # and sample a small number of outcomes N_SAMPLES for Monte Carlo estimation.
    N_SAMPLES = 100
    total_reward = 0
    
    # We must use the AGENT's estimated CPDs (from O_data) to sample
    # The structure G is provided
    
    # 1. Estimate all CPDs for graph G
    estimated_cpds = {}
    for i in range(N_VARS):
        parents_indices = np.where(G[:, i] == 1)[0]
        estimated_cpds[i] = estimate_cpd_param(O_data, i, parents_indices, 1) # P(Xi=1 | Pa)
        
    # 2. Monte Carlo sampling of outcomes
    for _ in range(N_SAMPLES):
        simulated_state = np.zeros(N_VARS, dtype=int)
        intervention_idx = action_idx if action_idx < N_VARS else None
        
        if intervention_idx is not None:
            simulated_state[intervention_idx] = 1 # do(X_i=1)
            
        for i in range(N_VARS):
            if i == intervention_idx: continue
            
            parents_indices = np.where(G[:, i] == 1)[0]
            
            if not parents_indices.size:
                prob_one = estimated_cpds[i][0]
            else:
                parent_values = simulated_state[parents_indices] # Use the state sampled so far
                parent_state_index = int(np.sum(parent_values * (2**np.arange(len(parents_indices))[::-1])))
                prob_one = estimated_cpds[i][parent_state_index]
                
            if random.random() < prob_one:
                simulated_state[i] = 1
            else:
                simulated_state[i] = 0

        total_reward += calculate_utility(tuple(simulated_state))
        
    return total_reward / N_SAMPLES


# --- THE CAUSAL RL ALGORITHM LOOP ---

for t in range(N_ITERATIONS):
    
    # 1. Sample a DAG G from the current belief P
    G_sampled = sample_dag(P_beliefs)

    # 2. Select Action (Action Selection / Policy) - Epsilon-Greedy
    
    a_star_idx = -1
    
    if random.random() < current_epsilon or not O_data:
        # Exploration: Choose a random action (intervention)
        a_star_idx = random.randint(0, N_ACTIONS - 1)
    else:
        # Exploitation: Choose action that maximizes expected reward given G_sampled
        max_expected_reward = -np.inf
        
        for a_idx in range(N_ACTIONS):
            # Calculate Expected Reward: E[Utility | do(a), G_sampled]
            expected_r = calculate_expected_reward(G_sampled, a_idx, O_data)
            
            if expected_r > max_expected_reward:
                max_expected_reward = expected_r
                a_star_idx = a_idx
                
    # 3. Intervene on the Environment
    # The true environment is simulated (assuming the agent has access to full state x)
    observed_state = simulate_environment(a_star_idx)
    
    # 4. Observe Reward
    reward = calculate_utility(observed_state)
    total_reward += reward
    
    # 5. Update Observation Buffer
    O_data.append({'action': a_star_idx, 'outcome': observed_state})
    
    # 6. Bayesian Structure Update (Bayes' Theorem for P_ij)
    # Update all N*(N-1) edge beliefs P_ij based on the observation
    
    for i in range(N_VARS):
        for j in range(N_VARS):
            if i == j: continue
            
            # Hypothesis 1: Edge (i -> j) exists
            G_H1 = G_sampled.copy()
            G_H1[i, j] = 1 # Ensure edge i->j is present
            
            # Hypothesis 0: Edge (i -> j) does NOT exist
            G_H0 = G_sampled.copy()
            G_H0[i, j] = 0 # Ensure edge i->j is absent
            
            # Likelihood P(x | do(a*), G_H1)
            L_H1 = calculate_likelihood(G_H1, a_star_idx, observed_state, O_data)
            
            # Likelihood P(x | do(a*), G_H0)
            L_H0 = calculate_likelihood(G_H0, a_star_idx, observed_state, O_data)
            
            # Prior P(H1) = P_beliefs[i, j]
            P_prior_H1 = P_beliefs[i, j]
            P_prior_H0 = 1.0 - P_beliefs[i, j]
            
            # Apply Bayes' Rule: P(H1 | x) = (L(H1) * P(H1)) / (L(H1) * P(H1) + L(H0) * P(H0))
            numerator = L_H1 * P_prior_H1
            denominator = numerator + (L_H0 * P_prior_H0)
            
            if denominator == 0:
                # Handle division by zero (e.g., if both likelihoods are zero, keep prior)
                P_beliefs[i, j] = P_prior_H1 
            else:
                P_beliefs[i, j] = numerator / denominator
                
    # 7. Update Epsilon (Exploration-Exploitation balance)
    current_epsilon = max(EPSILON_END, current_epsilon - EPSILON_DECAY)

    # 8. Print Status (Optional)
    if t % 100 == 0:
        print(f"Iteration {t}: Total Reward = {total_reward}, Epsilon = {current_epsilon:.2f}")

# --- RESULTS ---
print("\n--- Final Results ---")
print("True Causal Structure:\n", TRUE_G)
print("\nLearned Causal Structure Beliefs P(Xi -> Xj):\n", np.round(P_beliefs, 3))
print(f"\nTotal Reward Accumulated: {total_reward}")


# --- 5. VISUALIZATION OF CAUSAL STRENGTH ---

# The final P_beliefs matrix from the Causal RL Algorithm loop is used.
# P_beliefs: Learned Causal Structure Beliefs P(Xi -> Xj)

def visualize_causal_dag(P_beliefs):
    """
    Visualizes the learned causal structure with edge weights 
    representing the belief probability (causal strength).
    """
    
    N_VARS = P_beliefs.shape[0]
    VARIABLES = [f'X{i+1}' for i in range(N_VARS)]
    
    # Initialize a Directed Graph
    G = nx.DiGraph()
    G.add_nodes_from(VARIABLES)
    
    # Add weighted edges to the graph
    max_strength = 0.0
    for i in range(N_VARS):
        for j in range(N_VARS):
            strength = P_beliefs[i, j]
            if strength > 0.05: # Only draw edges with significant belief
                G.add_edge(VARIABLES[i], VARIABLES[j], weight=strength)
                if strength > max_strength:
                    max_strength = strength

    # --- Drawing Setup ---
    pos = nx.circular_layout(G) # Use a simple layout
    
    # Extract edge data for visualization (weights and colors)
    edges = G.edges()
    weights = [G[u][v]['weight'] for u, v in edges]
    
    # Scale edge weights for thickness (e.g., max thickness 5)
    edge_widths = [w * 5 / max_strength for w in weights] 
    
    # Color edges based on strength (e.g., darker for higher strength)
    # Using the 'Reds' colormap, but inverted (vmin, vmax) to put high strength at the top end
    edge_colors = weights
    
    # Draw the graph
    plt.figure(figsize=(8, 8))
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=2000, node_color='lightblue', edgecolors='black')
    
    # Draw edges with weight and color scaling
    # 'edge_cmap=plt.cm.Reds' defines the colormap
    # 'edge_vmax' and 'edge_vmin' define the range of the colormap
    nx.draw_networkx_edges(G, pos, edgelist=edges, 
                           width=edge_widths, 
                           edge_color=edge_colors, 
                           edge_cmap=plt.cm.Reds, 
                           edge_vmax=1.0, edge_vmin=0.0,
                           arrowsize=25, 
                           connectionstyle='arc3,rad=0.1')

    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')

    # Add edge labels (the actual belief probability)
    edge_labels = { (u, v): f"{w:.2f}" for u, v, w in G.edges(data='weight') }
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', label_pos=0.3)
    
    plt.title("Learned Causal Structure (Edge Strength P(Xi -> Xj))", fontsize=16)
    plt.axis('off')
    plt.show()

# Call the visualization function with the learned belief matrix
visualize_causal_dag(P_beliefs)