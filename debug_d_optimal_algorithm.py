"""
Debug the D-optimal algorithm to understand why it's not improving
"""

from mixture_designs import MixtureDesign
import numpy as np

# Create same setup as in your Streamlit app
component_names = ['Component_1', 'Component_2', 'Component_3', 'Component_4', 'Component_5', 'Fixed_1', 'Fixed_2', 'Fixed_3']
component_bounds = [(0, 1), (0, 1), (0, 0.1), (0, 0.1), (0, 0.1), (0.05, 0.05), (0.05, 0.05), (0.05, 0.05)]
fixed_components = {'Fixed_1': 0.05, 'Fixed_2': 0.05, 'Fixed_3': 0.05}

mixture = MixtureDesign(
    n_components=8,
    component_names=component_names,
    component_bounds=component_bounds,
    use_parts_mode=True,
    fixed_components=fixed_components
)

print("=== Debugging D-Optimal Algorithm ===\n")

# Let's manually trace through what's happening in the algorithm
np.random.seed(42)

# Generate candidates 
n_candidates = 500  # Smaller for debugging
candidates = mixture._generate_candidate_points(n_candidates)
print(f"Generated {len(candidates)} candidates")

# Check diversity of candidates
distances = []
for i in range(min(50, len(candidates))):
    for j in range(i+1, min(50, len(candidates))):
        dist = np.linalg.norm(candidates[i] - candidates[j])
        distances.append(dist)

print(f"Candidate diversity: min dist = {np.min(distances):.6f}, max dist = {np.max(distances):.6f}")

# Select initial design
n_runs = 25
initial_design = candidates[:n_runs].copy()

print(f"\nInitial design D-efficiency calculation:")
# Calculate D-efficiency with current method
d_eff_initial = mixture._calculate_d_efficiency(initial_design, "quadratic")
print(f"  D-efficiency = {d_eff_initial:.6f}")

# Now let's manually try swapping points and see if D-efficiency changes
print(f"\nTrying manual point swaps:")
for test_idx in range(5):
    # Try swapping first point with a random candidate
    test_design = initial_design.copy()
    swap_candidate_idx = np.random.randint(n_runs, len(candidates))
    test_design[0] = candidates[swap_candidate_idx]
    
    d_eff_test = mixture._calculate_d_efficiency(test_design, "quadratic")
    print(f"  Test {test_idx + 1}: Swap point 0 with candidate {swap_candidate_idx}, D-eff = {d_eff_test:.6f}")

# Let's also check what happens in the adjusted space vs final space
print(f"\n=== Checking space transformations ===")

# Look at a design point before and after post-processing
print(f"\nDesign point before post-processing:")
print(f"  Point 1: {initial_design[0].round(4)}")
print(f"  Sum: {np.sum(initial_design[0]):.6f}")

# Apply post-processing
processed_design = mixture._post_process_design_fixed_components(initial_design.copy())
print(f"\nSame point after post-processing:")
print(f"  Point 1: {processed_design[0].round(4)}")
print(f"  Sum: {np.sum(processed_design[0]):.6f}")

# Calculate D-efficiency on processed design
d_eff_processed = mixture._calculate_d_efficiency(processed_design, "quadratic")
print(f"\nD-efficiency of processed design: {d_eff_processed:.6f}")

# Check the model matrix size
model_matrix = mixture.create_mixture_model_matrix(processed_design, "quadratic")
print(f"\nModel matrix shape: {model_matrix.shape}")
print(f"  (Should have {len(processed_design)} rows and parameters for 8 components)")

# Let's check if the issue is with the fallback value
print(f"\n=== Checking why D-efficiency is stuck at 0.001 ===")

# Temporarily calculate with fixed components excluded to see the difference
mixture.truly_fixed_components = set()  # Temporarily treat all as variable
model_matrix_all = mixture.create_mixture_model_matrix(processed_design, "quadratic")
print(f"Model matrix with all components: {model_matrix_all.shape}")

# Calculate XtX determinant
XtX = np.dot(model_matrix_all.T, model_matrix_all)
print(f"XtX shape: {XtX.shape}")
print(f"XtX condition number: {np.linalg.cond(XtX):.2e}")

try:
    det_XtX = np.linalg.det(XtX)
    print(f"XtX determinant: {det_XtX:.2e}")
except:
    print("XtX determinant: FAILED TO COMPUTE")

# Restore fixed components
mixture.truly_fixed_components = {'Fixed_1', 'Fixed_2', 'Fixed_3'}
