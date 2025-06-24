"""
Debug why coordinate exchange is not finding improvements
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

print("=== Debugging Coordinate Exchange ===\n")

# Generate candidates
np.random.seed(42)
candidates = mixture._generate_candidate_points(100)
print(f"Generated {len(candidates)} candidates")

# Create initial design
n_runs = 10  # Small for debugging
current_design = candidates[:n_runs].copy()

print("\nInitial design (first 3 points):")
for i in range(min(3, n_runs)):
    print(f"  Point {i+1}: {current_design[i].round(4)}")

# Calculate initial D-efficiency
d_eff_initial = mixture._calculate_d_efficiency(current_design, "quadratic")
print(f"\nInitial D-efficiency: {d_eff_initial:.6f}")

# Create model matrix to understand what's happening
print("\nModel matrix analysis:")
model_matrix = mixture.create_mixture_model_matrix(current_design, "quadratic")
print(f"  Shape: {model_matrix.shape}")
print(f"  Rank: {np.linalg.matrix_rank(model_matrix)}")

# Check XtX
XtX = np.dot(model_matrix.T, model_matrix)
print(f"  XtX condition number: {np.linalg.cond(XtX):.2e}")

# Try manual coordinate exchange
print("\n=== Manual Coordinate Exchange Test ===")

# Try swapping first point with different candidates
improvements_found = 0
for test_idx in range(20):
    # Select a candidate that's different from current points
    cand_idx = n_runs + test_idx
    candidate = candidates[cand_idx]
    
    # Create test design
    test_design = current_design.copy()
    test_design[0] = candidate
    
    # Calculate new D-efficiency
    d_eff_test = mixture._calculate_d_efficiency(test_design, "quadratic")
    
    # Check if improved
    improvement = d_eff_test - d_eff_initial
    
    print(f"\nTest {test_idx + 1}:")
    print(f"  Original point: {current_design[0][:5].round(3)} (first 5 components)")
    print(f"  Candidate point: {candidate[:5].round(3)} (first 5 components)")
    print(f"  Original D-eff: {d_eff_initial:.6f}")
    print(f"  New D-eff: {d_eff_test:.6f}")
    print(f"  Improvement: {improvement:.6f}")
    
    if improvement > 1e-8:
        improvements_found += 1
        print("  *** IMPROVEMENT FOUND! ***")
        
        # Check what changed in the model matrix
        new_model_matrix = mixture.create_mixture_model_matrix(test_design, "quadratic")
        new_XtX = np.dot(new_model_matrix.T, new_model_matrix)
        
        old_det = np.linalg.det(XtX)
        new_det = np.linalg.det(new_XtX)
        
        print(f"  Old determinant: {old_det:.2e}")
        print(f"  New determinant: {new_det:.2e}")
        det_ratio = new_det/old_det if old_det != 0 else float('inf')
        if isinstance(det_ratio, float) and not np.isinf(det_ratio):
            print(f"  Det ratio: {det_ratio:.2f}")
        else:
            print(f"  Det ratio: {det_ratio}")
        
        # Update to the better design
        current_design[0] = candidate
        d_eff_initial = d_eff_test
        XtX = new_XtX
    
    if test_idx >= 5 and improvements_found == 0:
        print("\nNo improvements found in first 5 tests. Something is wrong...")
        break

print(f"\n=== Summary ===")
print(f"Total improvements found: {improvements_found} out of {test_idx + 1} tests")

if improvements_found == 0:
    print("\nPOSSIBLE ISSUES:")
    print("1. D-efficiency calculation may be stuck at minimum (0.001)")
    print("2. The design space may be too constrained")
    print("3. The determinant may be zero or near-zero for all designs")
    
    # Check if all D-efficiencies are the same
    all_d_effs = []
    for i in range(min(10, len(candidates) - n_runs)):
        test_design = current_design.copy()
        test_design[0] = candidates[n_runs + i]
        d_eff = mixture._calculate_d_efficiency(test_design, "quadratic")
        all_d_effs.append(d_eff)
    
    unique_d_effs = set(all_d_effs)
    print(f"\nUnique D-efficiency values found: {unique_d_effs}")
    
    if len(unique_d_effs) == 1:
        print("*** ALL D-EFFICIENCIES ARE THE SAME! ***")
        print("This explains why coordinate exchange can't find improvements.")
